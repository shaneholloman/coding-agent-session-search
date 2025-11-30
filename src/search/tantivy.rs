use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use tantivy::schema::*;
use tantivy::{Index, IndexReader, IndexWriter, doc};
use tracing::{debug, info, warn};

use crate::connectors::NormalizedConversation;

const SCHEMA_VERSION: &str = "v4";

/// Minimum time (ms) between merge operations
const MERGE_COOLDOWN_MS: i64 = 300_000; // 5 minutes

/// Segment count threshold above which merge is triggered
const MERGE_SEGMENT_THRESHOLD: usize = 4;

/// Global last merge timestamp (ms since epoch)
static LAST_MERGE_TS: AtomicI64 = AtomicI64::new(0);

/// Debug status for segment merge operations
#[derive(Debug, Clone)]
pub struct MergeStatus {
    /// Current number of searchable segments
    pub segment_count: usize,
    /// Timestamp of last merge (ms since epoch), 0 if never
    pub last_merge_ts: i64,
    /// Milliseconds since last merge, -1 if never merged
    pub ms_since_last_merge: i64,
    /// Segment count threshold for auto-merge
    pub merge_threshold: usize,
    /// Cooldown period between merges (ms)
    pub cooldown_ms: i64,
}

impl MergeStatus {
    /// Returns true if merge is recommended based on current status
    pub fn should_merge(&self) -> bool {
        self.segment_count >= self.merge_threshold
            && (self.ms_since_last_merge < 0 || self.ms_since_last_merge >= self.cooldown_ms)
    }
}

// Bump this when schema/tokenizer changes. Used to trigger rebuilds.
pub const SCHEMA_HASH: &str = "tantivy-schema-v4-edge-ngram-preview";

#[derive(Clone, Copy)]
pub struct Fields {
    pub agent: Field,
    pub workspace: Field,
    pub source_path: Field,
    pub msg_idx: Field,
    pub created_at: Field,
    pub title: Field,
    pub content: Field,
    pub title_prefix: Field,
    pub content_prefix: Field,
    pub preview: Field,
}

pub struct TantivyIndex {
    pub index: Index,
    writer: IndexWriter,
    pub fields: Fields,
}

impl TantivyIndex {
    pub fn open_or_create(path: &Path) -> Result<Self> {
        let schema = build_schema();
        std::fs::create_dir_all(path)?;

        let meta_path = path.join("schema_hash.json");
        let mut needs_rebuild = true;
        if meta_path.exists() {
            let meta = std::fs::read_to_string(&meta_path)?;
            if meta.contains(SCHEMA_HASH) {
                needs_rebuild = false;
            }
        }

        if needs_rebuild {
            // Recreate index directory completely to avoid stale lock files.
            let _ = std::fs::remove_dir_all(path);
            std::fs::create_dir_all(path)?;
        }

        let mut index = if path.join("meta.json").exists() && !needs_rebuild {
            Index::open_in_dir(path)?
        } else {
            Index::create_in_dir(path, schema.clone())?
        };

        ensure_tokenizer(&mut index);

        std::fs::write(
            &meta_path,
            format!("{{\"schema_hash\":\"{}\"}}", SCHEMA_HASH),
        )?;

        let writer = index
            .writer(50_000_000)
            .map_err(|e| anyhow!("create index writer: {e:?}"))?;
        let fields = fields_from_schema(&schema)?;
        Ok(Self {
            index,
            writer,
            fields,
        })
    }

    pub fn add_conversation(&mut self, conv: &NormalizedConversation) -> Result<()> {
        self.add_messages(conv, &conv.messages)
    }

    pub fn delete_all(&mut self) -> Result<()> {
        self.writer.delete_all_documents()?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        self.writer.commit()?;
        Ok(())
    }

    pub fn reader(&self) -> Result<IndexReader> {
        Ok(self.index.reader()?)
    }

    /// Get current number of searchable segments
    pub fn segment_count(&self) -> usize {
        self.index
            .searchable_segment_ids()
            .map(|ids| ids.len())
            .unwrap_or(0)
    }

    /// Returns debug info about merge status
    pub fn merge_status(&self) -> MergeStatus {
        let last_merge_ts = LAST_MERGE_TS.load(Ordering::Relaxed);
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let ms_since_last = if last_merge_ts > 0 {
            now_ms - last_merge_ts
        } else {
            -1 // never merged
        };
        MergeStatus {
            segment_count: self.segment_count(),
            last_merge_ts,
            ms_since_last_merge: ms_since_last,
            merge_threshold: MERGE_SEGMENT_THRESHOLD,
            cooldown_ms: MERGE_COOLDOWN_MS,
        }
    }

    /// Attempt to merge segments if idle conditions are met.
    /// Returns Ok(true) if merge was triggered, Ok(false) if skipped.
    /// Merge runs in background thread - this call is non-blocking.
    pub fn optimize_if_idle(&mut self) -> Result<bool> {
        let segment_ids = self.index.searchable_segment_ids()?;
        let segment_count = segment_ids.len();

        // Check if we have enough segments to warrant a merge
        if segment_count < MERGE_SEGMENT_THRESHOLD {
            debug!(
                segments = segment_count,
                threshold = MERGE_SEGMENT_THRESHOLD,
                "Skipping merge: segment count below threshold"
            );
            return Ok(false);
        }

        // Check cooldown period
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let last_merge = LAST_MERGE_TS.load(Ordering::Relaxed);
        if last_merge > 0 && (now_ms - last_merge) < MERGE_COOLDOWN_MS {
            debug!(
                ms_since_last = now_ms - last_merge,
                cooldown = MERGE_COOLDOWN_MS,
                "Skipping merge: cooldown period active"
            );
            return Ok(false);
        }

        // Trigger merge - this runs asynchronously in Tantivy's merge thread pool
        info!(
            segments = segment_count,
            "Starting background segment merge"
        );

        // merge() returns a FutureResult that runs async; we drop it to let it run in background
        // The merge will complete when Tantivy's internal thread pool processes it
        let _merge_future = self.writer.merge(&segment_ids);
        LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
        info!("Segment merge initiated (running in background)");
        Ok(true)
    }

    /// Force immediate segment merge and wait for completion.
    /// Use sparingly - blocks until merge finishes.
    pub fn force_merge(&mut self) -> Result<()> {
        let segment_ids = self.index.searchable_segment_ids()?;
        if segment_ids.is_empty() {
            return Ok(());
        }
        info!(segments = segment_ids.len(), "Force merging all segments");
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Start merge and wait for completion
        let merge_future = self.writer.merge(&segment_ids);
        match merge_future.wait() {
            Ok(_) => {
                LAST_MERGE_TS.store(now_ms, Ordering::Relaxed);
                info!("Force merge completed");
                Ok(())
            }
            Err(e) => {
                warn!(error = %e, "Force merge failed");
                Err(anyhow!("merge failed: {e}"))
            }
        }
    }

    pub fn add_messages(
        &mut self,
        conv: &NormalizedConversation,
        messages: &[crate::connectors::NormalizedMessage],
    ) -> Result<()> {
        for msg in messages {
            let mut d = doc! {
                self.fields.agent => conv.agent_slug.clone(),
                self.fields.source_path => conv.source_path.to_string_lossy().into_owned(),
                self.fields.msg_idx => msg.idx as u64,
                self.fields.content => msg.content.clone(),
            };
            if let Some(ws) = &conv.workspace {
                d.add_text(self.fields.workspace, ws.to_string_lossy());
            }
            if let Some(ts) = msg.created_at.or(conv.started_at) {
                d.add_i64(self.fields.created_at, ts);
            }
            if let Some(title) = &conv.title {
                d.add_text(self.fields.title, title);
                d.add_text(self.fields.title_prefix, generate_edge_ngrams(title));
            }
            d.add_text(
                self.fields.content_prefix,
                generate_edge_ngrams(&msg.content),
            );
            d.add_text(self.fields.preview, build_preview(&msg.content, 200));
            self.writer.add_document(d)?;
        }
        Ok(())
    }
}

fn generate_edge_ngrams(text: &str) -> String {
    let mut ngrams = String::with_capacity(text.len() * 2);
    // Split by non-alphanumeric characters to identify words
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 2 {
            continue;
        }
        // Generate edge ngrams of length 2..=20 (or word length)
        for len in 2..=chars.len().min(20) {
            if !ngrams.is_empty() {
                ngrams.push(' ');
            }
            ngrams.extend(chars[0..len].iter());
        }
    }
    ngrams
}

pub fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    let text = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("hyphen_normalize")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();

    let text_not_stored = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("hyphen_normalize")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );

    schema_builder.add_text_field("agent", TEXT | STORED);
    schema_builder.add_text_field("workspace", STRING | STORED);
    schema_builder.add_text_field("source_path", STORED);
    schema_builder.add_u64_field("msg_idx", INDEXED | STORED);
    schema_builder.add_i64_field("created_at", INDEXED | STORED | FAST);
    schema_builder.add_text_field("title", text.clone());
    schema_builder.add_text_field("content", text);
    schema_builder.add_text_field("title_prefix", text_not_stored.clone());
    schema_builder.add_text_field("content_prefix", text_not_stored);
    schema_builder.add_text_field("preview", TEXT | STORED);
    schema_builder.build()
}

pub fn fields_from_schema(schema: &Schema) -> Result<Fields> {
    let get = |name: &str| {
        schema
            .get_field(name)
            .map_err(|_| anyhow!("schema missing {}", name))
    };
    Ok(Fields {
        agent: get("agent")?,
        workspace: get("workspace")?,
        source_path: get("source_path")?,
        msg_idx: get("msg_idx")?,
        created_at: get("created_at")?,
        title: get("title")?,
        content: get("content")?,
        title_prefix: get("title_prefix")?,
        content_prefix: get("content_prefix")?,
        preview: get("preview")?,
    })
}

fn build_preview(content: &str, max_chars: usize) -> String {
    let char_count = content.chars().count();
    if char_count <= max_chars {
        return content.to_string();
    }
    let mut out = String::new();
    for ch in content.chars().take(max_chars) {
        out.push(ch);
    }
    out.push('…');
    out
}

pub fn index_dir(base: &Path) -> Result<std::path::PathBuf> {
    let dir = base.join("index").join(SCHEMA_VERSION);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

pub fn ensure_tokenizer(index: &mut Index) {
    use tantivy::tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer};
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(RemoveLongFilter::limit(40))
        .build();
    index.tokenizers().register("hyphen_normalize", analyzer);
}
