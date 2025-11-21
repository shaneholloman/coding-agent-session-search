use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use notify::{recommended_watcher, RecursiveMode, Watcher};

use crate::connectors::NormalizedConversation;
use crate::connectors::{
    Connector, amp::AmpConnector, claude_code::ClaudeCodeConnector, cline::ClineConnector,
    codex::CodexConnector, gemini::GeminiConnector, opencode::OpenCodeConnector,
};
use crate::search::tantivy::{TantivyIndex, index_dir};
use crate::storage::sqlite::SqliteStorage;

pub struct IndexOptions {
    pub full: bool,
    pub watch: bool,
    pub db_path: PathBuf,
    pub data_dir: PathBuf,
}

pub fn run_index(opts: IndexOptions) -> Result<()> {
    let mut storage = SqliteStorage::open(&opts.db_path)?;
    let mut t_index = TantivyIndex::open_or_create(&index_dir(&opts.data_dir)?)?;

    let connectors: Vec<Box<dyn Connector>> = vec![
        Box::new(CodexConnector::new()),
        Box::new(ClineConnector::new()),
        Box::new(GeminiConnector::new()),
        Box::new(ClaudeCodeConnector::new()),
        Box::new(OpenCodeConnector::new()),
        Box::new(AmpConnector::new()),
    ];

    for conn in connectors {
        let detect = conn.detect();
        if !detect.detected {
            continue;
        }
        let ctx = crate::connectors::ScanContext {
            data_root: opts.data_dir.clone(),
            since_ts: None,
        };
        let convs = conn.scan(&ctx)?;
        ingest_batch(&mut storage, &mut t_index, convs)?;
    }

    t_index.commit()?;

    if opts.watch {
        watch_sources(|| {
            let _ = run_index(IndexOptions {
                watch: false,
                ..opts.clone()
            });
        })?;
    }

    Ok(())
}

fn ingest_batch(
    storage: &mut SqliteStorage,
    t_index: &mut TantivyIndex,
    convs: Vec<NormalizedConversation>,
) -> Result<()> {
    for conv in convs {
        super::persist::persist_conversation(storage, t_index, &conv)?;
    }
    Ok(())
}

fn watch_sources<F: Fn() + Send + 'static>(callback: F) -> Result<()> {
    let mut watcher = recommended_watcher(move |res| {
        if res.is_ok() {
            callback();
        }
    })?;

    for dir in watch_roots() {
        let _ = watcher.watch(&dir, RecursiveMode::Recursive);
    }

    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}

fn watch_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    roots.push(
        std::env::var("CODEX_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".codex/sessions")),
    );
    roots.push(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".config/Code/User/globalStorage/saoudrizwan.claude-dev"),
    );
    roots.push(dirs::home_dir().unwrap_or_default().join(".gemini/tmp"));
    roots.push(
        dirs::home_dir()
            .unwrap_or_default()
            .join(".claude/projects"),
    );
    roots
}

pub mod persist {
    use anyhow::Result;

    use crate::connectors::NormalizedConversation;
    use crate::model::types::{Agent, AgentKind, Conversation, Message, MessageRole};
    use crate::search::tantivy::TantivyIndex;
    use crate::storage::sqlite::SqliteStorage;

    pub fn persist_conversation(
        storage: &mut SqliteStorage,
        t_index: &mut TantivyIndex,
        conv: &NormalizedConversation,
    ) -> Result<()> {
        let agent = Agent {
            slug: conv.agent_slug.clone(),
            name: conv.agent_slug.clone(),
            version: None,
            kind: AgentKind::Cli,
        };
        let agent_id = storage.ensure_agent(&agent)?;

        let workspace_id = if let Some(ws) = &conv.workspace {
            Some(storage.ensure_workspace(ws, None)?)
        } else {
            None
        };

        let messages = conv
            .messages
            .iter()
            .map(|m| Message {
                idx: m.idx,
                role: map_role(&m.role),
                author: m.author.clone(),
                created_at: m.created_at,
                content: m.content.clone(),
                extra_json: m.extra.clone(),
                snippets: Vec::new(),
            })
            .collect();

        let conversation = Conversation {
            id: 0,
            agent_slug: conv.agent_slug.clone(),
            workspace: conv.workspace.clone(),
            external_id: conv.external_id.clone(),
            title: conv.title.clone(),
            source_path: conv.source_path.clone(),
            started_at: conv.started_at,
            ended_at: conv.ended_at,
            approx_tokens: None,
            metadata_json: conv.metadata.clone(),
            messages,
        };

        let _ = storage.insert_conversation_tree(agent_id, workspace_id, &conversation)?;
        t_index.add_conversation(conv)?;
        Ok(())
    }

    fn map_role(role: &str) -> MessageRole {
        match role {
            "user" => MessageRole::User,
            "assistant" | "agent" => MessageRole::Agent,
            "tool" => MessageRole::Tool,
            "system" => MessageRole::System,
            other => MessageRole::Other(other.to_string()),
        }
    }
}
