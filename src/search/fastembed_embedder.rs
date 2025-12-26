//! FastEmbed-based ML embedder (MiniLM).
//!
//! Loads a local ONNX model + tokenizer bundle and produces semantic embeddings.
//! This implementation never downloads model assets; it expects the model files
//! to be present on disk and returns a clear error when they are missing.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};

use super::embedder::{Embedder, EmbedderError, EmbedderResult};

const MODEL_ID: &str = "all-minilm-l6-v2";
const MODEL_DIR_NAME: &str = "all-MiniLM-L6-v2";
const EMBEDDER_ID: &str = "minilm-384";
const EMBEDDING_DIMENSION: usize = 384;

const MODEL_FILE: &str = "model.onnx";
const TOKENIZER_JSON: &str = "tokenizer.json";
const CONFIG_JSON: &str = "config.json";
const SPECIAL_TOKENS_JSON: &str = "special_tokens_map.json";
const TOKENIZER_CONFIG_JSON: &str = "tokenizer_config.json";

/// FastEmbed-backed semantic embedder using MiniLM.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
    id: String,
    model_id: String,
    dimension: usize,
}

impl FastEmbedder {
    /// Stable embedder identifier for MiniLM (matches vector index naming).
    pub fn embedder_id_static() -> &'static str {
        EMBEDDER_ID
    }

    /// Stable model identifier for MiniLM.
    pub fn model_id_static() -> &'static str {
        MODEL_ID
    }

    /// Required model files for MiniLM (must all exist locally).
    pub fn required_model_files() -> &'static [&'static str] {
        &[
            MODEL_FILE,
            TOKENIZER_JSON,
            CONFIG_JSON,
            SPECIAL_TOKENS_JSON,
            TOKENIZER_CONFIG_JSON,
        ]
    }

    /// Default model directory relative to the cass data dir.
    pub fn default_model_dir(data_dir: &Path) -> PathBuf {
        data_dir.join("models").join(MODEL_DIR_NAME)
    }

    /// Load the MiniLM model + tokenizer from a local directory.
    ///
    /// This never downloads; it returns `EmbedderError::Unavailable` if any
    /// required file is missing.
    pub fn load_from_dir(model_dir: &Path) -> EmbedderResult<Self> {
        if !model_dir.is_dir() {
            return Err(EmbedderError::Unavailable(format!(
                "model directory not found: {}",
                model_dir.display()
            )));
        }

        let required = Self::required_model_files();
        let mut missing = Vec::new();
        for name in required {
            let path = model_dir.join(name);
            if !path.is_file() {
                missing.push(*name);
            }
        }
        if !missing.is_empty() {
            return Err(EmbedderError::Unavailable(format!(
                "model files missing in {}: {}",
                model_dir.display(),
                missing.join(", ")
            )));
        }

        let model_file = Self::read_required(model_dir.join(MODEL_FILE), MODEL_FILE)?;
        let tokenizer_file = Self::read_required(model_dir.join(TOKENIZER_JSON), TOKENIZER_JSON)?;
        let config_file = Self::read_required(model_dir.join(CONFIG_JSON), CONFIG_JSON)?;
        let special_tokens_map_file =
            Self::read_required(model_dir.join(SPECIAL_TOKENS_JSON), SPECIAL_TOKENS_JSON)?;
        let tokenizer_config_file =
            Self::read_required(model_dir.join(TOKENIZER_CONFIG_JSON), TOKENIZER_CONFIG_JSON)?;

        let tokenizer_files = TokenizerFiles {
            tokenizer_file,
            config_file,
            special_tokens_map_file,
            tokenizer_config_file,
        };

        let mut model = UserDefinedEmbeddingModel::new(model_file, tokenizer_files);
        model.pooling = Some(Pooling::Mean);

        let init_options = InitOptionsUserDefined::new();

        let model = TextEmbedding::try_new_from_user_defined(model, init_options)
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("fastembed init failed: {e}")))?;

        Ok(Self {
            model: Mutex::new(model),
            id: EMBEDDER_ID.to_string(),
            model_id: MODEL_ID.to_string(),
            dimension: EMBEDDING_DIMENSION,
        })
    }

    /// Stable model identifier for compatibility checks.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    fn read_required(path: PathBuf, label: &str) -> EmbedderResult<Vec<u8>> {
        fs::read(&path).map_err(|e| {
            EmbedderError::Unavailable(format!("unable to read {label} at {}: {e}", path.display()))
        })
    }

    fn normalize_in_place(embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        let mut model = self
            .model
            .lock()
            .map_err(|_| EmbedderError::Internal("fastembed lock poisoned".to_string()))?;

        let embeddings = model
            .embed(vec![text], None)
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("fastembed embed failed: {e}")))?;

        let mut embedding = embeddings.into_iter().next().ok_or_else(|| {
            EmbedderError::EmbeddingFailed("fastembed returned no embedding".to_string())
        })?;

        if embedding.len() != self.dimension {
            return Err(EmbedderError::EmbeddingFailed(format!(
                "fastembed dimension mismatch: expected {}, got {}",
                self.dimension,
                embedding.len()
            )));
        }

        Self::normalize_in_place(&mut embedding);
        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        for text in texts {
            if text.is_empty() {
                return Err(EmbedderError::InvalidInput(
                    "empty text in batch".to_string(),
                ));
            }
        }

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut model = self
            .model
            .lock()
            .map_err(|_| EmbedderError::Internal("fastembed lock poisoned".to_string()))?;

        let inputs = texts.to_vec();
        let mut embeddings = model
            .embed(inputs, None)
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("fastembed embed failed: {e}")))?;

        for embedding in embeddings.iter_mut() {
            if embedding.len() != self.dimension {
                return Err(EmbedderError::EmbeddingFailed(format!(
                    "fastembed dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                )));
            }
            Self::normalize_in_place(embedding);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn is_semantic(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fastembed_missing_files_returns_unavailable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let err = match FastEmbedder::load_from_dir(tmp.path()) {
            Ok(_) => panic!("expected missing-model error"),
            Err(err) => err,
        };
        match err {
            EmbedderError::Unavailable(msg) => {
                assert!(msg.contains("model files missing"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
