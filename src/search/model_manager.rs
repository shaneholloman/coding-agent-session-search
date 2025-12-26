//! Semantic model management (local-only detection).
//!
//! This module wires the FastEmbed MiniLM embedder into semantic search by:
//! - validating the local model files
//! - loading the vector index
//! - building filter maps from the SQLite database
//!
//! It does **not** download models. Missing files are surfaced as availability
//! states so the UI can guide the user.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::search::embedder::Embedder;
use crate::search::fastembed_embedder::FastEmbedder;
use crate::search::vector_index::{
    ROLE_ASSISTANT, ROLE_USER, SemanticFilterMaps, VectorIndex, vector_index_path,
};
use crate::storage::sqlite::SqliteStorage;

#[derive(Debug, Clone)]
pub enum SemanticAvailability {
    Ready {
        embedder_id: String,
    },
    ModelMissing {
        model_dir: PathBuf,
        missing_files: Vec<String>,
    },
    IndexMissing {
        index_path: PathBuf,
    },
    DatabaseUnavailable {
        db_path: PathBuf,
        error: String,
    },
    LoadFailed {
        context: String,
    },
}

impl SemanticAvailability {
    pub fn is_ready(&self) -> bool {
        matches!(self, SemanticAvailability::Ready { .. })
    }

    pub fn summary(&self) -> String {
        match self {
            SemanticAvailability::Ready { embedder_id } => {
                format!("semantic ready ({embedder_id})")
            }
            SemanticAvailability::ModelMissing { model_dir, .. } => {
                format!("model missing at {}", model_dir.display())
            }
            SemanticAvailability::IndexMissing { index_path } => {
                format!("vector index missing at {}", index_path.display())
            }
            SemanticAvailability::DatabaseUnavailable { error, .. } => {
                format!("db unavailable ({error})")
            }
            SemanticAvailability::LoadFailed { context } => {
                format!("semantic load failed ({context})")
            }
        }
    }
}

pub struct SemanticContext {
    pub embedder: Arc<dyn Embedder>,
    pub index: VectorIndex,
    pub filter_maps: SemanticFilterMaps,
    pub roles: Option<HashSet<u8>>,
}

pub struct SemanticSetup {
    pub availability: SemanticAvailability,
    pub context: Option<SemanticContext>,
}

pub fn load_semantic_context(data_dir: &Path, db_path: &Path) -> SemanticSetup {
    let model_dir = FastEmbedder::default_model_dir(data_dir);
    let missing_files = FastEmbedder::required_model_files()
        .iter()
        .filter(|name| !model_dir.join(*name).is_file())
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();

    if !missing_files.is_empty() {
        return SemanticSetup {
            availability: SemanticAvailability::ModelMissing {
                model_dir,
                missing_files,
            },
            context: None,
        };
    }

    let index_path = vector_index_path(data_dir, FastEmbedder::embedder_id_static());
    if !index_path.is_file() {
        return SemanticSetup {
            availability: SemanticAvailability::IndexMissing { index_path },
            context: None,
        };
    }

    let storage = match SqliteStorage::open_readonly(db_path) {
        Ok(storage) => storage,
        Err(err) => {
            return SemanticSetup {
                availability: SemanticAvailability::DatabaseUnavailable {
                    db_path: db_path.to_path_buf(),
                    error: err.to_string(),
                },
                context: None,
            };
        }
    };

    let filter_maps = match SemanticFilterMaps::from_storage(&storage) {
        Ok(maps) => maps,
        Err(err) => {
            return SemanticSetup {
                availability: SemanticAvailability::LoadFailed {
                    context: format!("filter maps: {err}"),
                },
                context: None,
            };
        }
    };

    let index = match VectorIndex::load(&index_path) {
        Ok(index) => index,
        Err(err) => {
            return SemanticSetup {
                availability: SemanticAvailability::LoadFailed {
                    context: format!("vector index: {err}"),
                },
                context: None,
            };
        }
    };

    let embedder = match FastEmbedder::load_from_dir(&model_dir) {
        Ok(embedder) => Arc::new(embedder) as Arc<dyn Embedder>,
        Err(err) => {
            return SemanticSetup {
                availability: SemanticAvailability::LoadFailed {
                    context: format!("model load: {err}"),
                },
                context: None,
            };
        }
    };

    let roles = Some(HashSet::from([ROLE_USER, ROLE_ASSISTANT]));

    SemanticSetup {
        availability: SemanticAvailability::Ready {
            embedder_id: embedder.id().to_string(),
        },
        context: Some(SemanticContext {
            embedder,
            index,
            filter_maps,
            roles,
        }),
    }
}
