//! Embedder trait and types for semantic search.
//!
//! This module defines the [`Embedder`] trait that all embedding implementations must satisfy.
//! The trait abstraction allows transparent embedder swapping, which is critical for the
//! consent-gated download flow where we start with a hash-based embedder and upgrade to
//! ML when the model is ready.
//!
//! # Implementations
//!
//! Two embedder types are planned:
//! - **Hash embedder**: Uses FNV-1a feature hashing for fast, deterministic embeddings
//!   without external dependencies. Always available.
//! - **ML embedder**: Uses FastEmbed with the MiniLM model for semantic embeddings.
//!   Requires model download with user consent.
//!
//! # Example
//!
//! ```ignore
//! use crate::search::embedder::{Embedder, EmbedderError};
//!
//! fn search_with_embedder(embedder: &dyn Embedder, query: &str) -> Result<(), EmbedderError> {
//!     let embedding = embedder.embed(query)?;
//!     println!("Embedding dimension: {}", embedding.len());
//!     println!("Embedder: {} (semantic: {})", embedder.id(), embedder.is_semantic());
//!     Ok(())
//! }
//! ```

use std::fmt;

/// Error type for embedder operations.
#[derive(Debug)]
pub enum EmbedderError {
    /// The embedder is not available (e.g., model not downloaded).
    Unavailable(String),
    /// Failed to embed the input text.
    EmbeddingFailed(String),
    /// Input text is empty or invalid.
    InvalidInput(String),
    /// Internal error in the embedder.
    Internal(String),
}

impl fmt::Display for EmbedderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbedderError::Unavailable(msg) => write!(f, "embedder unavailable: {msg}"),
            EmbedderError::EmbeddingFailed(msg) => write!(f, "embedding failed: {msg}"),
            EmbedderError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            EmbedderError::Internal(msg) => write!(f, "internal error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedderError {}

/// Result type for embedder operations.
pub type EmbedderResult<T> = Result<T, EmbedderError>;

/// Trait for text embedding implementations.
///
/// All embedders must produce fixed-dimension vectors suitable for similarity search.
/// Vectors should be normalized to unit length (L2 norm ≈ 1.0) for consistent
/// cosine similarity computation.
///
/// # Implementors
///
/// - `HashEmbedder`: FNV-1a feature hashing (always available, ~256 dimensions)
/// - `FastEmbedder`: MiniLM via FastEmbed (requires model download, 384 dimensions)
///
/// # Thread Safety
///
/// Implementations should be `Send + Sync` to allow use across threads.
pub trait Embedder: Send + Sync {
    /// Embed a single text into a vector.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to embed. Should be canonicalized (lowercase, whitespace
    ///   normalized) before calling for best results.
    ///
    /// # Returns
    ///
    /// A vector of `f32` values with length equal to [`dimension()`](Self::dimension).
    /// The vector is normalized to approximately unit length.
    ///
    /// # Errors
    ///
    /// - [`EmbedderError::InvalidInput`] if the text is empty.
    /// - [`EmbedderError::Unavailable`] if the embedder is not ready (e.g., model not downloaded).
    /// - [`EmbedderError::EmbeddingFailed`] if embedding fails for any other reason.
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>>;

    /// Embed multiple texts in a batch.
    ///
    /// Batch embedding can be significantly faster than calling [`embed()`](Self::embed)
    /// repeatedly, especially for ML embedders that can parallelize inference.
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of texts to embed. Each should be canonicalized.
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one per input text, in the same order as the input.
    /// Each embedding has length equal to [`dimension()`](Self::dimension).
    ///
    /// # Errors
    ///
    /// - [`EmbedderError::InvalidInput`] if any text is empty (all-or-nothing).
    /// - [`EmbedderError::Unavailable`] if the embedder is not ready.
    /// - [`EmbedderError::EmbeddingFailed`] if embedding fails.
    ///
    /// # Default Implementation
    ///
    /// The default implementation calls [`embed()`](Self::embed) for each text.
    /// Implementations should override this for batch-optimized inference.
    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// The output dimension of this embedder.
    ///
    /// All embeddings produced by this embedder will have exactly this many components.
    /// Common dimensions:
    /// - Hash embedder: 256
    /// - MiniLM: 384
    /// - Other models: varies
    fn dimension(&self) -> usize;

    /// Unique identifier for this embedder.
    ///
    /// This ID is stored in the vector index header to detect when the embedder changes.
    /// Format: `{type}-{dimension}[-{version}]`
    ///
    /// # Examples
    ///
    /// - `"fnv1a-256"` for hash embedder
    /// - `"minilm-384"` for MiniLM
    /// - `"minilm-384-v2"` for a newer version
    fn id(&self) -> &str;

    /// Whether this is a true semantic embedder (ML-based).
    ///
    /// Returns `true` for ML embedders that capture semantic meaning.
    /// Returns `false` for hash-based embedders that only capture lexical features.
    ///
    /// This is used by the TUI to display the appropriate status indicator
    /// and by the search layer to decide on hybrid vs. pure lexical search.
    fn is_semantic(&self) -> bool;
}

/// Metadata about an embedder for display and logging.
#[derive(Debug, Clone)]
pub struct EmbedderInfo {
    /// The embedder's unique identifier.
    pub id: String,
    /// The output dimension.
    pub dimension: usize,
    /// Whether it's a semantic (ML) embedder.
    pub is_semantic: bool,
}

impl EmbedderInfo {
    /// Create info from an embedder instance.
    pub fn from_embedder(embedder: &dyn Embedder) -> Self {
        Self {
            id: embedder.id().to_string(),
            dimension: embedder.dimension(),
            is_semantic: embedder.is_semantic(),
        }
    }
}

impl fmt::Display for EmbedderInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_semantic {
            "semantic"
        } else {
            "lexical"
        };
        write!(f, "{} ({}, {} dims)", self.id, kind, self.dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock embedder for testing.
    struct MockEmbedder {
        dimension: usize,
        is_semantic: bool,
    }

    impl Embedder for MockEmbedder {
        fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
            if text.is_empty() {
                return Err(EmbedderError::InvalidInput("empty text".to_string()));
            }
            // Generate a deterministic fake embedding based on text length
            Ok(vec![text.len() as f32 / 100.0; self.dimension])
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn id(&self) -> &str {
            if self.is_semantic {
                "mock-semantic-384"
            } else {
                "mock-hash-256"
            }
        }

        fn is_semantic(&self) -> bool {
            self.is_semantic
        }
    }

    #[test]
    fn test_embedder_trait_basic() {
        let embedder = MockEmbedder {
            dimension: 256,
            is_semantic: false,
        };

        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 256);
        assert_eq!(embedder.id(), "mock-hash-256");
        assert!(!embedder.is_semantic());
    }

    #[test]
    fn test_embedder_trait_semantic() {
        let embedder = MockEmbedder {
            dimension: 384,
            is_semantic: true,
        };

        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.id(), "mock-semantic-384");
        assert!(embedder.is_semantic());
    }

    #[test]
    fn test_embedder_batch() {
        let embedder = MockEmbedder {
            dimension: 256,
            is_semantic: false,
        };

        let texts = &["hello", "world", "test"];
        let embeddings = embedder.embed_batch(texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 256);
        }
    }

    #[test]
    fn test_embedder_empty_input_error() {
        let embedder = MockEmbedder {
            dimension: 256,
            is_semantic: false,
        };

        let result = embedder.embed("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EmbedderError::InvalidInput(_)
        ));
    }

    #[test]
    fn test_embedder_info() {
        let embedder = MockEmbedder {
            dimension: 384,
            is_semantic: true,
        };

        let info = EmbedderInfo::from_embedder(&embedder);
        assert_eq!(info.id, "mock-semantic-384");
        assert_eq!(info.dimension, 384);
        assert!(info.is_semantic);

        let display = format!("{info}");
        assert!(display.contains("mock-semantic-384"));
        assert!(display.contains("semantic"));
        assert!(display.contains("384"));
    }

    #[test]
    fn test_embedder_error_display() {
        let err = EmbedderError::Unavailable("model not downloaded".to_string());
        assert!(err.to_string().contains("unavailable"));
        assert!(err.to_string().contains("model not downloaded"));

        let err = EmbedderError::EmbeddingFailed("inference error".to_string());
        assert!(err.to_string().contains("embedding failed"));

        let err = EmbedderError::InvalidInput("empty".to_string());
        assert!(err.to_string().contains("invalid input"));

        let err = EmbedderError::Internal("panic".to_string());
        assert!(err.to_string().contains("internal error"));
    }
}
