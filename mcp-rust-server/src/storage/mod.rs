use crate::models::{Memory, MemoryQueryResult};
use crate::embeddings::EmbeddingGenerator;
use async_trait::async_trait;
use anyhow::Result;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

// Export ChromaDB storage implementation
mod chroma;
pub use chroma::ChromaMemoryStorage;

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("ChromaDB client error: {0}")]
    ClientError(String),
    #[error("Duplicate content hash: {0}")]
    DuplicateError(String),
    #[error("Memory not found: {0}")]
    NotFoundError(String),
    #[error("Metadata serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Database operation failed: {0}")]
    OperationFailed(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[async_trait]
pub trait MemoryStorage: Send + Sync {
    async fn store(&self, memory: &Memory) -> Result<(bool, String)>; // success, message
    async fn retrieve(&self, query_embedding: &Vec<f32>, n_results: usize) -> Result<Vec<MemoryQueryResult>>;
    async fn search_by_tag(&self, tags: &[String]) -> Result<Vec<Memory>>;
    async fn delete(&self, content_hash: &str) -> Result<(bool, String)>; // success, message
    async fn check_duplicate_exists(&self, content_hash: &str) -> Result<bool>;
}

// A simple in-memory implementation for the MVP
pub struct InMemoryStorage {
    memories: Arc<Mutex<HashMap<String, Memory>>>,
    embedding_generator: Arc<dyn EmbeddingGenerator>,
}

impl InMemoryStorage {
    pub fn new(embedding_generator: Arc<dyn EmbeddingGenerator>) -> Self {
        Self {
            memories: Arc::new(Mutex::new(HashMap::new())),
            embedding_generator,
        }
    }

    // Helper function to calculate cosine similarity between two embeddings
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a > 0.0 && magnitude_b > 0.0 {
            dot_product / (magnitude_a * magnitude_b)
        } else {
            0.0
        }
    }
}

#[async_trait]
impl MemoryStorage for InMemoryStorage {
    async fn check_duplicate_exists(&self, content_hash: &str) -> Result<bool> {
        let memories = self.memories.lock().await;
        Ok(memories.contains_key(content_hash))
    }

    async fn store(&self, memory: &Memory) -> Result<(bool, String)> {
        if self.check_duplicate_exists(&memory.content_hash).await? {
            return Ok((false, "Duplicate content detected".to_string()));
        }

        let content_hash = memory.content_hash.clone();
        
        // Generate embedding if not already present
        let mut memory_to_store = memory.clone();
        if memory_to_store.embedding.is_none() {
            memory_to_store.embedding = Some(self.embedding_generator.generate_embedding(&memory_to_store.content).await?);
        }
        
        // Store memory
        let mut memories = self.memories.lock().await;
        memories.insert(content_hash.clone(), memory_to_store);

        Ok((true, format!("Successfully stored memory with hash: {}", content_hash)))
    }

    async fn retrieve(&self, query_embedding: &Vec<f32>, n_results: usize) -> Result<Vec<MemoryQueryResult>> {
        let memories = self.memories.lock().await;
        
        // Calculate similarity scores for all memories
        let mut results: Vec<MemoryQueryResult> = Vec::new();
        for memory in memories.values() {
            if let Some(memory_embedding) = &memory.embedding {
                let score = Self::cosine_similarity(query_embedding, memory_embedding);
                results.push(MemoryQueryResult {
                    memory: memory.clone(),
                    relevance_score: score,
                });
            }
        }

        // Sort by relevance score (descending)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

        // Return top n results
        Ok(results.into_iter().take(n_results).collect())
    }

    async fn search_by_tag(&self, tags: &[String]) -> Result<Vec<Memory>> {
        let memories = self.memories.lock().await;
        
        let matching_memories: Vec<Memory> = memories
            .values()
            .filter(|memory| memory.tags.iter().any(|tag| tags.contains(tag)))
            .cloned()
            .collect();

        Ok(matching_memories)
    }

    async fn delete(&self, content_hash: &str) -> Result<(bool, String)> {
        let mut memories = self.memories.lock().await;
        
        if memories.remove(content_hash).is_some() {
            Ok((true, format!("Successfully deleted memory with hash: {}", content_hash)))
        } else {
            Ok((false, format!("No memory found with hash: {}", content_hash)))
        }
    }
}
