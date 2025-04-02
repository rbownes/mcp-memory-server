use crate::models::{Memory, MemoryQueryResult};
use crate::embeddings::EmbeddingGenerator;
use super::MemoryStorage;
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::{collections::HashMap, sync::Arc, path::Path};
use reqwest::Client;
use url::Url;

/// ChromaDB storage implementation
pub struct ChromaMemoryStorage {
    client: Client,
    base_url: Url,
    collection_name: String,
    embedding_generator: Arc<dyn EmbeddingGenerator>,
}

impl ChromaMemoryStorage {
    /// Create a new ChromaDB storage instance
    pub async fn new(
        base_url: Url,
        collection_name: String,
        embedding_generator: Arc<dyn EmbeddingGenerator>,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Ensure the collection exists
        let storage = Self {
            client,
            base_url,
            collection_name,
            embedding_generator,
        };

        storage.ensure_collection_exists().await?;

        Ok(storage)
    }

    /// Create a new ChromaDB storage instance from a local path
    pub async fn from_path<P: AsRef<Path>>(
        _path: P,
        collection_name: String,
        embedding_generator: Arc<dyn EmbeddingGenerator>,
    ) -> Result<Self> {
        // For local ChromaDB, we would typically use the HTTP API on localhost
        // This is a simplified approach - in a real implementation, you might want to
        // start the ChromaDB server if it's not running
        let base_url = Url::parse("http://localhost:8000").context("Failed to parse ChromaDB URL")?;
        
        Self::new(base_url, collection_name, embedding_generator).await
    }

    /// Ensure the collection exists, creating it if necessary
    async fn ensure_collection_exists(&self) -> Result<()> {
        // Check if collection exists
        let collections_url = self.base_url.join("/api/v1/collections")?;
        let response = self.client.get(collections_url)
            .send()
            .await
            .context("Failed to get collections")?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get collections: {}", response.status()));
        }

        let collections: serde_json::Value = response.json().await
            .context("Failed to parse collections response")?;

        // Check if our collection exists
        let collection_exists = if let Some(collections_array) = collections.as_array() {
            collections_array.iter().any(|c| {
                c.get("name").and_then(|n| n.as_str()) == Some(&self.collection_name)
            })
        } else {
            false
        };

        // Create collection if it doesn't exist
        if !collection_exists {
            let create_url = self.base_url.join("/api/v1/collections")?;
            let response = self.client.post(create_url)
                .json(&serde_json::json!({
                    "name": self.collection_name,
                    "metadata": { "hnsw:space": "cosine" } // Use cosine similarity
                }))
                .send()
                .await
                .context("Failed to create collection")?;

            if !response.status().is_success() {
                return Err(anyhow::anyhow!("Failed to create collection: {}", response.status()));
            }
        }

        Ok(())
    }

    /// Format memory metadata for ChromaDB
    fn format_metadata(&self, memory: &Memory) -> HashMap<String, serde_json::Value> {
        let mut metadata = HashMap::new();
        
        // Add basic fields
        metadata.insert("content_hash".to_string(), serde_json::Value::String(memory.content_hash.clone()));
        metadata.insert("timestamp_seconds".to_string(), serde_json::Value::Number(memory.timestamp_seconds.into()));
        
        // Add memory type if present
        if let Some(memory_type) = &memory.memory_type {
            metadata.insert("memory_type".to_string(), serde_json::Value::String(memory_type.clone()));
        }
        
        // Add tags as JSON array
        metadata.insert("tags".to_string(), serde_json::Value::Array(
            memory.tags.iter().map(|t| serde_json::Value::String(t.clone())).collect()
        ));
        
        // Add user metadata
        for (key, value) in &memory.metadata {
            metadata.insert(format!("metadata_{}", key), serde_json::Value::String(value.clone()));
        }
        
        metadata
    }

    /// Parse ChromaDB metadata back to Memory
    fn parse_metadata(&self, 
        id: &str, 
        document: &str, 
        metadata: &HashMap<String, serde_json::Value>,
        embedding: Option<Vec<f32>>
    ) -> Result<Memory> {
        // Extract basic fields
        let content = document.to_string();
        let content_hash = id.to_string();
        
        // Extract timestamp
        let timestamp_seconds = metadata.get("timestamp_seconds")
            .and_then(|v| v.as_i64())
            .unwrap_or_else(|| chrono::Utc::now().timestamp());
        
        // Extract memory type
        let memory_type = metadata.get("memory_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        // Extract tags
        let tags = metadata.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        
        // Extract user metadata
        let mut user_metadata = HashMap::new();
        for (key, value) in metadata {
            if let Some(stripped_key) = key.strip_prefix("metadata_") {
                if let Some(value_str) = value.as_str() {
                    user_metadata.insert(stripped_key.to_string(), value_str.to_string());
                }
            }
        }
        
        Ok(Memory {
            content,
            content_hash,
            tags,
            memory_type,
            timestamp_seconds,
            metadata: user_metadata,
            embedding,
        })
    }
}

#[async_trait]
impl MemoryStorage for ChromaMemoryStorage {
    async fn check_duplicate_exists(&self, content_hash: &str) -> Result<bool> {
        let get_url = self.base_url.join(&format!("/api/v1/collections/{}/get", self.collection_name))?;
        
        let response = self.client.post(get_url)
            .json(&serde_json::json!({
                "ids": [content_hash]
            }))
            .send()
            .await
            .context("Failed to check for duplicate")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to check for duplicate: {}", response.status()));
        }
        
        let result: serde_json::Value = response.json().await
            .context("Failed to parse duplicate check response")?;
            
        // Check if any documents were returned
        let ids = result.get("ids").and_then(|ids| ids.as_array());
        Ok(ids.map(|arr| !arr.is_empty()).unwrap_or(false))
    }

    async fn store(&self, memory: &Memory) -> Result<(bool, String)> {
        // Check for duplicates
        if self.check_duplicate_exists(&memory.content_hash).await? {
            return Ok((false, "Duplicate content detected".to_string()));
        }
        
        // Generate embedding if not already present
        let embedding = if let Some(ref emb) = memory.embedding {
            emb.clone()
        } else {
            self.embedding_generator.generate_embedding(&memory.content).await?
        };
        
        // Format metadata
        let metadata = self.format_metadata(memory);
        
        // Add to ChromaDB
        let add_url = self.base_url.join(&format!("/api/v1/collections/{}/add", self.collection_name))?;
        
        let response = self.client.post(add_url)
            .json(&serde_json::json!({
                "ids": [memory.content_hash],
                "embeddings": [embedding],
                "metadatas": [metadata],
                "documents": [memory.content]
            }))
            .send()
            .await
            .context("Failed to store memory")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to store memory: {}", response.status()));
        }
        
        Ok((true, format!("Successfully stored memory with hash: {}", memory.content_hash)))
    }

    async fn retrieve(&self, query_embedding: &Vec<f32>, n_results: usize) -> Result<Vec<MemoryQueryResult>> {
        // Query ChromaDB
        let query_url = self.base_url.join(&format!("/api/v1/collections/{}/query", self.collection_name))?;
        
        let response = self.client.post(query_url)
            .json(&serde_json::json!({
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["metadatas", "documents", "embeddings", "distances"]
            }))
            .send()
            .await
            .context("Failed to query memories")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to query memories: {}", response.status()));
        }
        
        let result: serde_json::Value = response.json().await
            .context("Failed to parse query response")?;
            
        // Process results
        let ids = result.get("ids").and_then(|ids| ids.as_array()).and_then(|arr| arr.get(0)).and_then(|ids| ids.as_array());
        let documents = result.get("documents").and_then(|docs| docs.as_array()).and_then(|arr| arr.get(0)).and_then(|docs| docs.as_array());
        let metadatas = result.get("metadatas").and_then(|meta| meta.as_array()).and_then(|arr| arr.get(0)).and_then(|meta| meta.as_array());
        let distances = result.get("distances").and_then(|dist| dist.as_array()).and_then(|arr| arr.get(0)).and_then(|dist| dist.as_array());
        let embeddings = result.get("embeddings").and_then(|emb| emb.as_array()).and_then(|arr| arr.get(0)).and_then(|emb| emb.as_array());
        
        let mut results = Vec::new();
        
        if let (Some(ids), Some(documents), Some(metadatas), Some(distances)) = (ids, documents, metadatas, distances) {
            for i in 0..ids.len() {
                if let (Some(id), Some(document), Some(metadata), Some(distance)) = (
                    ids.get(i).and_then(|v| v.as_str()),
                    documents.get(i).and_then(|v| v.as_str()),
                    metadatas.get(i).and_then(|v| v.as_object()),
                    distances.get(i).and_then(|v| v.as_f64()),
                ) {
                    // Convert metadata to HashMap
                    let metadata_map: HashMap<String, serde_json::Value> = metadata.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    
                    // Extract embedding if available
                    let embedding = embeddings.and_then(|embs| embs.get(i))
                        .and_then(|emb| emb.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|d| d as f32)).collect::<Vec<f32>>());
                    
                    // Parse memory
                    let memory = self.parse_metadata(id, document, &metadata_map, embedding)?;
                    
                    // Calculate relevance score (1 - distance for cosine similarity)
                    let relevance_score = 1.0 - distance as f32;
                    
                    results.push(MemoryQueryResult {
                        memory,
                        relevance_score,
                    });
                }
            }
        }
        
        Ok(results)
    }

    async fn search_by_tag(&self, tags: &[String]) -> Result<Vec<Memory>> {
        if tags.is_empty() {
            return Ok(Vec::new());
        }
        
        // Build where filter for tags
        let tag_conditions: Vec<serde_json::Value> = tags.iter()
            .map(|tag| {
                serde_json::json!({
                    "$contains": {
                        "path": "tags",
                        "value": tag
                    }
                })
            })
            .collect();
            
        let where_filter = if tag_conditions.len() == 1 {
            tag_conditions[0].clone()
        } else {
            serde_json::json!({
                "$or": tag_conditions
            })
        };
        
        // Query ChromaDB
        let get_url = self.base_url.join(&format!("/api/v1/collections/{}/get", self.collection_name))?;
        
        let response = self.client.post(get_url)
            .json(&serde_json::json!({
                "where": where_filter,
                "include": ["metadatas", "documents", "embeddings"]
            }))
            .send()
            .await
            .context("Failed to search by tags")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to search by tags: {}", response.status()));
        }
        
        let result: serde_json::Value = response.json().await
            .context("Failed to parse tag search response")?;
            
        // Process results
        let ids = result.get("ids").and_then(|ids| ids.as_array());
        let documents = result.get("documents").and_then(|docs| docs.as_array());
        let metadatas = result.get("metadatas").and_then(|meta| meta.as_array());
        let embeddings = result.get("embeddings").and_then(|emb| emb.as_array());
        
        let mut memories = Vec::new();
        
        if let (Some(ids), Some(documents), Some(metadatas)) = (ids, documents, metadatas) {
            for i in 0..ids.len() {
                if let (Some(id), Some(document), Some(metadata)) = (
                    ids.get(i).and_then(|v| v.as_str()),
                    documents.get(i).and_then(|v| v.as_str()),
                    metadatas.get(i).and_then(|v| v.as_object()),
                ) {
                    // Convert metadata to HashMap
                    let metadata_map: HashMap<String, serde_json::Value> = metadata.iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    
                    // Extract embedding if available
                    let embedding = embeddings.and_then(|embs| embs.get(i))
                        .and_then(|emb| emb.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|d| d as f32)).collect::<Vec<f32>>());
                    
                    // Parse memory
                    let memory = self.parse_metadata(id, document, &metadata_map, embedding)?;
                    memories.push(memory);
                }
            }
        }
        
        Ok(memories)
    }

    async fn delete(&self, content_hash: &str) -> Result<(bool, String)> {
        // Check if memory exists
        if !self.check_duplicate_exists(content_hash).await? {
            return Ok((false, format!("No memory found with hash: {}", content_hash)));
        }
        
        // Delete from ChromaDB
        let delete_url = self.base_url.join(&format!("/api/v1/collections/{}/delete", self.collection_name))?;
        
        let response = self.client.post(delete_url)
            .json(&serde_json::json!({
                "ids": [content_hash]
            }))
            .send()
            .await
            .context("Failed to delete memory")?;
            
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to delete memory: {}", response.status()));
        }
        
        Ok((true, format!("Successfully deleted memory with hash: {}", content_hash)))
    }
}
