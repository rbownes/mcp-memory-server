use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rmcp::schemars;

#[derive(Debug, Serialize, Deserialize, Clone, schemars::JsonSchema)]
pub struct Memory {
    pub content: String,
    pub content_hash: String, // Typically SHA256 hex string
    pub tags: Vec<String>,
    pub memory_type: Option<String>,
    // Store timestamp as seconds since epoch
    pub timestamp_seconds: i64,
    // Keep metadata simple for MVP
    pub metadata: HashMap<String, String>,
    // Embedding won't be serialized, but might be held in memory
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl Memory {
    // Helper to get DateTime from timestamp_seconds
    pub fn timestamp(&self) -> DateTime<Utc> {
        DateTime::<Utc>::from_timestamp(self.timestamp_seconds, 0)
            .unwrap_or_else(|| Utc::now())
    }
    
    // Helper to set timestamp from DateTime
    pub fn set_timestamp(&mut self, dt: DateTime<Utc>) {
        self.timestamp_seconds = dt.timestamp();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, schemars::JsonSchema)]
pub struct MemoryQueryResult {
    pub memory: Memory,
    pub relevance_score: f32,
}

// Request types for tools
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StoreMemoryRequest {
    pub content: String,
    pub tags: Option<Vec<String>>,
    pub memory_type: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RetrieveMemoryRequest {
    pub query: String,
    pub n_results: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchByTagRequest {
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct DeleteMemoryRequest {
    pub content_hash: String,
}
