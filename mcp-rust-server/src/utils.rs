use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use anyhow::Result;
use chrono::{DateTime, Utc};

// Helper to ensure consistent metadata serialization
#[derive(serde::Serialize)]
struct HashableMetadata<'a>(BTreeMap<&'a str, &'a str>);

pub fn generate_content_hash(content: &str, metadata: &HashMap<String, String>) -> Result<String> {
    // Normalize content
    let normalized_content = content.trim().to_lowercase();

    // Prepare metadata for consistent hashing
    let filtered_metadata: BTreeMap<&str, &str> = metadata
        .iter()
        // Exclude dynamic/non-content fields for hashing
        .filter(|(k, _)| !["timestamp", "content_hash", "embedding"].contains(&k.as_str()))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let metadata_json = serde_json::to_string(&HashableMetadata(filtered_metadata))?;

    // Combine and hash
    let mut hasher = Sha256::new();
    hasher.update(normalized_content.as_bytes());
    hasher.update(metadata_json.as_bytes());
    let hash_bytes = hasher.finalize();

    Ok(hex::encode(hash_bytes))
}

// MVP doesn't parse NLP time, just uses current time
pub fn get_current_timestamp() -> DateTime<Utc> {
    Utc::now()
}
