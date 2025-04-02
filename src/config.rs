use anyhow::{Context, Result};
use std::{env, path::PathBuf};
use directories_next::ProjectDirs;
use url::Url;

/// Storage backend options
#[derive(Debug, Clone, PartialEq)]
pub enum StorageBackend {
    /// In-memory storage (no persistence)
    InMemory,
    /// ChromaDB storage
    ChromaDB,
}

impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::InMemory
    }
}

/// Embedding model options
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingModel {
    /// Dummy embedding generator (for testing)
    Dummy,
    /// ONNX model
    Onnx,
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::Dummy
    }
}

#[derive(Debug)]
pub struct Config {
    // Storage configuration
    pub storage_backend: StorageBackend,
    pub chroma_db_path: PathBuf,
    pub chroma_db_url: Option<Url>,
    pub chroma_collection_name: String,
    
    // Embedding configuration
    pub embedding_model: EmbeddingModel,
    pub embedding_model_path: Option<PathBuf>,
    pub embedding_size: usize,
    
    // Server configuration
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            storage_backend: StorageBackend::default(),
            chroma_db_path: PathBuf::new(),
            chroma_db_url: None,
            chroma_collection_name: "memory_collection".to_string(),
            embedding_model: EmbeddingModel::default(),
            embedding_model_path: None,
            embedding_size: 384, // Default embedding size
            log_level: "info".to_string(),
        }
    }
}

impl Config {
    pub fn load() -> Result<Self> {
        // Load .env file if present (optional)
        let _ = dotenvy::dotenv();

        // Start with default config
        let mut config = Config::default();

        // Storage backend
        if let Ok(backend) = env::var("MCP_MEMORY_STORAGE_BACKEND") {
            config.storage_backend = match backend.to_lowercase().as_str() {
                "chromadb" => StorageBackend::ChromaDB,
                _ => StorageBackend::InMemory,
            };
        }

        // ChromaDB configuration
        let chroma_path_str = env::var("MCP_MEMORY_CHROMA_PATH")
            .or_else(|_| Self::get_default_path("chroma_db"))?;
        config.chroma_db_path = PathBuf::from(chroma_path_str);

        // Validate and create path
        Self::validate_or_create_path(&config.chroma_db_path)?;

        // ChromaDB URL (optional, for remote ChromaDB)
        if let Ok(url) = env::var("MCP_MEMORY_CHROMA_URL") {
            config.chroma_db_url = Some(Url::parse(&url).context("Invalid ChromaDB URL")?);
        }

        // ChromaDB collection name
        if let Ok(collection) = env::var("MCP_MEMORY_CHROMA_COLLECTION") {
            config.chroma_collection_name = collection;
        }

        // Embedding model
        if let Ok(model) = env::var("MCP_MEMORY_EMBEDDING_MODEL") {
            config.embedding_model = match model.to_lowercase().as_str() {
                "onnx" => EmbeddingModel::Onnx,
                _ => EmbeddingModel::Dummy,
            };
        }

        // Embedding model path
        if let Ok(path) = env::var("MCP_MEMORY_EMBEDDING_MODEL_PATH") {
            config.embedding_model_path = Some(PathBuf::from(path));
        }

        // Embedding size
        if let Ok(size) = env::var("MCP_MEMORY_EMBEDDING_SIZE") {
            if let Ok(size) = size.parse::<usize>() {
                config.embedding_size = size;
            }
        }

        // Log level
        if let Ok(level) = env::var("MCP_MEMORY_LOG_LEVEL") {
            config.log_level = level;
        }

        tracing::info!("Using ChromaDB path: {:?}", config.chroma_db_path);
        if let Some(url) = &config.chroma_db_url {
            tracing::info!("Using ChromaDB URL: {}", url);
        }

        Ok(config)
    }

    fn get_default_path(sub_dir: &str) -> Result<String> {
        let proj_dirs = ProjectDirs::from("ai", "Anthropic", "MCPMemoryService")
            .context("Failed to get project directories")?;
        let data_dir = proj_dirs.data_local_dir(); // Or data_dir() depending on preference
        let default_path = data_dir.join(sub_dir);
        Ok(default_path.to_string_lossy().to_string())
    }

    fn validate_or_create_path(path: &PathBuf) -> Result<()> {
        if !path.exists() {
            std::fs::create_dir_all(path)
                .context(format!("Failed to create directory: {:?}", path))?;
            tracing::info!("Created directory: {:?}", path);
        }
        // Basic writability check
        let test_file_path = path.join(".write_test");
        std::fs::File::create(&test_file_path)
            .and_then(|_| std::fs::remove_file(&test_file_path))
            .context(format!("Directory {:?} is not writable", path))?;
        Ok(())
    }
}
