use anyhow::{Result, Context};
use rmcp::{
    model::{CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    tool, Error as McpError, ServerHandler, ServiceExt,
    transport::stdio,
};
use std::sync::Arc;
// *** Import MakeWriter trait ***
use tracing_subscriber::{self, fmt::MakeWriter, EnvFilter}; // Make sure MakeWriter is imported

// Module declarations
mod config;
mod embeddings;
mod models;
mod storage;
mod utils;

// Import specific items
use config::Config;
use embeddings::{EmbeddingGenerator, DummyEmbeddingGenerator, OnnxEmbeddingGenerator, EmbeddingError};
use models::{StoreMemoryRequest, RetrieveMemoryRequest, SearchByTagRequest, DeleteMemoryRequest};
use storage::{MemoryStorage, InMemoryStorage, ChromaMemoryStorage};

// Helper functions to convert errors to McpError
fn to_mcp_error(error: anyhow::Error) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

// This function maps EmbeddingError variants to McpError types
fn embedding_error_to_mcp(error: EmbeddingError) -> McpError {
    match error {
        EmbeddingError::ModelNotFound(msg) |
        EmbeddingError::TokenizerNotFound(msg) |
        EmbeddingError::ModelLoadError(msg) |
        EmbeddingError::TokenizerLoadError(msg) => McpError::invalid_params(format!("Configuration Error: {}", msg), None),

        EmbeddingError::InferenceError(msg) |
        EmbeddingError::TokenizationError(msg) |
        EmbeddingError::TensorError(msg) |
        EmbeddingError::OutputProcessingError(msg) => McpError::internal_error(format!("Embedding Generation Failed: {}", msg), None),

        EmbeddingError::Other(e) => McpError::internal_error(format!("Embedding Error: {}", e), None),
    }
}

#[derive(Clone)]
struct MemoryServer {
    storage: Arc<dyn MemoryStorage>,
    embedding_generator: Arc<dyn EmbeddingGenerator>,
}

#[tool(tool_box)]
impl MemoryServer {
    fn new(storage: Arc<dyn MemoryStorage>, embedding_generator: Arc<dyn EmbeddingGenerator>) -> Self {
        Self { storage, embedding_generator }
    }

    #[tool(description = "Store a new memory")]
    async fn store_memory(
        &self,
        #[tool(aggr)] request: StoreMemoryRequest,
    ) -> Result<CallToolResult, McpError> {
        let metadata = request.metadata.unwrap_or_default();
        let content = request.content.clone();
        let content_hash = utils::generate_content_hash(&content, &metadata).map_err(to_mcp_error)?;

        let timestamp = utils::get_current_timestamp();
        let memory = models::Memory {
            content,
            content_hash,
            tags: request.tags.unwrap_or_default(),
            memory_type: request.memory_type,
            timestamp_seconds: timestamp.timestamp(),
            metadata,
            embedding: None,
        };

        let (success, message) = self.storage.store(&memory).await.map_err(to_mcp_error)?;

        if success {
            Ok(CallToolResult::success(vec![Content::text(message)]))
        } else {
            Ok(CallToolResult::error(vec![Content::text(message)]))
        }
    }

    #[tool(description = "Retrieve memories semantically similar to the query")]
    async fn retrieve_memory(
        &self,
        #[tool(aggr)] request: RetrieveMemoryRequest,
    ) -> Result<CallToolResult, McpError> {
        let query_embedding = self.embedding_generator
            .generate_embedding(&request.query).await
            .map_err(embedding_error_to_mcp)?;

        let results = self.storage.retrieve(&query_embedding, request.n_results.unwrap_or(5)).await
            .map_err(to_mcp_error)?;

        if results.is_empty() {
            Ok(CallToolResult::success(vec![Content::text(
                "No matching memories found".to_string(),
            )]))
        } else {
            let formatted_results = results
                .iter()
                .enumerate()
                .map(|(i, res)| {
                    format!(
                        "Memory {}:\nContent: {}\nHash: {}\nScore: {:.4}\nTags: {:?}\n---",
                        i + 1,
                        res.memory.content,
                        res.memory.content_hash,
                        res.relevance_score,
                        res.memory.tags
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Found {} memories:\n{}",
                results.len(),
                formatted_results
            ))]))
        }
    }

    #[tool(description = "Search memories by tags")]
    async fn search_by_tag(
        &self,
        #[tool(aggr)] request: SearchByTagRequest,
    ) -> Result<CallToolResult, McpError> {
         if request.tags.is_empty() {
             let error_message = "No tags provided for search.".to_string();
             return Ok(CallToolResult::error(vec![Content::text(error_message)]));
         }

        let memories = self.storage.search_by_tag(&request.tags).await.map_err(to_mcp_error)?;

        if memories.is_empty() {
            Ok(CallToolResult::success(vec![Content::text(
                "No memories found with the specified tags".to_string(),
            )]))
        } else {
            let formatted_memories = memories
                .iter()
                .enumerate()
                .map(|(i, memory)| {
                    format!(
                        "Memory {}:\nContent: {}\nHash: {}\nTags: {:?}\n---",
                        i + 1,
                        memory.content,
                        memory.content_hash,
                        memory.tags
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Found {} memories with tags {:?}:\n{}",
                memories.len(),
                request.tags,
                formatted_memories
            ))]))
        }
    }

    #[tool(description = "Delete a memory by its hash")]
    async fn delete_memory(
        &self,
        #[tool(aggr)] request: DeleteMemoryRequest,
    ) -> Result<CallToolResult, McpError> {
        let (success, message) = self.storage.delete(&request.content_hash).await.map_err(to_mcp_error)?;

        if success {
             Ok(CallToolResult::success(vec![Content::text(message)]))
        } else {
             Ok(CallToolResult::error(vec![Content::text(message)]))
        }
    }
}

#[tool(tool_box)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerInfo {
        let embedding_model_name = self.embedding_generator.name();
        let embedding_size = self.embedding_generator.get_embedding_size();

        let base_instructions = "This server provides memory storage and retrieval functionality. Use 'store_memory' to store new memories, 'retrieve_memory' for semantic search, 'search_by_tag' to find memories by tags, and 'delete_memory' to remove memories.";
        let instructions = format!("{} Currently using {} embedding model (size {}).", base_instructions, embedding_model_name, embedding_size);

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "mcp-memory-service-rs".to_string(),
                version: "0.1.2".to_string(),
            },
            instructions: Some(instructions),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let config_for_log = Config::load().unwrap_or_default();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config_for_log.log_level)))
        // *** FIX E0277/E0599: Wrap stderr() in a closure ***
        .with_writer(|| std::io::stderr()) // Closure satisfies MakeWriter trait bound
        .with_ansi(false)
        .init();

    // Load configuration
    let config = Config::load()?;
    tracing::info!("Configuration loaded: {:?}", config);

    // Initialize embedding generator based on configuration
    let embedding_generator: Arc<dyn EmbeddingGenerator> = match config.embedding_model {
        config::EmbeddingModel::Dummy => {
            tracing::info!("Using dummy embedding generator with size {}", config.embedding_size);
            Arc::new(DummyEmbeddingGenerator::new(config.embedding_size))
        },
        config::EmbeddingModel::Onnx => {
            tracing::info!("Attempting to initialize ONNX embedding generator...");
            if let Some(model_path) = config.embedding_model_path.clone() {
                match OnnxEmbeddingGenerator::new(model_path.clone(), None, config.embedding_size) {
                    Ok(generator) => {
                        tracing::info!("Successfully initialized ONNX embedding generator from path: {:?}", model_path);
                        Arc::new(generator)
                    },
                    Err(e) => {
                        tracing::error!("Failed to initialize ONNX embedding generator: {}", e);
                        tracing::warn!("Falling back to dummy embedding generator.");
                        Arc::new(DummyEmbeddingGenerator::new(config.embedding_size))
                    }
                }
            } else {
                tracing::error!("ONNX embedding model selected, but MCP_MEMORY_EMBEDDING_MODEL_PATH is not set.");
                 tracing::warn!("Falling back to dummy embedding generator.");
                Arc::new(DummyEmbeddingGenerator::new(config.embedding_size))
            }
        }
    };

    // Initialize storage based on configuration
    let storage: Arc<dyn MemoryStorage> = match config.storage_backend {
        config::StorageBackend::InMemory => {
            tracing::info!("Using in-memory storage");
            Arc::new(InMemoryStorage::new(embedding_generator.clone()))
        },
        config::StorageBackend::ChromaDB => {
            tracing::info!("Using ChromaDB storage");
            let storage_result = if let Some(url) = config.chroma_db_url.clone() {
                tracing::info!("Connecting to remote ChromaDB at {}", url);
                ChromaMemoryStorage::new(
                    url,
                    config.chroma_collection_name.clone(),
                    embedding_generator.clone(),
                ).await
            } else {
                tracing::info!("Using local ChromaDB (expecting server at http://localhost:8000 from path {:?})", config.chroma_db_path);
                 let default_chroma_url = url::Url::parse("http://localhost:8000")
                     .context("Failed to parse default ChromaDB URL")?;
                 ChromaMemoryStorage::new(
                     default_chroma_url,
                     config.chroma_collection_name.clone(),
                     embedding_generator.clone(),
                 ).await
            };

            match storage_result {
                 Ok(storage) => Arc::new(storage),
                 Err(e) => {
                     tracing::error!("Failed to initialize ChromaDB storage: {}", e);
                     tracing::warn!("Falling back to in-memory storage.");
                     Arc::new(InMemoryStorage::new(embedding_generator.clone()))
                 }
             }
        }
    };

    // Create and run server
    let service = MemoryServer::new(storage, embedding_generator).serve(stdio()).await?;

    tracing::info!("MCP Memory Service running on stdio. Waiting for requests...");

    service.waiting().await?;

    tracing::info!("MCP Memory Service shutting down.");

    Ok(())
}
