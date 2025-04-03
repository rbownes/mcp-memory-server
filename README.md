# MCP Memory Service (Rust Implementation)

This is a Rust implementation of the Model Context Protocol (MCP) Memory Service, using the official [Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk).

## Features

- Implements an MCP server that provides memory storage and retrieval functionality
- Communicates over stdio for easy integration with MCP clients
- Provides the following tools:
  - `store_memory`: Store a new memory with content, tags, and metadata
  - `retrieve_memory`: Retrieve memories semantically similar to a query
  - `search_by_tag`: Search memories by tags
  - `delete_memory`: Delete a memory by its hash
- Supports multiple storage backends:
  - In-memory storage (for testing and development)
  - ChromaDB storage (for production use)
- Supports multiple embedding models:
  - Dummy embedding generator (for testing and development)
  - ONNX embedding model (transformer-based embeddings using ONNX Runtime)

## Prerequisites

- Rust and Cargo (1.75.0 or later)
- Node.js and npm (for testing with the MCP inspector)
- Optional: ChromaDB server (for production use)

## Building

```bash
cargo build
```

For a release build:

```bash
cargo build --release
```

## Running

To run the server directly:

```bash
cargo run
```

With environment variables for configuration:

```bash
MCP_MEMORY_STORAGE_BACKEND=chromadb \
MCP_MEMORY_CHROMA_PATH=/path/to/chroma \
MCP_MEMORY_EMBEDDING_MODEL=onnx \
cargo run
```

## Configuration

The server can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_MEMORY_STORAGE_BACKEND` | Storage backend (`inmemory` or `chromadb`) | `inmemory` |
| `MCP_MEMORY_CHROMA_PATH` | Path to ChromaDB data directory | Platform-specific data directory |
| `MCP_MEMORY_CHROMA_URL` | URL to ChromaDB server (optional) | None |
| `MCP_MEMORY_CHROMA_COLLECTION` | ChromaDB collection name | `memory_collection` |
| `MCP_MEMORY_EMBEDDING_MODEL` | Embedding model (`dummy` or `onnx`) | `dummy` |
| `MCP_MEMORY_EMBEDDING_MODEL_PATH` | Path to ONNX model file (optional) | None |
| `MCP_MEMORY_EMBEDDING_SIZE` | Embedding vector size | 384 |
| `MCP_MEMORY_LOG_LEVEL` | Log level | `info` |

## Testing

This project includes a simple Node.js test script for testing the server. To run the tests:

```bash
npm test
```

## Project Structure

- `src/main.rs`: The main server implementation
- `src/config.rs`: Configuration handling
- `src/models.rs`: Data models
- `src/storage/`: Storage implementations
  - `mod.rs`: Storage trait and in-memory implementation
  - `chroma.rs`: ChromaDB storage implementation
- `src/embeddings.rs`: Embedding model implementations
- `src/utils.rs`: Utility functions
- `Cargo.toml`: Rust project configuration
- `package.json`: Node.js project configuration for testing tools
- `test-mcp-server.js`: A simple Node.js script to test the server programmatically

## Implementation Details

The server is implemented using the `rmcp` crate from the Rust MCP SDK. It uses:

- The `#[tool(tool_box)]` attribute macro to define tools
- The `ServerHandler` trait to implement the MCP protocol
- The `stdio()` transport for CLI usage
- Asynchronous Rust with Tokio for concurrent operations
- Tracing for structured logging

## Example Usage

After starting the server with `cargo run`, you can send JSON-RPC requests to its stdin. For example:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{"tools":true},"client_info":{"name":"test-client","version":"0.1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"list_tools","params":{}}
{"jsonrpc":"2.0","id":3,"method":"call_tool","params":{"name":"store_memory","arguments":{"content":"This is a test memory","tags":["test","example"]}}}
```

## ONNX Embedding Implementation

The ONNX embedding model implementation uses the ONNX Runtime to run transformer-based models for generating embeddings. Here's how it works:

1. **Model Loading**: The `OnnxEmbeddingGenerator` loads a pre-trained transformer model in ONNX format and a tokenizer from the specified paths.

2. **Tokenization**: Input text is tokenized using the HuggingFace `tokenizers` library, which converts text into token IDs, attention masks, and token type IDs.

3. **Inference**: The tokenized inputs are passed to the ONNX model, which produces the transformer's hidden states.

4. **Mean Pooling**: The last hidden state is processed using mean pooling (weighted by the attention mask) to create a fixed-size embedding vector.

5. **Normalization**: The resulting embedding is L2-normalized to ensure consistent vector magnitudes.

To use the ONNX embedding model:

1. Export a transformer model (like BERT, RoBERTa, etc.) to ONNX format using a tool like HuggingFace's `transformers.onnx`.
2. Save the tokenizer as a `tokenizer.json` file in the same directory as the model or specify a separate path.
3. Set the environment variables:
   ```bash
   MCP_MEMORY_EMBEDDING_MODEL=onnx
   MCP_MEMORY_EMBEDDING_MODEL_PATH=/path/to/model.onnx
   MCP_MEMORY_EMBEDDING_SIZE=768  # Adjust based on your model's output size
   ```

## Registering with an MCP Client

To register this server with an MCP client (like Claude), you need to add it to the client's MCP configuration. Here's an example JSON configuration:

```json
{
  "mcpServers": {
    "memory-service": {
      "command": "/path/to/mcp-rust-server",
      "args": [],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "inmemory",
        "MCP_MEMORY_EMBEDDING_MODEL": "onnx",
        "MCP_MEMORY_EMBEDDING_MODEL_PATH": "/path/to/model.onnx",
        "MCP_MEMORY_EMBEDDING_SIZE": "768",
        "MCP_MEMORY_LOG_LEVEL": "info"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

For Claude Desktop, this configuration would be added to:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

For Claude VSCode extension, the configuration would be added to:
- `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

## Future Improvements

- Add support for more transformer architectures
- Add more storage backends
- Add more embedding models
- Add more tools for memory management
- Add authentication and authorization
- Add support for more transports (HTTP, WebSocket, etc.)

## Architecture

```mermaid
graph TD
    subgraph "External Interactions"
        direction LR
        Client([MCP Client]) -. JSON-RPC .-> StdioTransport[Stdio Transport]
        EnvVars[(Environment Variables)] -.-> Config
        ChromaDBServer[(ChromaDB Server)] <-. HTTP API .-> ChromaStorage
        OnnxModel[("ONNX Model File (.onnx)")] <-. Loads .-> OnnxEmbed
        TokenizerFile[("Tokenizer File (tokenizer.json)")] <-. Loads .-> OnnxEmbed
    end

    subgraph "MCP Memory Service (Rust Application)"
        direction TB
        StdioTransport -- Forwards Requests --> ServerCore{MCP Server Core / main.rs}

        ServerCore -- Reads --> Config(Configuration / config.rs)
        ServerCore -- Instantiates --> EmbeddingImpl{{Selected Embedding Generator}}
        ServerCore -- Instantiates --> StorageImpl{{Selected Storage Backend}}
        ServerCore -- Uses Tool Impls --> ToolLogic(Tool Logic: store, retrieve, search, delete)

        ToolLogic -- Uses --> StorageImpl
        ToolLogic -- Uses --> EmbeddingImpl  / For retrieve query embedding
        ToolLogic -- Uses --> Models(Data Models / models.rs)
        ToolLogic -- Uses --> Utils(Utilities / utils.rs)

        subgraph "Embedding Layer (embeddings.rs)"
            direction TB
            EmbeddingImpl -- Is an instance of --> EmbeddingTrait(EmbeddingGenerator Trait)
            DummyEmbed(DummyEmbeddingGenerator) -- Implements --> EmbeddingTrait
            OnnxEmbed(OnnxEmbeddingGenerator) -- Implements --> EmbeddingTrait
            OnnxEmbed -- Uses Lib --> OrtLib[ort crate]
            OnnxEmbed -- Uses Lib --> TokenizersLib[tokenizers crate]
        end

        subgraph "Storage Layer (storage/*)"
            direction TB
            StorageImpl -- Is an instance of --> StorageTrait(MemoryStorage Trait)
            InMemoryStorage(InMemoryStorage) -- Implements --> StorageTrait
            ChromaStorage(ChromaMemoryStorage) -- Implements --> StorageTrait
            StorageImpl -- Uses --> EmbeddingImpl / For storing embeddings
            StorageImpl -- Uses --> Models
            ChromaStorage -- Uses Lib --> ReqwestLib[reqwest crate]
        end

    end

    %% Styling
    classDef external fill:#f9f,stroke:#333,stroke-width:1px;
    classDef rust_app fill:#e6ffed,stroke:#333,stroke-width:1px;
    classDef trait fill:#lightblue,stroke:#333,stroke-width:1px;
    classDef impl fill:#lightgrey,stroke:#333,stroke-width:1px;
    classDef module fill:#whitesmoke,stroke:#333,stroke-width:1px;
    classDef lib fill:#cornsilk,stroke:#333,stroke-width:1px;


    class Client,EnvVars,ChromaDBServer,OnnxModel,TokenizerFile external;
    class StdioTransport,ServerCore,ToolLogic,Config,Models,Utils rust_app;
    class EmbeddingTrait,StorageTrait trait;
    class EmbeddingImpl,StorageImpl,DummyEmbed,OnnxEmbed,InMemoryStorage,ChromaStorage impl;
    class OrtLib,TokenizersLib,ReqwestLib lib;
    class Embedding Logic, Storage Layer module
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
