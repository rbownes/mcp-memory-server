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
  - ONNX embedding model (stub implementation for future development)

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

## Future Improvements

- Implement a real ONNX embedding model
- Add more storage backends
- Add more embedding models
- Add more tools for memory management
- Add authentication and authorization
- Add support for more transports (HTTP, WebSocket, etc.)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
