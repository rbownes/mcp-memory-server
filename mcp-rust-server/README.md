# Minimal Rust MCP Server

This is a minimal implementation of a Model Context Protocol (MCP) server in Rust, using the official [Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk).

## Features

- Implements a simple MCP server that communicates over stdio
- Provides three tools:
  - `hello`: Greets a user with an optional name parameter
  - `increment`: Increments a counter by 1
  - `get_counter`: Returns the current counter value

## Prerequisites

- Rust and Cargo (1.85.1 or later)
- Node.js and npm (for testing with the MCP inspector)

## Building

```bash
cargo build
```

## Running

To run the server directly:

```bash
cargo run
```

## Testing

This project includes a simple Node.js test script for testing the server. To run the tests:

```bash
npm test
```

The test script will:
- Start the MCP server
- Initialize a connection
- List available tools
- Call each tool with appropriate parameters
- Display the server responses

## Project Structure

- `src/main.rs`: The main server implementation
- `Cargo.toml`: Rust project configuration
- `package.json`: Node.js project configuration for testing tools
- `test-mcp-server.js`: A simple Node.js script to test the server programmatically

## Implementation Details

The server is implemented using the `rmcp` crate from the Rust MCP SDK. It uses:

- The `#[tool(tool_box)]` attribute macro to define tools
- The `ServerHandler` trait to implement the MCP protocol
- The `stdio()` transport for CLI usage

## Example Usage

After starting the server with `cargo run`, you can send JSON-RPC requests to its stdin. For example:

```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocol_version":"2024-11-05","capabilities":{"tools":true},"client_info":{"name":"test-client","version":"0.1.0"}}}
{"jsonrpc":"2.0","id":2,"method":"list_tools","params":{}}
{"jsonrpc":"2.0","id":3,"method":"call_tool","params":{"name":"hello","arguments":{"name":"Rust MCP"}}}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
