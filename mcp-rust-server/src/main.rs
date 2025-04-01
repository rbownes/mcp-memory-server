use anyhow::Result;
use rmcp::{
    model::{CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo},
    tool, Error as McpError, ServerHandler, ServiceExt,
    transport::stdio,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct MinimalServer {
    counter: Arc<Mutex<i32>>,
}

#[tool(tool_box)]
impl MinimalServer {
    fn new() -> Self {
        Self {
            counter: Arc::new(Mutex::new(0)),
        }
    }

    #[tool(description = "Say hello to someone")]
    async fn hello(
        &self,
        #[tool(param)]
        #[schemars(description = "Name to greet")]
        name: Option<String>,
    ) -> Result<CallToolResult, McpError> {
        let name = name.unwrap_or_else(|| "world".to_string());
        Ok(CallToolResult::success(vec![Content::text(
            format!("Hello, {}!", name),
        )]))
    }

    #[tool(description = "Get the current counter value")]
    async fn get_counter(&self) -> Result<CallToolResult, McpError> {
        let counter = self.counter.lock().await;
        Ok(CallToolResult::success(vec![Content::text(
            counter.to_string(),
        )]))
    }

    #[tool(description = "Increment the counter by 1")]
    async fn increment(&self) -> Result<CallToolResult, McpError> {
        let mut counter = self.counter.lock().await;
        *counter += 1;
        Ok(CallToolResult::success(vec![Content::text(
            counter.to_string(),
        )]))
    }
}

#[tool(tool_box)]
impl ServerHandler for MinimalServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "minimal-rust-mcp-server".to_string(),
                version: "0.1.0".to_string(),
            },
            instructions: Some("This is a minimal MCP server with a hello tool and a counter.".to_string()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize the server
    let service = MinimalServer::new().serve(stdio()).await?;

    // Log that the server is running
    eprintln!("Minimal MCP server running on stdio");

    // Keep the server running
    service.waiting().await?;

    Ok(())
}
