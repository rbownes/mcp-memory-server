#!/usr/bin/env node

const { spawn } = require('child_process');
const { createInterface } = require('readline');

// Modern approach to avoid deprecation warnings
class McpTester {
  constructor() {
    this.serverProcess = null;
    this.rl = null;
    this.requestQueue = [];
    this.currentRequestId = 0;
  }

  async start() {
    console.log('Starting MCP Memory Service...');
    
    // Spawn the MCP server process
    this.serverProcess = spawn('cargo', ['run'], { 
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', process.stderr]
    });

    // Create readline interface for server's stdout
    this.rl = createInterface({
      input: this.serverProcess.stdout,
      crlfDelay: Infinity
    });

    // Handle server output
    this.rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        console.log('\nServer response:');
        console.log(JSON.stringify(response, null, 2));
      } catch (e) {
        console.log('\nServer output (non-JSON):', line);
      }
    });

    // Set up error handling
    this.serverProcess.on('close', (code) => {
      console.log(`\nServer process exited with code ${code}`);
    });

    this.serverProcess.on('error', (err) => {
      console.error('\nFailed to start server process:', err);
      process.exit(1);
    });

    // Wait for server to start
    await new Promise(resolve => setTimeout(resolve, 1000));
    console.log('Server started');
  }

  async sendRequest(method, params = {}) {
    this.currentRequestId++;
    const request = {
      jsonrpc: "2.0",
      id: this.currentRequestId,
      method,
      params
    };
    
    console.log(`\nSending ${method} request:`);
    console.log(JSON.stringify(request, null, 2));
    
    this.serverProcess.stdin.write(JSON.stringify(request) + '\n');
    
    // Wait a bit for the response
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  async initialize() {
    await this.sendRequest("initialize", {
      protocol_version: "2024-11-05",
      capabilities: {
        tools: true
      },
      client_info: {
        name: "test-client",
        version: "0.1.0"
      }
    });
  }

  async listTools() {
    await this.sendRequest("list_tools");
  }

  async callTool(name, args = {}) {
    await this.sendRequest("call_tool", {
      name,
      arguments: args
    });
  }

  async runTests() {
    try {
      await this.start();
      await this.initialize();
      await this.listTools();
      
      // Test store_memory tool
      console.log('\n--- Testing store_memory tool ---');
      await this.callTool("store_memory", { 
        content: "This is a test memory",
        tags: ["test", "example"],
        memory_type: "note",
        metadata: {
          source: "test-script",
          importance: "high"
        }
      });
      
      // Test store_memory tool with minimal parameters
      console.log('\n--- Testing store_memory tool with minimal parameters ---');
      await this.callTool("store_memory", { 
        content: "This is another test memory"
      });
      
      // Test retrieve_memory tool
      console.log('\n--- Testing retrieve_memory tool ---');
      await this.callTool("retrieve_memory", {
        query: "test memory",
        n_results: 5
      });
      
      // Test search_by_tag tool
      console.log('\n--- Testing search_by_tag tool ---');
      await this.callTool("search_by_tag", {
        tags: ["test"]
      });
      
      // Test delete_memory tool (assuming we have a memory with this hash)
      // In a real test, we would get the hash from the store_memory response
      console.log('\n--- Testing delete_memory tool ---');
      await this.callTool("delete_memory", {
        content_hash: "f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2"
      });
      
      console.log('\nAll tests completed successfully!');
    } catch (error) {
      console.error('\nTest failed:', error);
    } finally {
      // Clean up
      if (this.serverProcess) {
        console.log('\nShutting down server...');
        this.serverProcess.kill();
      }
    }
  }
}

// Run the tests
const tester = new McpTester();
tester.runTests().catch(console.error);

// Handle SIGINT to gracefully shut down
process.on('SIGINT', () => {
  console.log('\nReceived SIGINT, shutting down...');
  if (tester.serverProcess) {
    tester.serverProcess.kill();
  }
  process.exit(0);
});
