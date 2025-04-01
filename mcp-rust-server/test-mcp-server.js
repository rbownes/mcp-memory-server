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
    console.log('Starting MCP server...');
    
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
      
      // Test hello tool
      await this.callTool("hello", { name: "Rust MCP" });
      
      // Test increment tool
      await this.callTool("increment");
      
      // Test get_counter tool
      await this.callTool("get_counter");
      
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
