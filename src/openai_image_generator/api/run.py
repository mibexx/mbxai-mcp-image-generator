import uvicorn
import asyncio
from .app import app, server, register_tools

async def async_main():
    """Run the mbxai server."""
    # Register tools
    await register_tools()
    
    # Get available tools
    tools = await server.mcp_server.list_tools()
    
    # Extract tool names
    tool_names = [tool.name for tool in tools]
    print(f"Available tools: {', '.join(tool_names)}")
    
    # Run the server
    config = uvicorn.Config(app, host="0.0.0.0", port=5000)
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()

def main():
    """Entry point for the service."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 