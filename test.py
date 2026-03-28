from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# Add a personal info tool
@mcp.tool()
def personal_info() -> dict:
    """Return my personal information"""
    return {
        "name": "胡桃战神",
        "role": "萌妹",
        "location": "璃月",
        "skills": ["唱歌", "跳舞", "超度"],
        "interests": ["捉弄七七"]
    }


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

# Start the server if this file is run directly
if __name__ == "__main__":
    print("Starting MCP server on http://localhost:8097")
    mcp.run(transport="stdio")