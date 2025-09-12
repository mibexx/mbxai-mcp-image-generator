from mbxai.mcp.server import MCPServer
from ..config import get_config, get_image_storage_config
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import mimetypes

config = get_config()

# Import the tools
from ..project.image_generator import generate_image, edit_image

# Initialize mbxai server
server = MCPServer(
    name=config.name,
    description=config.description
)

# Get the FastAPI app from the mbxai server
app = server.app

# Add image serving endpoint
@app.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a stored image file.
    
    Args:
        filename: The filename of the image to serve
        
    Returns:
        FileResponse containing the image file
        
    Raises:
        HTTPException: If the file is not found or invalid
    """
    # Get image storage configuration
    storage_config = get_image_storage_config()
    storage_path = Path(storage_config.storage_path)
    
    # Construct file path
    file_path = storage_path / filename
    
    # Security check: ensure filename doesn't contain path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Check if file exists
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Determine the correct media type based on file extension
    media_type, _ = mimetypes.guess_type(str(file_path))
    if not media_type or not media_type.startswith('image/'):
        media_type = "image/png"  # fallback
    
    # Return the file
    return FileResponse(
        path=str(file_path),
        media_type=media_type
    )

# Function to register tools
async def register_tools():
    """Register all tools with the server."""
    await server.add_tool(generate_image)
    await server.add_tool(edit_image)
