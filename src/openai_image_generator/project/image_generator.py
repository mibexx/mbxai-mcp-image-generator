from typing import Any
import base64
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from ..utils.client import generate_image_async

# Create a FastMCP instance for this module
mcp = FastMCP("image_generator")

class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate")
    size: str = Field(..., description="The size of the image ('1024x1024', '1536x1024', '1024x1536', 'auto') (default: 'auto')")
    quality: str = Field(..., description="The quality of the image ('low', 'medium', 'high', 'auto') (default: 'auto')")
    output_compression: int | None = Field(..., description="Compression level (0-100%) for JPEG and WebP formats (suggested: 80 for JPEG/WebP, null for PNG)")

@mcp.tool()
async def generate_image(input: ImageGenerationInput) -> dict[str, Any]:
    """Generate an image based on a text prompt using OpenAI's image generation API.
    
    Args:
        input: ImageGenerationInput model containing prompt and generation parameters
        
    Returns:
        A dictionary containing the generated image data and metadata
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Generating image with prompt: {input.prompt[:100]}...")
        
        # Generate the image using the async client function with configured model
        from ..config import get_openai_config
        openai_config = get_openai_config()
        
        image_bytes = await generate_image_async(
            prompt=input.prompt,
            model=openai_config.image_model,
            size=input.size,
            quality=input.quality,
            output_compression=input.output_compression
        )
        
        # Convert bytes back to base64 for JSON response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        logger.info(f"Successfully generated image for prompt: {input.prompt[:50]}...")
        
        return {
            "success": True,
            "prompt": input.prompt,
            "model": openai_config.image_model,
            "size": input.size,
            "quality": input.quality,
            "output_compression": input.output_compression,
            "image_base64": image_base64,
            "image_size_bytes": len(image_bytes),
            "message": "Image generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "prompt": input.prompt,
            "model": openai_config.image_model,
            "size": input.size,
            "quality": input.quality,
            "output_compression": input.output_compression,
            "message": "Failed to generate image"
        } 