from typing import Any
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from ..utils.client import generate_image_async

# Create a FastMCP instance for this module
mcp = FastMCP("image_generator")

class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate")
    size: str = Field(..., description="The size of the image ('1024x1024', '1536x1024', '1024x1536', 'auto') (default: 'auto')")
    quality: str = Field(..., description="The quality of the image ('standard', 'hd') (default: 'standard')")

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
        
        image_url = await generate_image_async(
            prompt=input.prompt,
            model=openai_config.image_model,
            size=input.size,
            quality=input.quality
        )
        
        logger.info(f"Successfully received image URL: {image_url}")
        
        logger.info(f"Successfully generated image for prompt: {input.prompt[:50]}...")
        
        return {
            "success": True,
            "prompt": input.prompt,
            "model": openai_config.image_model,
            "size": input.size,
            "quality": input.quality,
            "image_url": image_url,
            "message": "Image generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        
        # Determine error type for better user feedback
        error_message = "Failed to generate image"
        if "timeout" in str(e).lower():
            error_message = "Timeout while generating image"
        elif "http" in str(e).lower():
            error_message = "Network error during image generation"
        
        return {
            "success": False,
            "error": str(e),
            "prompt": input.prompt,
            "model": openai_config.image_model,
            "size": input.size,
            "quality": input.quality,
            "message": error_message
        } 