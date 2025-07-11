from typing import Any
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from ..utils.client import generate_image_async

# Create a FastMCP instance for this module
mcp = FastMCP("image_generator")

class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate")
    size: str = Field(..., description="The size of the image ('1024x1024', '1024x1792', and '1792x1024')")
    quality: str = Field(..., description="The quality of the image ('standard', 'hd') (default: 'standard')")

@mcp.tool()
async def generate_image(input: ImageGenerationInput) -> dict[str, Any]:
    """Generate an image based on a text prompt using OpenAI's image generation API.
    
    Allowed sizes: 1024x1024, 1024x1792, and 1792x1024
    
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
        logger.error("=== Image Generation Tool Error ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input parameters:")
        logger.error(f"  - Prompt: {input.prompt}")
        logger.error(f"  - Size: {input.size}")
        logger.error(f"  - Quality: {input.quality}")
        
        # Determine error type for better user feedback
        error_message = "Failed to generate image"
        error_details = str(e)
        
        if "400" in str(e) or "bad request" in str(e).lower():
            error_message = "Invalid request parameters"
            logger.error("This appears to be a 400 Bad Request error. Common causes:")
            logger.error("  - Invalid prompt content (violates content policy)")
            logger.error("  - Invalid size or quality parameters")
            logger.error("  - Prompt too long or too short")
            logger.error("  - Unsupported characters in prompt")
            
        elif "timeout" in str(e).lower():
            error_message = "Timeout while generating image"
            logger.error("Request timed out - this may indicate network issues or API overload")
            
        elif "http" in str(e).lower() and "401" in str(e):
            error_message = "Authentication failed"
            logger.error("401 Unauthorized - check API key configuration")
            
        elif "http" in str(e).lower() and "403" in str(e):
            error_message = "Access forbidden"
            logger.error("403 Forbidden - check API permissions and quota")
            
        elif "http" in str(e).lower() and "429" in str(e):
            error_message = "Rate limit exceeded"
            logger.error("429 Too Many Requests - API rate limit exceeded")
            
        elif "http" in str(e).lower() and "500" in str(e):
            error_message = "Server error during image generation"
            logger.error("500 Internal Server Error - OpenAI API server issue")
            
        elif "http" in str(e).lower():
            error_message = "Network error during image generation"
            logger.error("HTTP error occurred during API request")
            
        # Log the full error context for debugging
        logger.error(f"Full error context: {error_details}")
        
        return {
            "success": False,
            "error": error_details,
            "prompt": input.prompt,
            "model": openai_config.image_model,
            "size": input.size,
            "quality": input.quality,
            "message": error_message
        } 