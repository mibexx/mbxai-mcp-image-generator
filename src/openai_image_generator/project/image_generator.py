from typing import Any
import logging
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from ..utils.client import generate_image_async, edit_image_async
import base64
import io
import httpx

# Create a FastMCP instance for this module
mcp = FastMCP("image_generator")


async def _process_image_input(image_input: str, name: str) -> io.BytesIO:
    """Process an image input that can be either a URL or base64 string.
    
    Args:
        image_input: Either a URL or base64 encoded string
        name: Name to assign to the file object
        
    Returns:
        BytesIO object containing the image data
        
    Raises:
        ValueError: If the input is invalid or download fails
    """
    # Check if it's a URL (starts with http:// or https://)
    if image_input.startswith(('http://', 'https://')):
        try:
            # Download the image from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_input)
                response.raise_for_status()
                
                img_file = io.BytesIO(response.content)
                img_file.name = name
                return img_file
                
        except Exception as e:
            raise ValueError(f"Failed to download image from URL {image_input}: {str(e)}")
    else:
        try:
            # Assume it's base64 encoded
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if image_input.startswith('data:'):
                image_input = image_input.split(',', 1)[1]
            
            img_data = base64.b64decode(image_input)
            img_file = io.BytesIO(img_data)
            img_file.name = name
            return img_file
            
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate")
    size: str = Field(..., description="The size of the image ('1024x1024', '1024x1792', and '1792x1024')")
    quality: str = Field(..., description="The quality of the image ('standard', 'hd')")
    model: str = Field(..., description="The model to use ('gpt-image-1' is default)")
    images: list[str] = Field(..., description="List of reference images as URLs or base64 encoded strings (empty list for no references)")


class ImageEditInput(BaseModel):
    prompt: str = Field(..., description="The text description of the edit to apply")
    images: list[str] = Field(..., description="List of images as URLs or base64 encoded strings - first is main image, rest are references")
    mask: str = Field(..., description="Mask image as URL or base64 encoded string for selective editing (use empty string for no mask)")
    size: str = Field(..., description="The size of the image ('1024x1024', '1024x1536', and '1536x1024')")
    model: str = Field(..., description="The model to use for editing ('gpt-image-1' or 'dall-e-3')")

@mcp.tool()
async def generate_image(input: ImageGenerationInput) -> dict[str, Any]:
    """Generate an image based on a text prompt using OpenAI's image generation API.
    
    Supports both standard image generation (dall-e-3) and reference image generation (gpt-image-1).
    When using reference images, set model to 'gpt-image-1' and provide base64 encoded images.
    
    Allowed sizes: 1024x1024, 1024x1792, and 1792x1024
    
    Args:
        input: ImageGenerationInput model containing prompt and generation parameters
        
    Returns:
        A dictionary containing the generated image data and metadata
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Generating image with prompt: {input.prompt[:100]}...")
        
        # Set defaults for required fields if empty
        from ..config import get_openai_config
        openai_config = get_openai_config()
        
        # Use defaults if empty strings or None provided
        model_to_use = input.model if input.model and input.model.strip() else openai_config.image_model
        size = input.size if input.size and input.size.strip() else "1024x1024"
        quality = input.quality if input.quality and input.quality.strip() else "standard"
        
        # Check if using reference images with gpt-image-1
        if input.images and len(input.images) > 0 and any(img.strip() for img in input.images):
            if model_to_use != "gpt-image-1":
                logger.warning("Reference images provided but model is not gpt-image-1. Switching to gpt-image-1.")
                model_to_use = "gpt-image-1"
            
            # Process images (URLs or base64) to file-like objects for reference image generation
            image_files = []
            for i, img_input in enumerate(input.images):
                if img_input and img_input.strip():  # Skip empty strings
                    try:
                        img_file = await _process_image_input(img_input, f"reference_{i}.png")
                        image_files.append(img_file)
                        logger.info(f"Processed reference image {i}: {len(img_file.getvalue())} bytes")
                    except Exception as e:
                        raise ValueError(f"Failed to process reference image {i}: {str(e)}")
            
            # Use edit_image_async for reference image generation (no mask needed)
            image_url = await edit_image_async(
                prompt=input.prompt,
                image_files=image_files,
                mask_file=None,
                model=model_to_use,
                size=size
            )
            
        else:
            # Standard image generation without reference images
            image_url = await generate_image_async(
                prompt=input.prompt,
                model=model_to_use,
                size=size,
                quality=quality
            )
        
        return {
            "success": True,
            "prompt": input.prompt,
            "model": model_to_use,
            "size": size,
            "quality": quality,
            "image_url": image_url,
            "message": "Image generated successfully"
        }
        
    except Exception as e:
        logger.error("=== Image Generation Tool Error ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input parameters:")
        logger.error(f"  - Prompt: {input.prompt}")
        logger.error(f"  - Model: {model_to_use if 'model_to_use' in locals() else input.model}")
        logger.error(f"  - Size: {size if 'size' in locals() else input.size}")
        logger.error(f"  - Quality: {quality if 'quality' in locals() else input.quality}")
        logger.error(f"  - Images count: {len(input.images) if input.images else 0}")
        
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
            "model": model_to_use if 'model_to_use' in locals() else input.model,
            "size": size if 'size' in locals() else input.size,
            "quality": quality if 'quality' in locals() else input.quality,
            "message": error_message
        }


@mcp.tool()
async def edit_image(input: ImageEditInput) -> dict[str, Any]:
    """Edit an image based on a text prompt using OpenAI's image editing API.
    
    Supports multiple reference images and optional mask for selective editing.
    Uses the gpt-image-1 model for advanced image editing capabilities.
    
    Args:
        input: ImageEditInput model containing prompt, images, mask, and editing parameters
        
    Returns:
        A dictionary containing the edited image data and metadata
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Editing image with prompt: {input.prompt[:100]}...")
        
        # Set default values if empty strings provided
        size = input.size if input.size and input.size.strip() else "1024x1024"
        model = input.model if input.model and input.model.strip() else "gpt-image-1"
        
        logger.info(f"Using size: {size}, model: {model}")
        
        # Process images (URLs or base64) to file-like objects
        image_files = []
        for i, img_input in enumerate(input.images):
            try:
                img_file = await _process_image_input(img_input, f"image_{i}.png")
                image_files.append(img_file)
                logger.info(f"Processed image {i}: {len(img_file.getvalue())} bytes")
            except Exception as e:
                raise ValueError(f"Failed to process image {i}: {str(e)}")
        
        # Process mask if provided
        mask_file = None
        if input.mask and input.mask.strip():
            try:
                mask_file = await _process_image_input(input.mask, "mask.png")
                logger.info(f"Processed mask: {len(mask_file.getvalue())} bytes")
            except Exception as e:
                raise ValueError(f"Failed to process mask: {str(e)}")
        
        # Edit the image using the async client function
        image_url = await edit_image_async(
            prompt=input.prompt,
            image_files=image_files,
            mask_file=mask_file,
            model=model,
            size=size
        )
        
        return {
            "success": True,
            "prompt": input.prompt,
            "model": model,
            "size": size,
            "images_count": len(input.images),
            "mask_provided": input.mask is not None and input.mask.strip(),
            "image_url": image_url,
            "message": "Image edited successfully"
        }
        
    except Exception as e:
        logger.error("=== Image Editing Tool Error ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input parameters:")
        logger.error(f"  - Prompt: {input.prompt}")
        logger.error(f"  - Model: {model if 'model' in locals() else input.model}")
        logger.error(f"  - Size: {size if 'size' in locals() else input.size}")
        logger.error(f"  - Images count: {len(input.images)}")
        logger.error(f"  - Mask provided: {input.mask is not None and input.mask.strip()}")
        
        # Determine error type for better user feedback
        error_message = "Failed to edit image"
        error_details = str(e)
        
        if "400" in str(e) or "bad request" in str(e).lower():
            error_message = "Invalid request parameters"
            logger.error("This appears to be a 400 Bad Request error. Common causes:")
            logger.error("  - Invalid prompt content (violates content policy)")
            logger.error("  - Invalid image format or size")
            logger.error("  - Invalid mask format or dimensions")
            logger.error("  - Unsupported model or parameters")
            
        elif "timeout" in str(e).lower():
            error_message = "Timeout while editing image"
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
            error_message = "Server error during image editing"
            logger.error("500 Internal Server Error - OpenAI API server issue")
            
        elif "http" in str(e).lower():
            error_message = "Network error during image editing"
            logger.error("HTTP error occurred during API request")
            
        # Log the full error context for debugging
        logger.error(f"Full error context: {error_details}")
        
        return {
            "success": False,
            "error": error_details,
            "prompt": input.prompt,
            "model": model if 'model' in locals() else input.model,
            "size": size if 'size' in locals() else input.size,
            "images_count": len(input.images),
            "mask_provided": input.mask is not None and input.mask.strip(),
            "message": error_message
        }