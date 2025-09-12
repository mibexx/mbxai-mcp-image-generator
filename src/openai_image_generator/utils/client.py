from ..config import get_openrouter_api_config, get_service_api_config, get_openai_config, get_image_storage_config
from typing import Any
import httpx
import asyncio
import logging
import os
import uuid
from pathlib import Path
from pydantic import BaseModel

from openai import OpenAI
from mbxai.openrouter import OpenRouterModel, OpenRouterClient
from mbxai.tools import ToolClient

class ServiceApiClient:
    """Client for making API calls to other services using the job system."""
    
    def __init__(self, timeout: int = 3600, max_retries: int = 3, retry_delay: int = 5, poll_interval: int = 10):
        """Initialize the service API client.
        
        Args:
            timeout: Overall timeout in seconds (default: 3600s = 60 minutes)
            max_retries: Maximum number of retries for 503 errors (default: 3)
            retry_delay: Delay between retries in seconds (default: 5)
            poll_interval: Interval in seconds to poll for job status (default: 10)
        """
        service_api_config = get_service_api_config()
        self.base_url = service_api_config.api_url
        self.token = service_api_config.token
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_interval = poll_interval
        self.client = httpx.AsyncClient(timeout=timeout)
        self.logger = logging.getLogger(__name__)
        
    async def call_service(self, namespace: str, service_name: str, endpoint: str, method: str = "POST", data: dict[str, Any] | BaseModel | None = None) -> dict[str, Any]:
        """Call a service endpoint using the job system with polling.
        
        Args:
            namespace: The namespace of the service
            service_name: The name of the service
            endpoint: The endpoint to call
            method: The HTTP method to use
            data: The data to send in the request body, either a dict or Pydantic model
            
        Returns:
            The response data from the job
        """
        # Convert Pydantic model to dict if provided
        if isinstance(data, BaseModel):
            json_data = data.model_dump()
            # Log the request data for debugging
            self.logger.info(f"Request data for {service_name}/{endpoint}: {json_data}")
        else:
            json_data = data
            
        # Step 1: Create a job
        job_id = await self._create_job(namespace, service_name, endpoint, method, json_data)
        self.logger.info(f"Created job {job_id} for {namespace}/{service_name}/{endpoint}")
        
        # Step 2: Poll for job completion
        start_time = asyncio.get_event_loop().time()
        while True:
            # Check if we've exceeded the timeout
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > self.timeout:
                raise TimeoutError(f"Job {job_id} timed out after {self.timeout}s")
                
            # Check job status
            status = await self._get_job_status(job_id)
            if status == "success":
                self.logger.info(f"Job {job_id} completed successfully")
                break
            elif status == "failed":
                self.logger.error(f"Job {job_id} failed")
                raise RuntimeError(f"Job {job_id} failed to complete")
                
            # Wait before polling again
            self.logger.debug(f"Job {job_id} is still running, polling again in {self.poll_interval}s")
            await asyncio.sleep(self.poll_interval)
            
        # Step 3: Get job result
        result = await self._get_job_result(job_id)
        
        # Step 4: Delete the job after successful retrieval
        try:
            await self._delete_job(job_id)
            self.logger.info(f"Successfully deleted job {job_id}")
        except Exception as e:
            # Log warning but don't fail the entire operation
            self.logger.warning(f"Failed to delete job {job_id}: {str(e)}")
        
        return result
    
    async def _create_job(self, namespace: str, service_name: str, endpoint: str, method: str, data: dict[str, Any] | None) -> str:
        """Create a job to execute a service endpoint.
        
        Args:
            namespace: The namespace of the service
            service_name: The name of the service
            endpoint: The endpoint to call
            method: The HTTP method to use
            data: The request data
            
        Returns:
            The job ID
        """
        job_url = f"{self.base_url}/job/{namespace}/{service_name}/api/{endpoint}"
        
        # Set up headers with authentication token
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        # Initialize retry counter
        retries = 0
        
        while True:
            try:
                self.logger.info(f"Creating job at {job_url} (attempt {retries + 1}/{self.max_retries + 1})")
                response = await self.client.request(method, job_url, json=data, headers=headers)
                response.raise_for_status()
                job_data = response.json()
                
                if "job_id" not in job_data:
                    raise ValueError(f"Invalid job response: {job_data}")
                    
                return job_data["job_id"]
                
            except httpx.ReadTimeout:
                self.logger.error(f"Request to {job_url} timed out")
                raise
                
            except httpx.HTTPStatusError as e:
                # Check if it's a 503 Service Unavailable error and we have retries left
                if e.response.status_code == 503 and retries < self.max_retries:
                    retries += 1
                    wait_time = self.retry_delay * retries  # Progressive backoff
                    self.logger.warning(f"Service unavailable (503) for {job_url}, retrying in {wait_time}s (attempt {retries}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                
                self.logger.error(f"HTTP error {e.response.status_code} calling {job_url}")
                raise
                
            except Exception as e:
                self.logger.error(f"Unexpected error creating job at {job_url}: {str(e)}")
                raise
    
    async def _get_job_status(self, job_id: str) -> str:
        """Get the status of a job.
        
        Args:
            job_id: The job ID
            
        Returns:
            The job status ('running', 'success', or 'failed')
        """
        status_url = f"{self.base_url}/job/status/{job_id}"
        
        # Set up headers with authentication token
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        try:
            response = await self.client.get(status_url, headers=headers)
            response.raise_for_status()
            status_data = response.json()
            
            return status_data.get("status", "running")
            
        except Exception as e:
            self.logger.error(f"Error getting status for job {job_id}: {str(e)}")
            raise
    
    async def _get_job_result(self, job_id: str) -> dict[str, Any]:
        """Get the result of a completed job.
        
        Args:
            job_id: The job ID
            
        Returns:
            The job result data
        """
        result_url = f"{self.base_url}/job/result/{job_id}"
        
        # Set up headers with authentication token
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        self.logger.info(f"Fetching job result for job {job_id} from {result_url}")
            
        try:
            response = await self.client.get(result_url, headers=headers)
            self.logger.info(f"Job result status code: {response.status_code}")
            
            # Log headers for debugging
            self.logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Try to log response body regardless of status code for debugging
            try:
                response_data = response.json()
                # Truncate large response data for logging
                log_data = str(response_data)
                if len(log_data) > 1000:
                    log_data = log_data[:1000] + "... [truncated]"
                self.logger.debug(f"Response data: {log_data}")
            except Exception as e:
                self.logger.warning(f"Could not parse response as JSON: {str(e)}")
                self.logger.debug(f"Raw response text: {response.text[:500]}... [truncated]")
            
            # Now raise for status after logging
            response.raise_for_status()
            
            result_data = response.json()
            self.logger.info(f"Successfully retrieved result for job {job_id}")
            return result_data
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error {e.response.status_code} fetching result for job {job_id}")
            try:
                error_data = e.response.json()
                self.logger.error(f"Error response data: {error_data}")
            except Exception:
                self.logger.error(f"Error response text: {e.response.text[:500]}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error getting result for job {job_id}: {str(e)}")
            raise
    
    async def _delete_job(self, job_id: str) -> None:
        """Delete a job after it has been processed.
        
        Args:
            job_id: The job ID to delete
        """
        delete_url = f"{self.base_url}/job/delete/{job_id}"
        
        # Set up headers with authentication token
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        try:
            self.logger.debug(f"Deleting job {job_id} at {delete_url}")
            response = await self.client.delete(delete_url, headers=headers)
            response.raise_for_status()
            self.logger.debug(f"Successfully deleted job {job_id}")
            
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error {e.response.status_code} deleting job {job_id}")
            raise
            
        except Exception as e:
            self.logger.error(f"Error deleting job {job_id}: {str(e)}")
            raise
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def _ensure_upload_directory() -> Path:
    """Ensure the upload directory exists and return its path.
    
    Returns:
        Path object for the upload directory
    """
    config = get_image_storage_config()
    upload_path = Path(config.storage_path)
    upload_path.mkdir(parents=True, exist_ok=True)
    return upload_path


async def _download_and_save_image(image_url: str) -> str:
    """Download an image from a URL and save it locally.
    
    Args:
        image_url: The URL of the image to download
        
    Returns:
        The filename of the saved image
        
    Raises:
        Exception: If image download or save fails
    """
    try:
        # Ensure upload directory exists
        upload_path = _ensure_upload_directory()
        
        # Generate unique filename
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        file_path = upload_path / filename
        
        # Download the image
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            
            # Save the image
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
        logging.info(f"Successfully downloaded and saved image: {filename}")
        return filename
        
    except Exception as e:
        logging.error(f"Error downloading image from {image_url}: {str(e)}")
        raise


def _generate_proxy_url(filename: str) -> str:
    """Generate a proxy URL for accessing a stored image.
    
    Args:
        filename: The filename of the stored image
        
    Returns:
        The proxy URL for accessing the image
    """
    config = get_image_storage_config()
    return f"{config.proxy_domain}{config.proxy_endpoint}/{filename}"


def get_openrouter_client(model: OpenRouterModel = OpenRouterModel.GPT41) -> OpenRouterClient:
    """Get the OpenRouter client."""
    return OpenRouterClient(api_key=get_openrouter_api_config().api_key, base_url=get_openrouter_api_config().base_url, model=model)

def get_tool_client(model: OpenRouterModel = OpenRouterModel.GPT41) -> ToolClient:
    """Get the Tool client."""
    return ToolClient(get_openrouter_client(model))


def generate_image(prompt: str, model: str | None = None, size: str = "1024x1024", quality: str = "standard") -> str:
    """Generate an image based on a text prompt using OpenAI's image generation API.
    
    Args:
        prompt: The text description of the image to generate
        model: The model to use for generation (default: from config, currently "dall-e-3")
        size: The size of the image ('1024x1024', '1024x1536', '1536x1024') (default: "1024x1024")
        quality: The quality of the image ('standard', 'hd') (default: "standard")
        
    Returns:
        The proxy URL of the downloaded and stored image
        
    Raises:
        Exception: If image generation, download, or storage fails
    """
    # Use the async version with asyncio.run for synchronous interface
    return asyncio.run(generate_image_async(prompt, model, size, quality))


def _validate_prompt(prompt: str) -> tuple[bool, str]:
    """Validate a prompt for image generation.
    
    Args:
        prompt: The prompt to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger = logging.getLogger(__name__)
    
    # Check for empty or whitespace-only prompts
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"
    
    # Check prompt length
    if len(prompt) < 3:
        return False, "Prompt is too short (minimum 3 characters)"
    
    if len(prompt) > 4000:
        return False, "Prompt is too long (maximum 4000 characters)"
    
    # Check for potentially problematic content
    problematic_terms = [
        'nude', 'naked', 'explicit', 'sexual', 'violence', 'blood', 'gore',
        'weapon', 'gun', 'knife', 'bomb', 'terrorist', 'hate', 'discrimination'
    ]
    
    prompt_lower = prompt.lower()
    for term in problematic_terms:
        if term in prompt_lower:
            logger.warning(f"Prompt contains potentially problematic term: {term}")
            # Don't reject, just warn - let OpenAI's content filter handle it
    
    # Check for unsupported characters or encoding issues
    try:
        prompt.encode('utf-8')
    except UnicodeEncodeError:
        return False, "Prompt contains unsupported characters"
    
    # Check for excessive repetition
    words = prompt.split()
    if len(words) > 2:
        word_counts = {}
        for word in words:
            word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
        
        for word, count in word_counts.items():
            if count > 5:  # More than 5 repetitions of the same word
                logger.warning(f"Prompt contains excessive repetition of word: {word}")
    
    return True, ""


async def generate_image_async(prompt: str, model: str | None = None, size: str = "1024x1024", quality: str = "standard") -> str:
    """Async version of generate_image for use in async contexts.
    
    Args:
        prompt: The text description of the image to generate
        model: The model to use for generation (default: from config, currently "dall-e-3")
        size: The size of the image ('1024x1024', '1024x1536', '1536x1024') (default: "1024x1024")
        quality: The quality of the image ('standard', 'hd') (default: "standard")
        
    Returns:
        The proxy URL of the downloaded and stored image
        
    Raises:
        Exception: If image generation, download, or storage fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Log input parameters for debugging
        logger.info("=== Image Generation Request ===")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Model: {model}")
        logger.info(f"Size: {size}")
        logger.info(f"Quality: {quality}")
        
        # Validate input parameters
        logger.info("Validating input parameters...")
        
        # Validate prompt using comprehensive validation
        is_valid, error_msg = _validate_prompt(prompt)
        if not is_valid:
            raise ValueError(f"Prompt validation failed: {error_msg}")
        
        # Validate size parameter
        valid_sizes = ['1024x1024', '1024x1536', '1536x1024']
        if size not in valid_sizes:
            logger.warning(f"Invalid size '{size}', using '1024x1024'. Valid sizes: {valid_sizes}")
            size = '1024x1024'
        
        # Validate quality parameter
        valid_qualities = ['standard', 'hd']
        if quality not in valid_qualities:
            logger.warning(f"Invalid quality '{quality}', using 'standard'. Valid qualities: {valid_qualities}")
            quality = 'standard'
        
        logger.info("Input validation completed successfully")
        
        # Get OpenAI configuration
        logger.info("Retrieving OpenAI configuration...")
        openai_config = get_openai_config()
        logger.info(f"OpenAI base URL: {openai_config.base_url}")
        logger.info(f"API key present: {'Yes' if openai_config.api_key else 'No'}")
        
        # Use configured model if none provided
        if model is None:
            model = openai_config.image_model
            logger.info(f"Using configured model: {model}")
        else:
            logger.info(f"Using provided model: {model}")
        
        # Initialize async OpenAI client
        logger.info("Initializing OpenAI client...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url
        )
        
        # Prepare parameters for API call
        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "response_format": "url"
        }
        
        logger.info("=== API Request Parameters ===")
        logger.info(f"Model: {params['model']}")
        logger.info(f"Size: {params['size']}")
        logger.info(f"Quality: {params['quality']}")
        logger.info(f"Response Format: {params['response_format']}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Log prompt content (truncated for privacy)
        if len(prompt) > 200:
            logger.info(f"Prompt preview: {prompt[:200]}...")
        else:
            logger.info(f"Full prompt: {prompt}")
        
        # Generate the image
        logger.info("Making API request to OpenAI...")
        try:
            result = await client.images.generate(**params)
            logger.info("API request completed successfully")
            
        except Exception as api_error:
            logger.error("=== API Request Failed ===")
            logger.error(f"Error type: {type(api_error).__name__}")
            logger.error(f"Error message: {str(api_error)}")
            
            # Log additional error details if available
            if hasattr(api_error, 'response'):
                logger.error(f"Response status: {api_error.response.status_code}")
                logger.error(f"Response headers: {dict(api_error.response.headers)}")
                
                try:
                    error_data = api_error.response.json()
                    logger.error(f"Error response body: {error_data}")
                    
                    # Log specific error details
                    if 'error' in error_data:
                        error_info = error_data['error']
                        logger.error(f"OpenAI error type: {error_info.get('type', 'unknown')}")
                        logger.error(f"OpenAI error code: {error_info.get('code', 'unknown')}")
                        logger.error(f"OpenAI error message: {error_info.get('message', 'unknown')}")
                        
                        # Log parameter validation errors
                        if 'param' in error_info:
                            logger.error(f"Parameter error: {error_info['param']}")
                        
                except Exception as json_error:
                    logger.error(f"Could not parse error response as JSON: {str(json_error)}")
                    logger.error(f"Raw error response: {api_error.response.text[:1000]}")
            
            raise
        
        # Get the original image URL
        if not result.data or len(result.data) == 0:
            raise ValueError("No image data returned from API")
        
        original_image_url = result.data[0].url
        logger.info(f"Successfully generated image URL: {original_image_url}")
        
        # Download and save the image
        logger.info("Downloading and saving image...")
        filename = await _download_and_save_image(original_image_url)
        logger.info(f"Image saved as: {filename}")
        
        # Generate and return proxy URL
        proxy_url = _generate_proxy_url(filename)
        logger.info(f"Image stored locally and accessible via: {proxy_url}")
        logger.info("=== Image Generation Completed Successfully ===")
        
        return proxy_url
        
    except Exception as e:
        logger.error("=== Image Generation Failed ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input parameters - Prompt: {prompt[:100]}..., Model: {model}, Size: {size}, Quality: {quality}")
        
        # Re-raise the exception for the caller to handle
        raise


def edit_image(prompt: str, image_files: list, mask_file=None, model: str | None = None, size: str = "1024x1024") -> str:
    """Edit an image based on a text prompt using OpenAI's image editing API.
    
    Args:
        prompt: The text description of the edit to apply
        image_files: List of file objects - first is main image, rest are references
        mask_file: Optional mask file object for selective editing
        model: The model to use for editing (default: from config, currently "gpt-image-1")
        size: The size of the image ('1024x1024', '1024x1536', '1536x1024') (default: "1024x1024")
        
    Returns:
        The proxy URL of the downloaded and stored edited image
        
    Raises:
        Exception: If image editing, download, or storage fails
    """
    # Use the async version with asyncio.run for synchronous interface
    return asyncio.run(edit_image_async(prompt, image_files, mask_file, model, size))


async def edit_image_async(prompt: str, image_files: list, mask_file=None, model: str | None = None, size: str = "1024x1024") -> str:
    """Async version of edit_image for use in async contexts.
    
    Args:
        prompt: The text description of the edit to apply
        image_files: List of file objects - first is main image, rest are references
        mask_file: Optional mask file object for selective editing
        model: The model to use for editing (default: from config, currently "gpt-image-1")
        size: The size of the image ('1024x1024', '1024x1536', '1536x1024') (default: "1024x1024")
        
    Returns:
        The proxy URL of the downloaded and stored edited image
        
    Raises:
        Exception: If image editing, download, or storage fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Log input parameters for debugging
        logger.info("=== Image Editing Request ===")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Model: {model}")
        logger.info(f"Size: {size}")
        logger.info(f"Number of input images: {len(image_files) if image_files else 0}")
        logger.info(f"Mask provided: {'Yes' if mask_file else 'No'}")
        
        # Validate input parameters
        logger.info("Validating input parameters...")
        
        # Validate prompt using comprehensive validation
        is_valid, error_msg = _validate_prompt(prompt)
        if not is_valid:
            raise ValueError(f"Prompt validation failed: {error_msg}")
        
        # Validate image files
        if not image_files or len(image_files) == 0:
            raise ValueError("At least one image file is required")
        
        # Validate size parameter
        valid_sizes = ['1024x1024', '1024x1536', '1536x1024']
        if size not in valid_sizes:
            logger.warning(f"Invalid size '{size}', using '1024x1024'. Valid sizes: {valid_sizes}")
            size = '1024x1024'
        
        logger.info("Input validation completed successfully")
        
        # Get OpenAI configuration
        logger.info("Retrieving OpenAI configuration...")
        openai_config = get_openai_config()
        logger.info(f"OpenAI base URL: {openai_config.base_url}")
        logger.info(f"API key present: {'Yes' if openai_config.api_key else 'No'}")
        
        # Use gpt-image-1 model for editing if none provided
        if model is None:
            model = "gpt-image-1"  # Default to gpt-image-1 for editing
            logger.info(f"Using default editing model: {model}")
        else:
            logger.info(f"Using provided model: {model}")
        
        # Initialize async OpenAI client
        logger.info("Initializing OpenAI client...")
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url
        )
        
        # Prepare parameters for API call
        params = {
            "model": model,
            "prompt": prompt,
            "image": image_files,
            "size": size
        }
        
        # Add mask if provided
        if mask_file:
            params["mask"] = mask_file
        
        logger.info("=== API Request Parameters ===")
        logger.info(f"Model: {params['model']}")
        logger.info(f"Size: {params['size']}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        logger.info(f"Images count: {len(image_files)}")
        logger.info(f"Mask provided: {'Yes' if mask_file else 'No'}")
        
        # Log prompt content (truncated for privacy)
        if len(prompt) > 200:
            logger.info(f"Prompt preview: {prompt[:200]}...")
        else:
            logger.info(f"Full prompt: {prompt}")
        
        # Edit the image
        logger.info("Making API request to OpenAI for image editing...")
        try:
            result = await client.images.edit(**params)
            logger.info("API request completed successfully")
            
        except Exception as api_error:
            logger.error("=== API Request Failed ===")
            logger.error(f"Error type: {type(api_error).__name__}")
            logger.error(f"Error message: {str(api_error)}")
            
            # Log additional error details if available
            if hasattr(api_error, 'response'):
                logger.error(f"Response status: {api_error.response.status_code}")
                logger.error(f"Response headers: {dict(api_error.response.headers)}")
                
                try:
                    error_data = api_error.response.json()
                    logger.error(f"Error response body: {error_data}")
                    
                    # Log specific error details
                    if 'error' in error_data:
                        error_info = error_data['error']
                        logger.error(f"OpenAI error type: {error_info.get('type', 'unknown')}")
                        logger.error(f"OpenAI error code: {error_info.get('code', 'unknown')}")
                        logger.error(f"OpenAI error message: {error_info.get('message', 'unknown')}")
                        
                        # Log parameter validation errors
                        if 'param' in error_info:
                            logger.error(f"Parameter error: {error_info['param']}")
                        
                except Exception as json_error:
                    logger.error(f"Could not parse error response as JSON: {str(json_error)}")
                    logger.error(f"Raw error response: {api_error.response.text[:1000]}")
            
            raise
        
        # Get the edited image URL
        if not result.data or len(result.data) == 0:
            raise ValueError("No image data returned from API")
        
        edited_image_url = result.data[0].url
        logger.info(f"Successfully edited image URL: {edited_image_url}")
        
        # Download and save the image
        logger.info("Downloading and saving edited image...")
        filename = await _download_and_save_image(edited_image_url)
        logger.info(f"Edited image saved as: {filename}")
        
        # Generate and return proxy URL
        proxy_url = _generate_proxy_url(filename)
        logger.info(f"Edited image stored locally and accessible via: {proxy_url}")
        logger.info("=== Image Editing Completed Successfully ===")
        
        return proxy_url
        
    except Exception as e:
        logger.error("=== Image Editing Failed ===")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Input parameters - Prompt: {prompt[:100]}..., Model: {model}, Size: {size}")
        
        # Re-raise the exception for the caller to handle
        raise