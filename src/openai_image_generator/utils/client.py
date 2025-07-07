from ..config import get_openrouter_api_config, get_service_api_config, get_openai_config
from typing import Any
import httpx
import asyncio
import logging
import base64
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

def get_openrouter_client(model: OpenRouterModel = OpenRouterModel.GPT41) -> OpenRouterClient:
    """Get the OpenRouter client."""
    return OpenRouterClient(api_key=get_openrouter_api_config().api_key, base_url=get_openrouter_api_config().base_url, model=model)

def get_tool_client(model: OpenRouterModel = OpenRouterModel.GPT41) -> ToolClient:
    """Get the Tool client."""
    return ToolClient(get_openrouter_client(model))


def generate_image(prompt: str, model: str | None = None, size: str = "auto", quality: str = "auto") -> bytes:
    """Generate an image based on a text prompt using OpenAI's image generation API.
    
    Args:
        prompt: The text description of the image to generate
        model: The model to use for generation (default: from config, currently "dall-e-3")
        size: The size of the image ('1024x1024', '1024x1792', '1792x1024', 'auto') (default: "auto")
        quality: The quality of the image ('standard', 'hd') (default: "standard")
        
    Returns:
        The generated image as bytes
        
    Raises:
        Exception: If image generation fails
    """
    try:
        # Get OpenAI configuration
        openai_config = get_openai_config()
        
        # Use configured model if none provided
        if model is None:
            model = openai_config.image_model
        
        # Initialize OpenAI client
        client = OpenAI(
            api_key=openai_config.api_key,
            base_url=openai_config.base_url
        )
        
        # Prepare parameters for API call
        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "response_format": "b64_json"
        }
        
        # Generate the image
        result = client.images.generate(**params)
        
        # Extract and decode the base64 image data
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        return image_bytes
        
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        raise


async def generate_image_async(prompt: str, model: str | None = None, size: str = "auto", quality: str = "auto") -> bytes:
    """Async version of generate_image for use in async contexts.
    
    Args:
        prompt: The text description of the image to generate
        model: The model to use for generation (default: from config, currently "dall-e-3")
        size: The size of the image ('1024x1024', '1536x1024', '1024x1536', 'auto') (default: "auto")
        quality: The quality of the image ('low', 'medium', 'high', 'auto') (default: "auto")
        
    Returns:
        The generated image as bytes
        
    Raises:
        Exception: If image generation fails
    """
    try:
        # Get OpenAI configuration
        openai_config = get_openai_config()
        
        # Use configured model if none provided
        if model is None:
            model = openai_config.image_model
        
        # Initialize async OpenAI client
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
            "response_format": "b64_json"
        }
        
        # Generate the image
        result = await client.images.generate(**params)
        
        # Extract and decode the base64 image data
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        return image_bytes
        
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        raise