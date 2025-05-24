#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Any, TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("byterover_service")


class ByteroverError(Exception):
    """Custom exception for ByteRover API errors"""
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"ByteRover API Error ({status}): {message}")


class Message(TypedDict):
    """Message structure for ByteRover API"""
    role: str  # "user" or "assistant"
    content: str


class MemoryResult(TypedDict, total=False):
    """Structure for a memory result from ByteRover API"""
    id: Optional[str]
    memory: str
    score: float
    hash: Optional[str]
    createdAt: Optional[str]
    updatedAt: Optional[str]
    bookmarked: Optional[bool]
    comments: Optional[List[Any]]
    metadata: Optional[Dict[str, Any]]
    projectId: Optional[str]


class Relation(TypedDict):
    """Structure for a relation between memories"""
    source: str
    relationship: str
    destination: str


class SearchResponse(TypedDict):
    """Structure for search response from ByteRover API"""
    results: List[MemoryResult]
    relations: Optional[List[Relation]]


class ByteroverService:
    """Service for interacting with the ByteRover API"""
    
    def __init__(self, api_key: str = "", user_id: str = ""):
        """Initialize the ByteRover service
        
        Args:
            api_key: The ByteRover public API key
            user_id: The user ID
        """
        self.api_key = api_key
        self.user_id = user_id
        self.base_url = "https://api.byterover.dev/api/v1"
        self.is_configured = bool(api_key and user_id)
    
    def update_credentials(self, api_key: str, user_id: str) -> bool:
        """Update the ByteRover API key and user ID
        
        Args:
            api_key: The ByteRover public API key
            user_id: The user ID
            
        Returns:
            True if the configuration was updated successfully
        """
        if not api_key or not user_id:
            return False
        
        self.api_key = api_key
        self.user_id = user_id
        self.is_configured = True
        logger.info("ByteRover service credentials updated")
        return True
    
    def is_service_configured(self) -> bool:
        """Check if the service is configured with valid credentials
        
        Returns:
            True if the service has valid credentials
        """
        return self.is_configured
    
    async def _request(self, endpoint: str, method: str = "GET", body: Optional[Dict[str, Any]] = None) -> Any:
        """Make a request to the ByteRover API
        
        Args:
            endpoint: The API endpoint to call
            method: The HTTP method to use
            body: The request body
            
        Returns:
            The response data
            
        Raises:
            ByteroverError: If the API returns an error
            Exception: For other errors
        """
        import aiohttp
        
        if not self.is_configured:
            raise ValueError(
                "ByteRover service is not configured with valid credentials. "
                "Please set the API key and user ID first."
            )
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            logger.info(f"Calling {url}")
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body
                ) as response:
                    if not response.ok:
                        raise ByteroverError(
                            response.status,
                            response.reason or "Unknown error"
                        )
                    
                    return await response.json()
        except ByteroverError:
            raise
        except Exception as e:
            raise Exception(f"Failed to make request to ByteRover API: {str(e)}")
    
    def _request_sync(self, endpoint: str, method: str = "GET", body: Optional[Dict[str, Any]] = None) -> Any:
        """Synchronous version of _request
        
        Args:
            endpoint: The API endpoint to call
            method: The HTTP method to use
            body: The request body
            
        Returns:
            The response data
            
        Raises:
            ByteroverError: If the API returns an error
            Exception: For other errors
        """
        import requests
        
        if not self.is_configured:
            raise ValueError(
                "ByteRover service is not configured with valid credentials. "
                "Please set the API key and user ID first."
            )
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            logger.info(f"Calling {url}")
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=body
            )
            
            if not response.ok:
                raise ByteroverError(
                    response.status_code,
                    response.reason or "Unknown error"
                )
            
            return response.json()
        except ByteroverError:
            raise
        except Exception as e:
            raise Exception(f"Failed to make request to ByteRover API: {str(e)}")
    
    def search_memories(self, query: str, limit: int = 5) -> SearchResponse:
        """Search for memories
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            The search response
        """
        if not self.is_configured:
            logger.warning("Cannot search memories: ByteRover service is not configured with valid credentials")
            return {"results": []}
        
        endpoint = "/memories/search"
        body = {
            "query": query,
            "limit": limit,
            "userId": self.user_id
        }
        
        return self._request_sync(endpoint, method="POST", body=body)
    
    def create_memory(self, messages: List[Message]) -> None:
        """Create a memory
        
        Args:
            messages: The messages to create a memory from
        """
        if not self.is_configured:
            logger.warning("Cannot create memory: ByteRover service is not configured with valid credentials")
            return
        
        endpoint = "/memories"
        body = {
            "messages": messages,
            "userId": self.user_id
        }
        
        return self._request_sync(endpoint, method="POST", body=body)
