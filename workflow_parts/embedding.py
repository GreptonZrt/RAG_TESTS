"""
Embedding - Workflow Part

Embedding creation using Azure OpenAI or OpenAI APIs.
"""

import os
from typing import List, Union, Callable
from dataclasses import dataclass
from time import sleep


@dataclass
class EmbeddingItem:
    """Represents a single embedding vector."""
    embedding: List[float]


def get_embedding_client():
    """
    Get the appropriate embedding client (Azure OpenAI or OpenAI).
    
    Returns:
        Client object that can create embeddings
        
    Raises:
        RuntimeError: If no API credentials are configured
    """
    from openai import AzureOpenAI, OpenAI
    
    # Try Azure OpenAI first
    if (os.getenv("AZURE_OPENAI_ENDPOINT") and 
        os.getenv("AZURE_OPENAI_API_KEY") and 
        os.getenv("API_VERSION")):
        return AzureOpenAI(
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
    
    # Fallback to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    raise RuntimeError(
        "No OpenAI/Azure credentials found. "
        "Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + API_VERSION "
        "OR OPENAI_API_KEY"
    )


def create_embeddings(
    texts: Union[str, List[str]],
    deployment_name: str = None,
    batch_size: int = 32,
    max_retries: int = 5
) -> List[EmbeddingItem]:
    """
    Create embeddings for text(s) using configured API.
    
    Args:
        texts: Single text string or list of text strings
        deployment_name: Model/deployment name (uses env var if None)
        batch_size: Number of texts to batch per API call
        max_retries: Maximum retry attempts for failed calls
        
    Returns:
        List[EmbeddingItem]: Embeddings for the input texts
        
    Raises:
        RuntimeError: If all retries fail
    """
    if deployment_name is None:
        deployment_name = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    
    # Normalize input to list
    if isinstance(texts, str):
        input_data = [texts]
    else:
        input_data = list(texts)
    
    # Filter out empty texts
    input_data = [t for t in input_data if t.strip()]
    
    if not input_data:
        print("  WARNING: No valid text to embed")
        return []
    
    client = get_embedding_client()
    all_embeddings = []
    
    def _call_with_retries(batch: List[str]) -> List:
        """Call embedding API with exponential backoff retry."""
        backoff = 1
        for attempt in range(max_retries):
            try:
                # For Azure OpenAI, use 'model' with deployment name; SDK handles it via api_version
                resp = client.embeddings.create(
                    model=deployment_name,
                    input=batch
                )
                return resp.data
            except Exception as e:
                error_msg = str(e)
                print(f"Embedding API call failed (attempt {attempt + 1}/{max_retries})")
                print(f"  Model/Deployment: {deployment_name}")
                print(f"  Error: {error_msg}")
                
                if "404" in error_msg or "NotFoundError" in str(type(e)):
                    print(f"  Hint: Deployment '{deployment_name}' not found. Check Azure Portal.")
                
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Embedding API failed after {max_retries} retries: {error_msg}")
                
                sleep(backoff)
                backoff *= 2
    
    # Process in batches
    for i in range(0, len(input_data), batch_size):
        batch = input_data[i:i + batch_size]
        batch_embeddings = _call_with_retries(batch)
        
        for emb in batch_embeddings:
            all_embeddings.append(EmbeddingItem(embedding=emb.embedding))
    
    return all_embeddings


def get_embedding_fn(deployment_name: str = None) -> Callable[[str], EmbeddingItem]:
    """
    Returns a function that generates embeddings for single text inputs.
    Useful for retrieval functions that need to embed queries one at a time.
    
    Args:
        deployment_name: Model/deployment name (uses env var if None)
        
    Returns:
        Callable that takes a string and returns an EmbeddingItem
        
    Example:
        >>> embed_fn = get_embedding_fn()
        >>> result = embed_fn("Hello world")
        >>> print(len(result.embedding))  # 1536 for text-embedding-3-large
    """
    def embedding_function(text: str) -> EmbeddingItem:
        """Embed a single text string."""
        embeddings = create_embeddings(text, deployment_name=deployment_name)
        if embeddings:
            return embeddings[0]
        else:
            # Return zero vector as fallback
            return EmbeddingItem(embedding=[0.0] * 1536)
    
    return embedding_function
