"""
Generation - Workflow Part

Response generation using LLM for RAG workflows.
"""

import os
from typing import Dict, Optional


def get_generation_client():
    """
    Get the appropriate client for response generation (Azure OpenAI or OpenAI).
    
    Returns:
        Client object that can create chat completions
        
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


def generate_response(
    query: str,
    context_chunks: list,
    deployment_name: str = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0
) -> Dict:
    """
    Generate a response using LLM based on query and context.
    
    Args:
        query: The user query
        context_chunks: List of retrieved context chunks
        deployment_name: Model/deployment name (uses env var if None)
        system_prompt: System prompt for the LLM
        temperature: Temperature for response generation
        
    Returns:
        Dict: Contains 'content' key with the generated response
    """
    if deployment_name is None:
        deployment_name = os.getenv("CHAT_DEPLOYMENT", "gpt-4o")
    
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant that strictly answers based on the given context. "
            "If the answer cannot be derived directly from the provided context, "
            "respond with: 'I do not have enough information to answer that.'"
        )
    
    # Format context
    context_text = "\n".join([
        f"Context {i + 1}:\n{chunk}\n" 
        for i, chunk in enumerate(context_chunks)
    ])
    
    user_message = f"{context_text}\nQuestion: {query}"
    
    client = get_generation_client()
    
    # For both Azure and standard OpenAI, use 'model' parameter
    # The client type (AzureOpenAI vs OpenAI) determines how it's handled internally
    response = client.chat.completions.create(
        model=deployment_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    return {
        "content": response.choices[0].message.content,
        "response_obj": response
    }
