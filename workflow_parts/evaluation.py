"""
Evaluation - Workflow Part

Response evaluation using LLM for RAG workflows.
"""

import json
from typing import Dict, Optional


def evaluate_response(
    query: str,
    ai_response: str,
    ideal_answer: Optional[str] = None,
    deployment_name: str = None
) -> Dict:
    """
    Evaluate the quality of an AI-generated response.
    
    Args:
        query: The original query
        ai_response: The AI-generated response
        ideal_answer: The expected/ideal answer (ground truth)
        deployment_name: Model/deployment name for evaluation
        
    Returns:
        Dict: Contains 'score' (0-1 or None), 'reasoning', and any parsing errors
    """
    from workflow_parts.generation import get_generation_client
    import os
    
    if deployment_name is None:
        deployment_name = os.getenv("CHAT_DEPLOYMENT", "gpt-4o")
    
    evaluate_system_prompt = (
        "You are an evaluation system for RAG responses. "
        "Assess if the AI response correctly answers the query based on the provided context. "
        "Respond with ONLY a JSON object in this exact format: "
        '{\"score\": 0 or 0.5 or 1, \"reasoning\": \"your explanation\"}'
    )
    
    ideal_answer_text = f"\n\nGround Truth Answer:\n{ideal_answer}" if ideal_answer else ""
    
    evaluation_prompt = (
        f"Query: {query}\n\n"
        f"AI Response:\n{ai_response}"
        f"{ideal_answer_text}"
    )
    
    try:
        client = get_generation_client()
        
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0,
            messages=[
                {"role": "system", "content": evaluate_system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        
        eval_text = response.choices[0].message.content
        
        # Try to parse JSON response
        try:
            eval_json = json.loads(eval_text)
            return {
                "score": eval_json.get("score"),
                "reasoning": eval_json.get("reasoning", ""),
                "raw_response": eval_text
            }
        except json.JSONDecodeError:
            return {
                "score": None,
                "reasoning": eval_text,
                "raw_response": eval_text,
                "parse_error": "Could not parse JSON from response"
            }
    except Exception as e:
        return {
            "score": None,
            "error": str(e)
        }
