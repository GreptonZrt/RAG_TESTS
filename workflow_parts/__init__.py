"""
Data Loading - Workflow Part

Handles PDF extraction and initial data loading for RAG workflows.
"""

import fitz
from typing import Optional


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from all pages of the PDF
    """
    pdf_file = fitz.open(pdf_path)
    all_text = ""
    
    for page_num in range(pdf_file.page_count):
        page = pdf_file[page_num]
        text = page.get_text("text")
        all_text += text
    
    return all_text


def load_validation_data(val_file: str) -> list:
    """
    Load validation queries from a JSON file.
    
    Args:
        val_file: Path to JSON file containing validation queries
        
    Returns:
        list: List of query dictionaries with 'question' and 'ideal_answer' keys
    """
    import json
    
    with open(val_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def extract_queries_from_validation_data(validation_data: list) -> list:
    """
    Extract just the questions from validation data.
    
    Args:
        validation_data: List of validation query dicts
        
    Returns:
        list: List of question strings
    """
    return [item['question'] for item in validation_data]
