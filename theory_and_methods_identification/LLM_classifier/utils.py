"""
Utility functions for the section classifier module.

This module contains helper functions and utilities used throughout the section classifier system.
"""

import os
import json
from typing import List
from datetime import datetime
from .models.base_models import BaseCategory


def load_initial_theoretical_frameworks() -> List[BaseCategory]:
    """
    Load the initial theoretical frameworks as Category objects from the example JSON file.
    
    Returns:
        List of Category objects representing common theoretical frameworks in physics education research
    """
    example_path = os.path.join(os.path.dirname(__file__), "examples", "theory-categories.json")
    try:
        with open(example_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [BaseCategory(**item) for item in data]
    except Exception as e:
        raise RuntimeError(f"Failed to load initial theoretical frameworks from {example_path}: {e}")


def load_categories_from_json(file_path: str) -> List[BaseCategory]:
    """
    Load categories from a JSON file with Pydantic validation.
    
    Args:
        file_path: Path to the JSON file containing categories
    
    Returns:
        List of Category objects with validated data
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid or doesn't match expected structure
        RuntimeError: For other loading errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            categories_data = json.load(f)
        
        if not isinstance(categories_data, list):
            raise ValueError("JSON file must contain a list of categories")
        
        categories = [BaseCategory(**cat_data) for cat_data in categories_data]
        print(f"Successfully loaded {len(categories)} categories from {file_path}")
        return categories
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Category file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading categories from {file_path}: {str(e)}")

def create_timestamped_path(path: str) -> str:
        # Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, ext = os.path.splitext(path)
        timestamped_file_path = f"{base_name}_{timestamp}{ext}"

        return timestamped_file_path



def save_categories_to_json(categories: List[BaseCategory], file_path: str) -> None:
    """
    Save categories to a JSON file with timestamp in filename.
    
    Args:
        categories: List of Category objects to save
        file_path: Path to the JSON file where categories will be saved
                   (timestamp will be added before the .json extension)
    
    Raises:
        RuntimeError: If saving fails
    """
    try:
        categories_data = [cat.model_dump() for cat in categories]

        timestamped_file_path = create_timestamped_path(file_path)

        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(timestamped_file_path, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2, ensure_ascii=False)
        print(f"Categories saved to {timestamped_file_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving categories to {file_path}: {str(e)}") 