"""
Pydantic models for the section classifier module.

This module contains all the data models used throughout the section classifier system.
"""

from typing import Generic, List, TypeVar
from pydantic import BaseModel, Field, field_validator
from .base_models import (
    BaseCategory, BaseClassificationResponse, DiscoveryResponse, ReviewResponse, CategoryType, RemovedCategory, ReviewResult
)





class TextExcerpt(BaseModel):
    """Model for text excerpts that support a classification"""
    category: str = Field(..., description="Category title this excerpt supports")
    excerpt: str = Field(..., description="Text excerpt that supports the classification")
    relevance_score: float = Field(..., description="Relevance score for this excerpt (0.0-1.0)")
    
    class Config:
        extra = "forbid"



class Section_ClassificationResponse(BaseClassificationResponse[BaseCategory]):
    """Model for LLM section classification responses"""
    text_excerpts: List[TextExcerpt] = Field(default_factory=list, description="Text excerpts supporting each classification")

    @field_validator('text_excerpts')
    @classmethod
    def validate_text_excerpts(cls, v):
        if not all(isinstance(excerpt, TextExcerpt) for excerpt in v):
            raise ValueError("All text excerpts must be TextExcerpt instances")
        return v
    
    class Config:
        extra = "forbid"


class Section(BaseModel):
    """Model for a section of text"""
    article_id: str = Field(..., description="ID of the article the section belongs to")
    title: str = Field(..., description="Title of the section")
    text: str = Field(..., description="Text of the section")

    

    class Config:
        extra = "forbid"  # Prevent additional fields


# These needs to be defined like this because of a problem with passing raw instantiated generic types into the openai api
class Section_ReviewResponse(ReviewResponse[RemovedCategory]):
    pass

class Section_ReviewResult(ReviewResult[BaseCategory, RemovedCategory]):
    pass

class Section_DiscoveryResponse(DiscoveryResponse[BaseCategory]):
    pass

