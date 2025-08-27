from pydantic import BaseModel, Field
from typing import Generic, List, TypeVar
from .base_models import BaseCategory, BaseClassificationResponse, DiscoveryResponse, ReviewResponse, RemovedCategory, ReviewResult


class Category_ClassificationResponse(BaseClassificationResponse[BaseCategory]):
    """Model for LLM section classification responses"""
    category: str = Field(..., description="Category title")


    class Config:
        extra = "forbid"



# These needs to be defined like this because of a problem with passing raw instantiated generic types into the openai api
class Category_ReviewResponse(ReviewResponse[RemovedCategory]):
    pass

class Category_ReviewResult(ReviewResult[BaseCategory, RemovedCategory]):
    pass

class Category_DiscoveryResponse(DiscoveryResponse[BaseCategory]):
    pass
