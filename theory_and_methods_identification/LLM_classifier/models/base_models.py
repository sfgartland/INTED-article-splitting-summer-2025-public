from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Generic, List, TypeVar
import re




class BaseCategory(BaseModel):
    """Base category model that can be extended for specific use cases"""
    title: str = Field(..., description="Title of the category")
    description: str = Field(..., description="Description of the category")
    
    def __eq__(self, other):
        """Check if two categories are equal based on title and description"""
        if not isinstance(other, BaseCategory):
            return False
        return self.title.lower().strip() == other.title.lower().strip() and \
               self.description.lower().strip() == other.description.lower().strip()
    
    def __hash__(self):
        """Make Category hashable for use in sets and as dictionary keys"""
        return hash((self.title.lower().strip(), self.description.lower().strip()))
    
    def matches_title(self, other_title: str, case_sensitive: bool = False) -> bool:
        """Check if this category's title matches the given title"""
        if case_sensitive:
            return self.title.strip() == other_title.strip()
        return self.title.lower().strip() == other_title.lower().strip()
    
    
    def contains_keywords(self, keywords: List[str], case_sensitive: bool = False) -> bool:
        """
        Check if this category contains any of the given keywords in title or description
        
        Args:
            keywords: List of keywords to search for
            case_sensitive: Whether to perform case-sensitive matching
            
        Returns:
            bool: True if any keyword is found
        """
        text_to_search = [self.title, self.description]
        if not case_sensitive:
            text_to_search = [text.lower() for text in text_to_search]
            keywords = [kw.lower() for kw in keywords]
        
        for keyword in keywords:
            for text in text_to_search:
                if keyword in text:
                    return True
        return False
    
    # class Config:
    #     extra = "forbid"  # Prevent additional fields


CategoryType = TypeVar("CategoryType", bound=BaseCategory)

class ProbabilityScore(BaseModel):
    category: str = Field(..., description="Category title")
    probability: float = Field(..., description="Probability score for the category")

    class Config:
        extra = "forbid"


class BaseClassificationResponse(BaseModel, Generic[CategoryType]):
    """Base model for classification responses. Add your own properties"""
    classifications: List[str] = Field(default_factory=list, description="List of category classifications")
    probabilities: List[ProbabilityScore] = Field(default_factory=list, description="Probability scores for each classification")
    new_categories: List[CategoryType] = Field(default_factory=list, description="New categories not in the existing list")
    
    model_config = ConfigDict(model_title_generator=lambda x: re.sub(r'\[.*?\]', '', x.__name__))
    

    @field_validator('classifications')
    @classmethod
    def validate_classifications(cls, v):
        if not all(isinstance(cat, str) and cat.strip() for cat in v):
            raise ValueError("All classifications must be non-empty strings")
        return v

    @field_validator('probabilities')
    @classmethod
    def validate_probabilities(cls, v):
        if not all(isinstance(prob, ProbabilityScore) for prob in v):
            raise ValueError("All probabilities must be ProbabilityScore instances")
        return v
    

class DiscoveryResponse(BaseModel, Generic[CategoryType]):
    """Model for LLM category discovery responses"""
    categories: List[CategoryType] = Field(..., description="List of discovered categories")
    
    class Config:
        extra = "forbid"  # Prevent additional fields



class Merge(BaseModel):
    """Model for category merges during review"""
    merged_title: str = Field(..., description="Merged category title")
    merged_description: str = Field(..., description="Merged category description")
    frameworks_to_merge: List[str] = Field(..., description="List of category titles that have been merged")
    reasoning: str = Field(..., description="Reasoning for the merge")
    
    class Config:
        extra = "forbid" 

class RemovedCategory(BaseCategory):
    """Model for removed categories during review"""
    reason: str = Field(..., description="Reason for removal")
    
    class Config:
        extra = "forbid"  # Prevent additional fields

RemovedCategoryType = TypeVar("RemovedCategoryType", bound=RemovedCategory)

class ReviewResponse(BaseModel, Generic[RemovedCategoryType]):
    """LLM response model for category review - only contains removal and merge suggestions"""
    removed_categories: List[RemovedCategoryType] = Field(default_factory=list, description="Categories that should be removed")
    merges: List[Merge] = Field(default_factory=list, description="Category merges to be applied")
    reasoning: str = Field(..., description="Explanation of the review process and decisions")
    
    class Config:
        extra = "forbid"  # Prevent additional fields 

class ReviewResult(ReviewResponse, Generic[CategoryType, RemovedCategoryType]):
    """Review result model for both LLM responses and internal results"""
    cleaned_categories: List[CategoryType] = Field(default_factory=list, description="Categories that passed review")


class CategoryTemperatureLevel(BaseModel):
    """Pydantic type for controlling how easily the classifier creates new categories"""
    locked: str = Field(..., description="Instruction for locked mode - never creates new categories")
    strict: str = Field(..., description="Instruction for strict mode - only creates new categories with high confidence")
    balanced: str = Field(..., description="Instruction for balanced mode - creates new categories with moderate confidence")
    creative: str = Field(..., description="Instruction for creative mode - creates new categories easily")
    

