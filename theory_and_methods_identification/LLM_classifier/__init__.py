"""
Section Classifier Module

A modular Python system for classifying different types of academic sections using OpenAI's GPT models.

This module provides:
- Base classes for section classification
- Specific implementations for different section types
- Category discovery and review functionality
- Batch processing capabilities
- Meta-category creation functionality
- Pydantic models for type-safe validation
- Cost estimation utilities for budget planning
"""

from .models.base_models import (
    BaseCategory,
    BaseClassificationResponse,
    DiscoveryResponse,
    ReviewResponse,
    RemovedCategory,
    Merge,
    ProbabilityScore,
    ReviewResult,
    CategoryTemperatureLevel
)
from .models.category_models import (
    Category_ClassificationResponse,
)
from .models.section_models import (
    Section,
    Section_ClassificationResponse,
    TextExcerpt
)
from .base import BaseClassifier
from .classifiers.section_classifier import SectionClassifier
from .classifiers.category_classifier import CategoryClassifier
from .utils import (
    load_categories_from_json, 
    save_categories_to_json,
    load_initial_theoretical_frameworks,
    create_timestamped_path
)
from .cost_estimator import (
    CostEstimator,
    CostEstimationWrapper,
    CostEstimate,
    BatchCostEstimate,
    TokenCounter,
    estimate_classification_cost,
    estimate_batch_cost
)

__version__ = "0.1.0"
__author__ = "INTED Article Splitting Project"

__all__ = [
    # Models
    "BaseCategory",
    "Section",
    "TextExcerpt",
    "ProbabilityScore",
    "ReviewResult",
    "CategoryTemperatureLevel",
    
    # Response models for validation
    "BaseClassificationResponse",
    "DiscoveryResponse", 
    "Category_ClassificationResponse",
    "Section_ClassificationResponse",
    "ReviewResponse",

    "RemovedCategory",
    "Merge",
    
    # Base classes
    "BaseClassifier",
    "SectionClassifier",
    "CategoryClassifier",
    
    # Concrete classifier implementations (may be None if parent module not available)
    "TheoreticalFrameworkClassifier",
    "FrameworkClassifier",
    
    # Utilities
    "load_categories_from_json",
    "save_categories_to_json",
    "load_initial_theoretical_frameworks",
    "create_timestamped_path",
    
    # Cost estimation
    "CostEstimator",
    "CostEstimationWrapper", 
    "CostEstimate",
    "BatchCostEstimate",
    "TokenCounter",
    "estimate_classification_cost",
    "estimate_batch_cost",
] 