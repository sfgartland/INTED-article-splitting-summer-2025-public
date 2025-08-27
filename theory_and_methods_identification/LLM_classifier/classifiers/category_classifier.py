from abc import ABC
from ..base import BaseClassifier
from ..models.base_models import BaseCategory
from ..models.category_models import Category_DiscoveryResponse, Category_ReviewResponse, Category_ReviewResult, Category_ClassificationResponse



class CategoryClassifier(BaseClassifier[Category_ClassificationResponse, BaseCategory, Category_ReviewResponse, Category_ReviewResult, Category_DiscoveryResponse, BaseCategory], ABC):
    pass