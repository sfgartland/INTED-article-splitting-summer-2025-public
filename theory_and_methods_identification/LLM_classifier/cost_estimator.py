"""
Cost estimation wrapper for BaseClassifier classes.

This module provides utilities to estimate the cost of running classification operations
using OpenAI models before actually executing them.
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any, Type, Union
from dataclasses import dataclass
from datetime import datetime
import tiktoken
from abc import ABC

from .base import BaseClassifier
from .models.base_models import (
    BaseCategory, BaseClassificationResponse, DiscoveryResponse,
    ReviewResponse, ReviewResult
)

# OpenAI pricing as of 2024 (prices per 1M tokens)
# Source: https://openai.com/pricing
OPENAI_PRICING = {
    # GPT-4 models
    "gpt-4o": {"input": 2.50, "output": 10.00},  # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-micro": {"input": 0.25, "output": 1.00},
    "gpt-4o-nano": {"input": 0.25, "output": 1.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1": {"input": 2.00, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4": {"input": 30.00, "output": 60.00},
    
}

# Default model mappings for cost estimation
DEFAULT_MODEL_MAPPINGS = {
    "gpt-4.1-micro": "gpt-4o-micro",
    "gpt-4.1-nano": "gpt-4o-nano",
    "o3-mini": "gpt-4o-mini",
    "o3-micro": "gpt-4o-micro",
    "o3-nano": "gpt-4o-nano",
}


@dataclass
class CostEstimate:
    """Container for cost estimation results."""
    operation_type: str
    model_used: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "model_used": self.model_used,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BatchCostEstimate:
    """Container for batch operation cost estimates."""
    individual_estimates: List[CostEstimate]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    average_cost_per_item: float
    currency: str = "USD"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @classmethod
    def from_estimates(cls, estimates: List[CostEstimate]) -> 'BatchCostEstimate':
        """Create batch estimate from individual estimates."""
        if not estimates:
            return cls([], 0, 0, 0.0, 0.0)
        
        total_input = sum(est.input_tokens for est in estimates)
        total_output = sum(est.output_tokens for est in estimates)
        total_cost = sum(est.total_cost for est in estimates)
        avg_cost = total_cost / len(estimates) if estimates else 0.0
        
        return cls(
            individual_estimates=estimates,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost=total_cost,
            average_cost_per_item=avg_cost,
            currency=estimates[0].currency if estimates else "USD"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "individual_estimates": [est.to_dict() for est in self.individual_estimates],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "average_cost_per_item": self.average_cost_per_item,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat()
        }


class TokenCounter:
    """Utility class for counting tokens in text."""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenization (default: gpt-4)
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4)
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_tokens_in_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        total_tokens = 0
        for message in messages:
            # Add tokens for role and content
            total_tokens += self.count_tokens(message.get("role", ""))
            total_tokens += self.count_tokens(message.get("content", ""))
            # Add overhead for message formatting (approximate)
            total_tokens += 4
        return total_tokens


class CostEstimator:
    """Cost estimation utility for BaseClassifier operations."""
    
    def __init__(self, 
                 pricing_data: Optional[Dict[str, Dict[str, float]]] = None,
                 model_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize cost estimator.
        
        Args:
            pricing_data: Custom pricing data (defaults to OPENAI_PRICING)
            model_mappings: Custom model mappings (defaults to DEFAULT_MODEL_MAPPINGS)
        """
        self.pricing_data = pricing_data or OPENAI_PRICING.copy()
        self.model_mappings = model_mappings or DEFAULT_MODEL_MAPPINGS.copy()
        self.token_counter = TokenCounter()
    
    def get_model_pricing(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a specific model.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with input and output pricing per 1M tokens
            
        Raises:
            ValueError: If model pricing is not available
        """
        # Try direct match first
        if model in self.pricing_data:
            return self.pricing_data[model]
        
        # Try mapped model
        mapped_model = self.model_mappings.get(model)
        if mapped_model and mapped_model in self.pricing_data:
            return self.pricing_data[mapped_model]
        
        # Try pattern matching for model variants
        for pattern, pricing in self.pricing_data.items():
            if model.startswith(pattern) or pattern in model:
                return pricing
        
        raise ValueError(f"No pricing data available for model: {model}")
    
    def estimate_cost(self, 
                     input_tokens: int, 
                     output_tokens: int, 
                     model: str,
                     operation_type: str = "unknown") -> CostEstimate:
        """
        Estimate cost for a single operation.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name
            operation_type: Type of operation (e.g., "classification", "discovery")
            
        Returns:
            CostEstimate object
        """
        pricing = self.get_model_pricing(model)
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostEstimate(
            operation_type=operation_type,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
    
    def estimate_classification_cost(self,
                                   classifier: BaseClassifier,
                                   element_data: Any,
                                   categories: List[BaseCategory],
                                   category_creation_temperature: str = "balanced") -> CostEstimate:
        """
        Estimate cost for a single classification operation.
        
        Args:
            classifier: BaseClassifier instance
            element_data: Data to classify
            categories: List of categories
            category_creation_temperature: Temperature level for category creation
            
        Returns:
            CostEstimate object
        """
        # Get prompts
        system_prompt, user_prompt = classifier.get_classification_prompts(
            element_data, categories, category_creation_temperature
        )
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Count tokens
        input_tokens = self.token_counter.count_tokens_in_messages(messages)
        
        # Estimate output tokens based on response model
        estimated_output = self._estimate_output_tokens()
        
        return self.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=estimated_output,
            model=classifier.general_model,
            operation_type="classification"
        )
    
    def estimate_category_discovery_cost(self,
                                       classifier: BaseClassifier,
                                       text_samples: List[str],
                                       max_context_tokens: int = 128000) -> CostEstimate:
        """
        Estimate cost for category discovery operation.
        
        Args:
            classifier: BaseClassifier instance
            text_samples: List of text samples for discovery
            max_context_tokens: Maximum context tokens
            
        Returns:
            CostEstimate object
        """
        system_prompt = classifier.get_category_discovery_system_prompt()
        
        # Determine how many samples fit in context
        n_samples = classifier._auto_select_sample_count(
            text_samples, system_prompt, max_context_tokens
        )
        
        if n_samples == 0:
            return CostEstimate(
                operation_type="category_discovery",
                model_used=classifier.general_model,
                input_tokens=0,
                output_tokens=0,
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0
            )
        
        # Create messages with actual samples
        messages = [{"role": "system", "content": system_prompt}]
        for i, sample in enumerate(text_samples[:n_samples], 1):
            messages.append({
                "role": "user",
                "content": f"Sample Section {i}:\n{sample.strip()}"
            })
        
        # Count tokens
        input_tokens = self.token_counter.count_tokens_in_messages(messages)
        
        # Estimate output tokens for category discovery
        estimated_output = self._estimate_category_discovery_output(n_samples)
        
        return self.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=estimated_output,
            model=classifier.general_model,
            operation_type="category_discovery"
        )
    
    def estimate_category_review_cost(self,
                                    classifier: BaseClassifier,
                                    categories: List[BaseCategory]) -> CostEstimate:
        """
        Estimate cost for category review operation.
        
        Args:
            classifier: BaseClassifier instance
            categories: List of categories to review
            
        Returns:
            CostEstimate object
        """
        # Get prompts
        system_prompt, user_prompt = classifier.get_category_review_prompts(categories)
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Count tokens
        input_tokens = self.token_counter.count_tokens_in_messages(messages)
        
        # Estimate output tokens for review response
        estimated_output = self._estimate_category_review_output(len(categories))
        
        return self.estimate_cost(
            input_tokens=input_tokens,
            output_tokens=estimated_output,
            model=classifier.reasoning_model,
            operation_type="category_review"
        )
    
    def estimate_batch_classification_cost(self,
                                         classifier: BaseClassifier,
                                         data: List[Any],
                                         categories: List[BaseCategory],
                                         category_creation_temperature: str = "balanced",
                                         max_new_categories: int = 10) -> BatchCostEstimate:
        """
        Estimate cost for batch classification operation.
        
        Args:
            classifier: BaseClassifier instance
            data: List of data to classify
            categories: Initial list of categories
            category_creation_temperature: Temperature level for category creation
            max_new_categories: Maximum number of new categories to expect
            
        Returns:
            BatchCostEstimate object
        """
        estimates = []
        current_categories = categories.copy()
        
        for i, element_data in enumerate(data):
            # Estimate cost for this element
            estimate = self.estimate_classification_cost(
                classifier, element_data, current_categories, category_creation_temperature
            )
            estimates.append(estimate)
            
            # Simulate potential new categories being added
            if i < max_new_categories and len(current_categories) < len(categories) + max_new_categories:
                # Add a mock new category to simulate discovery
                new_category = BaseCategory(
                    title=f"New Category {i+1}",
                    description=f"Automatically discovered category {i+1}"
                )
                current_categories.append(new_category)
        
        return BatchCostEstimate.from_estimates(estimates)
    
    def _estimate_output_tokens(self) -> int:
        """Estimate output tokens for classification response."""
        return 80 # Based on output
    
    def _estimate_category_discovery_output(self, n_samples: int) -> int:
        """Estimate output tokens for category discovery."""
        # Base tokens for JSON structure
        base_tokens = 30
        
        # Tokens for discovered categories (assuming 5-10 categories per discovery)
        estimated_categories = min(10, max(5, n_samples // 2))
        category_tokens = estimated_categories * 80  # 80 tokens per category
        
        return base_tokens + category_tokens
    
    def _estimate_category_review_output(self, n_categories: int) -> int:
        """Estimate output tokens for category review."""
        # Base tokens for JSON structure
        base_tokens = 50
        
        # Tokens for reasoning
        reasoning_tokens = 200
        
        # Tokens for removal suggestions (assuming 20% of categories might be removed)
        removal_tokens = int(n_categories * 0.2) * 60
        
        # Tokens for merge suggestions (assuming 10% of categories might be merged)
        merge_tokens = int(n_categories * 0.1) * 100
        
        return base_tokens + reasoning_tokens + removal_tokens + merge_tokens


class CostEstimationWrapper:
    """
    Wrapper class that provides cost estimation capabilities for BaseClassifier instances.
    
    This wrapper allows you to estimate costs before running actual operations,
    helping with budget planning and optimization.
    """
    
    def __init__(self, 
                 classifier: BaseClassifier,
                 estimator: Optional[CostEstimator] = None):
        """
        Initialize the cost estimation wrapper.
        
        Args:
            classifier: BaseClassifier instance to wrap
            estimator: CostEstimator instance (creates default if None)
        """
        self.classifier = classifier
        self.estimator = estimator or CostEstimator()
        self._cost_history: List[CostEstimate] = []
    
    def estimate_single_classification(self,
                                     element_data: Any,
                                     categories: List[BaseCategory],
                                     category_creation_temperature: str = "balanced") -> CostEstimate:
        """
        Estimate cost for a single classification operation.
        
        Args:
            element_data: Data to classify
            categories: List of categories
            category_creation_temperature: Temperature level for category creation
            
        Returns:
            CostEstimate object
        """
        return self.estimator.estimate_classification_cost(
            self.classifier, element_data, categories, category_creation_temperature
        )
    
    def estimate_batch_classification(self,
                                    data: List[Any],
                                    categories: List[BaseCategory],
                                    category_creation_temperature: str = "balanced",
                                    max_new_categories: int = 10) -> BatchCostEstimate:
        """
        Estimate cost for batch classification operation.
        
        Args:
            data: List of data to classify
            categories: Initial list of categories
            category_creation_temperature: Temperature level for category creation
            max_new_categories: Maximum number of new categories to expect
            
        Returns:
            BatchCostEstimate object
        """
        return self.estimator.estimate_batch_classification_cost(
            self.classifier, data, categories, category_creation_temperature, max_new_categories
        )
    
    def estimate_category_discovery(self,
                                  text_samples: List[str],
                                  max_context_tokens: int = 128000) -> CostEstimate:
        """
        Estimate cost for category discovery operation.
        
        Args:
            text_samples: List of text samples for discovery
            max_context_tokens: Maximum context tokens
            
        Returns:
            CostEstimate object
        """
        return self.estimator.estimate_category_discovery_cost(
            self.classifier, text_samples, max_context_tokens
        )
    
    def estimate_category_review(self,
                               categories: List[BaseCategory]) -> CostEstimate:
        """
        Estimate cost for category review operation.
        
        Args:
            categories: List of categories to review
            
        Returns:
            CostEstimate object
        """
        return self.estimator.estimate_category_review_cost(
            self.classifier, categories
        )
    
    def estimate_full_workflow(self,
                             data: List[Any],
                             initial_categories: List[BaseCategory],
                             text_samples: Optional[List[str]] = None,
                             category_creation_temperature: str = "balanced",
                             max_new_categories: int = 10) -> Dict[str, Union[CostEstimate, BatchCostEstimate]]:
        """
        Estimate cost for a full classification workflow.
        
        This includes:
        1. Category discovery (if text_samples provided)
        2. Category review
        3. Batch classification
        
        Args:
            data: List of data to classify
            initial_categories: Initial list of categories
            text_samples: Optional text samples for category discovery
            category_creation_temperature: Temperature level for category creation
            max_new_categories: Maximum number of new categories to expect
            
        Returns:
            Dictionary with cost estimates for each step
        """
        estimates = {}
        
        # Step 1: Category discovery (if samples provided)
        if text_samples:
            estimates["category_discovery"] = self.estimate_category_discovery(text_samples)
            # Assume discovery adds some categories
            discovered_categories = initial_categories + [
                BaseCategory(title=f"Discovered {i}", description=f"Category {i}")
                for i in range(min(5, len(text_samples) // 2))
            ]
        else:
            discovered_categories = initial_categories
        
        # Step 2: Category review
        estimates["category_review"] = self.estimate_category_review(discovered_categories)
        
        # Step 3: Batch classification
        estimates["batch_classification"] = self.estimate_batch_classification(
            data, discovered_categories, category_creation_temperature, max_new_categories
        )
        
        return estimates
    
    def generate_cost_report(self, estimates: Dict[str, Union[CostEstimate, BatchCostEstimate]]) -> str:
        """
        Generate a human-readable cost report.
        
        Args:
            estimates: Dictionary of cost estimates
            
        Returns:
            Formatted cost report string
        """
        report_lines = []
        report_lines.append("# Cost Estimation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for operation, estimate in estimates.items():
            report_lines.append(f"## {operation.replace('_', ' ').title()}")
            report_lines.append("")
            
            if isinstance(estimate, BatchCostEstimate):
                report_lines.append(f"- **Total Cost**: ${estimate.total_cost:.4f}")
                report_lines.append(f"- **Items Processed**: {len(estimate.individual_estimates)}")
                report_lines.append(f"- **Average Cost per Item**: ${estimate.average_cost_per_item:.4f}")
                report_lines.append(f"- **Total Input Tokens**: {estimate.total_input_tokens:,}")
                report_lines.append(f"- **Total Output Tokens**: {estimate.total_output_tokens:,}")
                total_cost += estimate.total_cost
                total_input_tokens += estimate.total_input_tokens
                total_output_tokens += estimate.total_output_tokens
            else:
                report_lines.append(f"- **Cost**: ${estimate.total_cost:.4f}")
                report_lines.append(f"- **Model**: {estimate.model_used}")
                report_lines.append(f"- **Input Tokens**: {estimate.input_tokens:,}")
                report_lines.append(f"- **Output Tokens**: {estimate.output_tokens:,}")
                total_cost += estimate.total_cost
                total_input_tokens += estimate.input_tokens
                total_output_tokens += estimate.output_tokens
            
            report_lines.append("")
        
        report_lines.append("## Summary")
        report_lines.append("")
        report_lines.append(f"- **Total Estimated Cost**: ${total_cost:.4f}")
        report_lines.append(f"- **Total Input Tokens**: {total_input_tokens:,}")
        report_lines.append(f"- **Total Output Tokens**: {total_output_tokens:,}")
        report_lines.append("")
        report_lines.append("*Note: These are estimates based on current pricing and may vary.*")
        
        return "\n".join(report_lines)
    
    def save_cost_report(self, 
                        estimates: Dict[str, Union[CostEstimate, BatchCostEstimate]], 
                        filename: str) -> None:
        """
        Save cost report to a file.
        
        Args:
            estimates: Dictionary of cost estimates
            filename: Output filename
        """
        report = self.generate_cost_report(estimates)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def get_classifier(self) -> BaseClassifier:
        """Get the wrapped classifier instance."""
        return self.classifier


# Convenience functions for easy usage
def estimate_classification_cost(classifier: BaseClassifier,
                               element_data: Any,
                               categories: List[BaseCategory],
                               category_creation_temperature: str = "balanced") -> CostEstimate:
    """
    Convenience function to estimate classification cost.
    
    Args:
        classifier: BaseClassifier instance
        element_data: Data to classify
        categories: List of categories
        category_creation_temperature: Temperature level for category creation
        
    Returns:
        CostEstimate object
    """
    wrapper = CostEstimationWrapper(classifier)
    return wrapper.estimate_single_classification(element_data, categories, category_creation_temperature)


def estimate_batch_cost(classifier: BaseClassifier,
                       data: List[Any],
                       categories: List[BaseCategory],
                       category_creation_temperature: str = "balanced",
                       max_new_categories: int = 10) -> BatchCostEstimate:
    """
    Convenience function to estimate batch classification cost.
    
    Args:
        classifier: BaseClassifier instance
        data: List of data to classify
        categories: Initial list of categories
        category_creation_temperature: Temperature level for category creation
        max_new_categories: Maximum number of new categories to expect
        
    Returns:
        BatchCostEstimate object
    """
    wrapper = CostEstimationWrapper(classifier)
    return wrapper.estimate_batch_classification(data, categories, category_creation_temperature, max_new_categories) 