import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def calculate_probability_estimates(
    distribution: Dict[str, float],
    *,
    include_entropy: bool = True,
    include_confidence_intervals: bool = True,
    confidence_level: float = 0.95,
    include_mode: bool = True,
    include_variance: bool = True
) -> Dict[str, Any]:
    """
    Calculate various estimates based on a probability distribution.
    
    Args:
        distribution: Dictionary mapping categories to probabilities
        include_entropy: Whether to calculate Shannon entropy
        include_confidence_intervals: Whether to calculate confidence intervals
        confidence_level: Confidence level for intervals (default: 0.95)
        include_mode: Whether to identify the most likely category
        include_variance: Whether to calculate variance and standard deviation
    
    Returns:
        Dictionary containing various probability estimates
    """
    if not distribution:
        raise ValueError("Distribution cannot be empty")
    
    # Validate that probabilities sum to approximately 1
    total_prob = sum(distribution.values())
    if not np.isclose(total_prob, 1.0, atol=1e-6):
        raise ValueError(f"Probabilities must sum to 1.0, got {total_prob:.6f}")
    
    categories = list(distribution.keys())
    probabilities = list(distribution.values())
    
    estimates = {
        'categories': categories,
        'probabilities': probabilities,
        'total_probability': total_prob,
        'num_categories': len(categories)
    }
    
    # Most likely category (mode)
    if include_mode:
        max_prob_idx = np.argmax(probabilities)
        estimates['mode'] = {
            'category': categories[max_prob_idx],
            'probability': probabilities[max_prob_idx]
        }
    
    # Expected value (weighted average)
    estimates['expected_value'] = {
        'category': categories[max_prob_idx],  # For categorical data, mode is the expected value
        'probability': probabilities[max_prob_idx]
    }
    
    # Variance and standard deviation
    if include_variance:
        mean_prob = np.mean(probabilities)
        variance = np.sum([(p - mean_prob) ** 2 for p in probabilities]) / len(probabilities)
        std_dev = np.sqrt(variance)
        
        estimates['variance'] = {
            'value': variance,
            'std_deviation': std_dev,
            'coefficient_of_variation': std_dev / mean_prob if mean_prob > 0 else 0
        }
    
    # Shannon entropy (measure of uncertainty)
    if include_entropy:
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        max_entropy = np.log2(len(categories))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        estimates['entropy'] = {
            'value': entropy,
            'max_possible': max_entropy,
            'normalized': normalized_entropy,
            'uncertainty_level': 'high' if normalized_entropy > 0.7 else 'medium' if normalized_entropy > 0.3 else 'low'
        }
    
    # Confidence intervals (using bootstrap-like approach)
    if include_confidence_intervals:
        # For categorical data, we can estimate confidence intervals for the mode probability
        mode_prob = probabilities[max_prob_idx]
        n_simulated = 1000  # Number of simulated samples
        
        # Bootstrap confidence interval for the mode probability
        bootstrap_samples = np.random.binomial(n_simulated, mode_prob, size=10000) / n_simulated
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)
        
        estimates['confidence_intervals'] = {
            'level': confidence_level,
            'mode_probability': {
                'lower': max(0, ci_lower),
                'upper': min(1, ci_upper),
                'point_estimate': mode_prob
            }
        }
    
    # Additional useful metrics
    estimates['distribution_characteristics'] = {
        'is_uniform': np.allclose(probabilities, [1/len(categories)] * len(categories), atol=1e-6),
        'has_single_peak': len([p for p in probabilities if p == max(probabilities)]) == 1,
        'probability_range': {
            'min': min(probabilities),
            'max': max(probabilities),
            'range': max(probabilities) - min(probabilities)
        }
    }
    
    return estimates


def analyze_probability_distributions(
    distributions: List[Dict[str, float]],
    categories: Optional[List[str]] = None,
    *,
    include_aggregate_stats: bool = True,
    include_comparison: bool = True
) -> Dict[str, Any]:
    """
    Analyze multiple probability distributions and provide aggregate statistics.
    
    Args:
        distributions: List of probability distribution dictionaries
        categories: Optional list of all possible categories (for consistent analysis)
        include_aggregate_stats: Whether to calculate aggregate statistics
        include_comparison: Whether to compare distributions
    
    Returns:
        Dictionary containing aggregate analysis results
    """
    if not distributions:
        raise ValueError("At least one distribution must be provided")
    
    # Get all unique categories if not provided
    if categories is None:
        all_categories = set()
        for dist in distributions:
            all_categories.update(dist.keys())
        categories = sorted(list(all_categories))
    
    analysis = {
        'num_distributions': len(distributions),
        'categories': categories,
        'individual_estimates': []
    }
    
    # Calculate estimates for each distribution
    for i, dist in enumerate(distributions):
        try:
            estimates = calculate_probability_estimates(dist)
            estimates['distribution_index'] = i
            analysis['individual_estimates'].append(estimates)
        except Exception as e:
            analysis['individual_estimates'].append({
                'distribution_index': i,
                'error': str(e)
            })
    
    # Aggregate statistics
    if include_aggregate_stats and len(analysis['individual_estimates']) > 1:
        valid_estimates = [est for est in analysis['individual_estimates'] if 'error' not in est]
        
        if valid_estimates:
            # Aggregate entropy statistics
            entropies = [est['entropy']['value'] for est in valid_estimates]
            analysis['aggregate_stats'] = {
                'entropy': {
                    'mean': np.mean(entropies),
                    'std': np.std(entropies),
                    'min': np.min(entropies),
                    'max': np.max(entropies)
                },
                'mode_probability': {
                    'mean': np.mean([est['mode']['probability'] for est in valid_estimates]),
                    'std': np.std([est['mode']['probability'] for est in valid_estimates])
                }
            }
    
    # Distribution comparison
    if include_comparison and len(analysis['individual_estimates']) > 1:
        valid_estimates = [est for est in analysis['individual_estimates'] if 'error' not in est]
        
        if valid_estimates:
            # Compare modes across distributions
            modes = [est['mode']['category'] for est in valid_estimates]
            mode_counts = pd.Series(modes).value_counts()
            
            analysis['comparison'] = {
                'mode_consistency': {
                    'most_common_mode': mode_counts.index[0] if len(mode_counts) > 0 else None,
                    'mode_frequency': mode_counts.iloc[0] / len(modes) if len(mode_counts) > 0 else 0,
                    'mode_distribution': mode_counts.to_dict()
                },
                'entropy_variability': {
                    'low_entropy_count': sum(1 for est in valid_estimates if est['entropy']['normalized'] < 0.3),
                    'medium_entropy_count': sum(1 for est in valid_estimates if 0.3 <= est['entropy']['normalized'] <= 0.7),
                    'high_entropy_count': sum(1 for est in valid_estimates if est['entropy']['normalized'] > 0.7)
                }
            }
    
    return analysis 