import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import chi2_contingency, fisher_exact


def calculate_additional_metrics(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calculate precision, recall, F1-score, and support for each class.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        labels: List of label names (optional)
    
    Returns:
        Dictionary containing precision, recall, F1-score, and support metrics
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Calculate micro average
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    metrics = {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'weighted': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        }
    }
    
    return metrics


def perform_statistical_tests(confusion_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical tests on the confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix as DataFrame
    
    Returns:
        Dictionary containing statistical test results
    """
    # Remove 'All' row and column for statistical tests
    cm_for_tests = confusion_matrix.iloc[:-1, :-1]
    
    results = {}
    
    # Chi-square test for independence
    try:
        chi2_stat, chi2_p_value, chi2_dof, chi2_expected = chi2_contingency(cm_for_tests.values)
        results['chi_square'] = {
            'statistic': chi2_stat,
            'p_value': chi2_p_value,
            'degrees_of_freedom': chi2_dof,
            'significant': chi2_p_value < 0.05
        }
    except Exception as e:
        results['chi_square'] = {'error': str(e)}
    
    # Fisher's exact test (for 2x2 matrices)
    if cm_for_tests.shape == (2, 2):
        try:
            fisher_stat, fisher_p_value = fisher_exact(cm_for_tests.values)
            results['fisher_exact'] = {
                'statistic': fisher_stat,
                'p_value': fisher_p_value,
                'significant': fisher_p_value < 0.05
            }
        except Exception as e:
            results['fisher_exact'] = {'error': str(e)}
    
    return results 