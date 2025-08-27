import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import chi2_contingency, fisher_exact
from scipy import stats
from datetime import datetime

# Import from other modules
from .metrics import calculate_additional_metrics, perform_statistical_tests
from .visualization import create_enhanced_confusion_matrix_plot, create_metrics_summary_plot
# from .probability_analysis import calculate_probability_estimates, analyze_probability_distributions
from .formatters import format_evaluation_results
from .validators import load_and_validate_categories, CategoryItem

def evaluate_classification_accuracy(
    merged_df: pd.DataFrame,
    classifications_json: List[CategoryItem],
    *,
    pred_col: str,
    true_label_col: str,
    section_title_col: str,
    include_statistical_tests: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the classification algorithm and analyze patterns.
    Args:
        merged_df: DataFrame with prediction and true label columns
        classifications_json: JSON array with category and true_label_equivalent mappings
        pred_col (str): Column name for predicted category
        true_label_col (str): Column name for true/ground truth labels
        section_title_col (str): Column name for section title (for misclassification analysis)
        include_statistical_tests (bool): Whether to include statistical significance tests
    Returns:
        dict: Dictionary containing various accuracy metrics and analysis
    """
    # Create mapping from true_label_equivalent to category
    true_label_to_category = {}
    category_to_true_label = {}
    for item in classifications_json:
        if item.true_label_equivalent:  # Only add if true_label_equivalent is not None
            true_label_to_category[item.true_label_equivalent] = f"{item.category}"
            category_to_true_label[item.category] = item.true_label_equivalent
    # Filter out rows where we have both classification and true label
    valid_df = merged_df.dropna(subset=[pred_col, true_label_col]).copy()
    # Map true labels to their corresponding categories for comparison
    valid_df['true_label_category'] = valid_df[true_label_col].map(true_label_to_category)
    # Calculate basic accuracy
    correct_predictions = (valid_df[pred_col] == valid_df['true_label_category']).sum()
    total_predictions = len(valid_df)
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    # Create confusion matrix with all categories included
    all_categories = set(valid_df['true_label_category'].unique()) | set(valid_df[pred_col].unique())
    all_categories = [cat for cat in all_categories if pd.notna(cat)]
    confusion_matrix = pd.crosstab(
        valid_df['true_label_category'],
        valid_df[pred_col],
        margins=True,
        dropna=False
    )
    for category in all_categories:
        if category not in confusion_matrix.index:
            confusion_matrix.loc[category] = 0
        if category not in confusion_matrix.columns:
            confusion_matrix[category] = 0
    confusion_matrix = confusion_matrix.reindex(sorted(all_categories + ['All']), axis=0)
    confusion_matrix = confusion_matrix.reindex(sorted(all_categories + ['All']), axis=1)
    row_labels = []
    for label in confusion_matrix.index[:-1]:
        if label in category_to_true_label:
            row_labels.append(f"{label} ({category_to_true_label[label]})")
        else:
            row_labels.append(label)
    row_labels.append('All')
    col_labels = []
    for label in confusion_matrix.columns[:-1]:
        if label in category_to_true_label:
            col_labels.append(f"{label} ({category_to_true_label[label]})")
        else:
            col_labels.append(label)
    col_labels.append('All')
    confusion_matrix.index = row_labels
    confusion_matrix.columns = col_labels
    # Calculate additional metrics (precision, recall, F1-score)
    y_true = valid_df['true_label_category'].tolist()
    y_pred = valid_df[pred_col].tolist()
    additional_metrics = calculate_additional_metrics(y_true, y_pred, labels=all_categories)
    # Per-category metrics
    category_metrics = {}
    for category in valid_df['true_label_category'].unique():
        if pd.isna(category):
            continue
        category_mask = valid_df['true_label_category'] == category
        category_total = category_mask.sum()
        category_correct = (valid_df[category_mask][pred_col] == category).sum()
        category_metrics[category] = {
            'total': category_total,
            'correct': category_correct,
            'accuracy': category_correct / category_total if category_total > 0 else 0
        }
    # Analyze misclassification patterns
    misclassified = valid_df[valid_df[pred_col] != valid_df['true_label_category']]
    misclassification_pairs = misclassified.groupby(['true_label_category', pred_col]).size().sort_values(ascending=False)
    mapping_issues = []
    unmapped_true_labels = set(valid_df[true_label_col].unique()) - set(true_label_to_category.keys())
    unmapped_categories = set(valid_df[pred_col].unique()) - set(category_to_true_label.keys())
    if unmapped_true_labels:
        mapping_issues.append(f"Unmapped true labels: {unmapped_true_labels}")
    if unmapped_categories:
        mapping_issues.append(f"Unmapped categories: {unmapped_categories}")
    true_label_distribution = valid_df[true_label_col].value_counts()
    category_distribution = valid_df[pred_col].value_counts()
    if section_title_col in misclassified.columns:
        misclassified_sections = misclassified[[section_title_col, 'true_label_category', pred_col]].groupby(
            ['true_label_category', pred_col]
        ).agg({section_title_col: lambda x: list(x.unique())}).reset_index()
    else:
        misclassified_sections = None
    statistical_tests = {}
    if include_statistical_tests:
        statistical_tests = perform_statistical_tests(confusion_matrix)
    results = {
        'overall_accuracy': overall_accuracy,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'confusion_matrix': confusion_matrix,
        'category_metrics': category_metrics,
        'additional_metrics': additional_metrics,
        'misclassification_pairs': misclassification_pairs,
        'mapping_issues': mapping_issues,
        'true_label_distribution': true_label_distribution,
        'category_distribution': category_distribution,
        'misclassified_sections': misclassified_sections,
        'true_label_to_category_mapping': true_label_to_category,
        'statistical_tests': statistical_tests
    }
    return results


def run_classification_evaluation(
    df: Union[pd.DataFrame, Union[str, Path]],
    *,
    pred_col: str,
    true_label_col: str,
    section_title_col: str,
    probability_dist_col: str,
    classifications_json_path: str,
    show_plot: bool = True,
    include_statistical_tests: bool = True,
    create_summary_plots: bool = True,
    save_report: bool = False,
    output_format: str = "markdown",
    report_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run classification evaluation and display results.
    
    Args:
        df: DataFrame with classification results and true labels, or path to pickle file
        classifications_json_path: Path to the classifications JSON file
        show_plot: Whether to display the confusion matrix plot
        pred_col: Column name for predicted category
        true_label_col: Column name for true/ground truth labels
        section_title_col: Column name for section title (for misclassification analysis)
        probability_dist_col: Column name for probability distribution (mandatory for future probability-based evaluation)
        include_statistical_tests: Whether to include statistical significance tests
        create_summary_plots: Whether to create additional summary plots
        save_report: Whether to save the report and plots to a timestamped folder
        output_format: Output format ('console', 'markdown', 'html')
        report_name: Custom name for the report and output folder (timestamp will be appended)
    
    Returns:
        dict: Evaluation results
    """
    # Handle input data - either DataFrame or pickle file path
    input_source = "in memory DataFrame"
    if isinstance(df, (str, Path)):
        # Load DataFrame from pickle file
        df_path = Path(df)
        input_source = df_path.name  # Just the filename and extension
        try:
            df = pd.read_pickle(df_path)
        except Exception as e:
            raise ValueError(f"Failed to load DataFrame from pickle file '{df_path}': {e}")
    
    classifications_json = load_and_validate_categories(classifications_json_path)

    evaluation_results = evaluate_classification_accuracy(
        df,
        classifications_json,
        pred_col=pred_col,
        true_label_col=true_label_col,
        section_title_col=section_title_col,
        include_statistical_tests=include_statistical_tests
    )
    
    # Add input information to evaluation results
    evaluation_results['input_source'] = input_source
    evaluation_results['classifications_json_path'] = classifications_json_path
    evaluation_results['input_parameters'] = {
        'pred_col': pred_col,
        'true_label_col': true_label_col,
        'section_title_col': section_title_col,
        'probability_dist_col': probability_dist_col,
        'include_statistical_tests': include_statistical_tests
    }
    
    # Prepare output directory and timestamp if saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if report_name:
        safe_report_name = report_name.replace(' ', '_')
        folder_prefix = f"{safe_report_name}_{timestamp}"
    else:
        folder_prefix = f"evaluation_report_{timestamp}"
    output_dir = None
    plot_info = {}
    if save_report:
        output_dir = Path(folder_prefix)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots and track plot information
    confusion_labels = [label for label in evaluation_results['confusion_matrix'].index if label != 'All']
    category_labels = []
    for label in confusion_labels:
        if '(' in label:
            category_labels.append(label.split(' (')[0])
        else:
            category_labels.append(label)
    category_labels = sorted(list(set(category_labels)))
    
    # Save plots if requested, and always use relative paths in report
    if show_plot or save_report:
        confusion_matrix_path = None
        if save_report:
            confusion_matrix_path = output_dir / f"confusion_matrix_{timestamp}.png"
        create_enhanced_confusion_matrix_plot(
            evaluation_results['confusion_matrix'],
            save_path=str(confusion_matrix_path) if confusion_matrix_path else None
        )
        if confusion_matrix_path:
            plot_info['confusion_matrix'] = confusion_matrix_path.name
    
    if create_summary_plots or save_report:
        metrics_summary_path = None
        if save_report:
            metrics_summary_path = output_dir / f"metrics_summary_{timestamp}.png"
        create_metrics_summary_plot(
            evaluation_results['additional_metrics'],
            category_labels,
            save_path=str(metrics_summary_path) if metrics_summary_path else None
        )
        if metrics_summary_path:
            plot_info['metrics_summary'] = metrics_summary_path.name
    
    # Add plot information to evaluation results
    evaluation_results['plot_info'] = plot_info
    evaluation_results['plots_created'] = {
        'confusion_matrix': show_plot or save_report,
        'metrics_summary': create_summary_plots or save_report
    }
    # Add report title for formatters
    if report_name:
        display_title = report_name.replace('_', ' ').strip()
    else:
        display_title = "Classification Accuracy Evaluation Report"
    evaluation_results['report_title'] = f"{display_title} ({timestamp})"
    
    # Format output
    formatted_output = format_evaluation_results(
        evaluation_results, 
        output_format=output_format,
        include_statistical_tests=include_statistical_tests
    )
    
    # Save report after plots are created
    if save_report:
        if output_format == "markdown":
            report_file = output_dir / f"{folder_prefix}.md"
        elif output_format == "html":
            report_file = output_dir / f"{folder_prefix}.html"
        else:
            report_file = output_dir / f"{folder_prefix}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f"\nReport and plots saved to: {output_dir}")
    
    # Display output
    if output_format == "console":
        print(formatted_output)
    elif output_format == "markdown":
        from IPython.display import Markdown
        display(Markdown(formatted_output))
    elif output_format == "html":
        from IPython.display import HTML
        display(HTML(formatted_output))
    
    return evaluation_results 