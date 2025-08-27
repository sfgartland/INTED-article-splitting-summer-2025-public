from typing import Dict, Any
from datetime import datetime


def format_evaluation_results(
    evaluation_results: Dict[str, Any],
    output_format: str = "console",
    include_statistical_tests: bool = True
) -> str:
    """
    Format evaluation results for different output formats.
    
    Args:
        evaluation_results: Results from evaluate_classification_accuracy
        output_format: Format type ('console', 'markdown', 'html')
        include_statistical_tests: Whether to include statistical tests
    
    Returns:
        Formatted string output
    """
    if output_format == "markdown":
        return _format_markdown(evaluation_results, include_statistical_tests)
    elif output_format == "html":
        return _format_html(evaluation_results, include_statistical_tests)
    else:  # console format
        return _format_console(evaluation_results, include_statistical_tests)


def _format_console(evaluation_results: Dict[str, Any], include_statistical_tests: bool) -> str:
    """Format results for console output."""
    output = []
    
    # Title
    report_title = evaluation_results.get('report_title', 'CLASSIFICATION ACCURACY EVALUATION')
    output.append(f"=== {report_title.upper()} ===\n")
    
    # Input information
    output.append("=== INPUT INFORMATION ===")
    input_source = evaluation_results.get('input_source', 'Not specified')
    if input_source != "in memory DataFrame":
        output.append(f"Data file: {input_source}")
    else:
        output.append(f"Data source: {input_source}")
    output.append(f"Classifications JSON: {evaluation_results.get('classifications_json_path', 'Not specified')}")
    output.append("Parameters:")
    input_params = evaluation_results.get('input_parameters', {})
    for param, value in input_params.items():
        output.append(f"  {param}: {value}")
    output.append("")
    
    # Overall accuracy
    output.append(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.3f} "
                 f"({evaluation_results['correct_predictions']}/{evaluation_results['total_predictions']})")
    
    # Additional metrics
    additional_metrics = evaluation_results['additional_metrics']
    output.append(f"\nMacro Average Metrics:")
    output.append(f"  Precision: {additional_metrics['macro']['precision']:.3f}")
    output.append(f"  Recall: {additional_metrics['macro']['recall']:.3f}")
    output.append(f"  F1-Score: {additional_metrics['macro']['f1']:.3f}")
    
    output.append(f"\nWeighted Average Metrics:")
    output.append(f"  Precision: {additional_metrics['weighted']['precision']:.3f}")
    output.append(f"  Recall: {additional_metrics['weighted']['recall']:.3f}")
    output.append(f"  F1-Score: {additional_metrics['weighted']['f1']:.3f}")
    
    # Per-category accuracy
    output.append("\n=== PER-CATEGORY ACCURACY ===")
    for category, metrics in evaluation_results['category_metrics'].items():
        output.append(f"{category}: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']})")
    
    # Misclassification patterns
    output.append("\n=== TOP MISCLASSIFICATION PATTERNS ===")
    for (true_label, pred_label), count in evaluation_results['misclassification_pairs'].head(10).items():
        output.append(f"{true_label} → {pred_label}: {count} times")
    
    # Mapping issues
    if evaluation_results['mapping_issues']:
        output.append("\n=== MAPPING ISSUES ===")
        for issue in evaluation_results['mapping_issues']:
            output.append(issue)
    
    # Statistical tests
    if include_statistical_tests and evaluation_results['statistical_tests']:
        output.append("\n=== STATISTICAL TESTS ===")
        stats = evaluation_results['statistical_tests']
        
        if 'chi_square' in stats and 'error' not in stats['chi_square']:
            chi2 = stats['chi_square']
            output.append(f"Chi-square test: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.3f}")
            output.append(f"  Significant: {'Yes' if chi2['significant'] else 'No'}")
        
        if 'fisher_exact' in stats and 'error' not in stats['fisher_exact']:
            fisher = stats['fisher_exact']
            output.append(f"Fisher's exact test: statistic = {fisher['statistic']:.3f}, p = {fisher['p_value']:.3f}")
            output.append(f"  Significant: {'Yes' if fisher['significant'] else 'No'}")
    
    output.append("\n=== CONFUSION MATRIX ===")
    output.append("format: AI category(hand label)")
    
    # Distributions
    output.append("\n=== TRUE LABEL DISTRIBUTION ===")
    output.append(str(evaluation_results['true_label_distribution']))
    
    output.append("\n=== CLASSIFICATION DISTRIBUTION ===")
    output.append(str(evaluation_results['category_distribution']))
    
    # Metrics summary
    output.append("\n=== METRICS SUMMARY ===")
    output.append("Understanding the Metrics:")
    output.append("- Precision: Of the instances predicted as positive, what fraction was actually positive?")
    output.append("- Recall: Of the actual positive instances, what fraction was correctly predicted?")
    output.append("- F1-Score: Harmonic mean of precision and recall, providing a balanced measure.")
    output.append("")
    output.append("Macro vs Weighted Averages:")
    output.append("- Macro Average: Simple average across all classes. Each class contributes equally regardless of size.")
    output.append("- Weighted Average: Average weighted by the number of true instances for each class. Larger classes have more influence.")
    output.append("")
    output.append("When to Use Each:")
    output.append("- Use Macro when all classes are equally important, regardless of class imbalance.")
    output.append("- Use Weighted when you want to account for class imbalance and give more weight to larger classes.")
    
    # Plot information
    plots_created = evaluation_results.get('plots_created', {})
    plot_info = evaluation_results.get('plot_info', {})
    
    if plots_created.get('confusion_matrix') or plots_created.get('metrics_summary'):
        output.append("\n=== PLOTS GENERATED ===")
        if plots_created.get('confusion_matrix'):
            if 'confusion_matrix' in plot_info:
                output.append(f"Confusion Matrix: {plot_info['confusion_matrix']}")
            else:
                output.append("Confusion Matrix: Displayed (not saved)")
        if plots_created.get('metrics_summary'):
            if 'metrics_summary' in plot_info:
                output.append(f"Metrics Summary: {plot_info['metrics_summary']}")
            else:
                output.append("Metrics Summary: Displayed (not saved)")
    
    return "\n".join(output)


def _format_markdown(evaluation_results: Dict[str, Any], include_statistical_tests: bool) -> str:
    """Format results for markdown output."""
    output = []
    
    # Header
    report_title = evaluation_results.get('report_title', 'Classification Accuracy Evaluation Report')
    output.append(f"# {report_title}")
    output.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    output.append("")
    
    # Input information
    output.append("## Input Information")
    output.append("")
    output.append("| Parameter | Value |")
    output.append("|-----------|-------|")
    input_source = evaluation_results.get('input_source', 'Not specified')
    if input_source != "in memory DataFrame":
        output.append(f"| Data file | {input_source} |")
    else:
        output.append(f"| Data source | {input_source} |")
    output.append(f"| Classifications JSON | {evaluation_results.get('classifications_json_path', 'Not specified')} |")
    output.append("")
    
    input_params = evaluation_results.get('input_parameters', {})
    if input_params:
        output.append("### Parameters")
        output.append("| Parameter | Value |")
        output.append("|-----------|-------|")
        for param, value in input_params.items():
            output.append(f"| {param} | {value} |")
        output.append("")
    
    # Overall accuracy
    output.append("## Overall Performance")
    output.append("")
    accuracy = evaluation_results['overall_accuracy']
    correct = evaluation_results['correct_predictions']
    total = evaluation_results['total_predictions']
    output.append(f"**Overall Accuracy:** {accuracy:.1%} ({correct}/{total})")
    output.append("")
    
    # Additional metrics
    additional_metrics = evaluation_results['additional_metrics']
    output.append("### Macro Average Metrics")
    output.append("| Metric | Value |")
    output.append("|--------|-------|")
    output.append(f"| Precision | {additional_metrics['macro']['precision']:.3f} |")
    output.append(f"| Recall | {additional_metrics['macro']['recall']:.3f} |")
    output.append(f"| F1-Score | {additional_metrics['macro']['f1']:.3f} |")
    output.append("")
    
    output.append("### Weighted Average Metrics")
    output.append("| Metric | Value |")
    output.append("|--------|-------|")
    output.append(f"| Precision | {additional_metrics['weighted']['precision']:.3f} |")
    output.append(f"| Recall | {additional_metrics['weighted']['recall']:.3f} |")
    output.append(f"| F1-Score | {additional_metrics['weighted']['f1']:.3f} |")
    output.append("")
    
    # Per-category accuracy
    output.append("## Per-Category Performance")
    output.append("")
    output.append("| Category | Accuracy | Correct/Total |")
    output.append("|----------|----------|---------------|")
    for category, metrics in evaluation_results['category_metrics'].items():
        output.append(f"| {category} | {metrics['accuracy']:.1%} | {metrics['correct']}/{metrics['total']} |")
    output.append("")
    
    # Misclassification patterns
    output.append("## Top Misclassification Patterns")
    output.append("")
    output.append("| True Label | Predicted Label | Count |")
    output.append("|------------|-----------------|-------|")
    for (true_label, pred_label), count in evaluation_results['misclassification_pairs'].head(10).items():
        output.append(f"| {true_label} | {pred_label} | {count} |")
    output.append("")
    
    # Mapping issues
    if evaluation_results['mapping_issues']:
        output.append("## Mapping Issues")
        output.append("")
        for issue in evaluation_results['mapping_issues']:
            output.append(f"- {issue}")
        output.append("")
    
    # Statistical tests
    if include_statistical_tests and evaluation_results['statistical_tests']:
        output.append("## Statistical Tests")
        output.append("")
        stats = evaluation_results['statistical_tests']
        
        if 'chi_square' in stats and 'error' not in stats['chi_square']:
            chi2 = stats['chi_square']
            output.append(f"**Chi-square test:** χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.3f}")
            output.append(f"*Significant:* {'Yes' if chi2['significant'] else 'No'}")
            output.append("")
        
        if 'fisher_exact' in stats and 'error' not in stats['fisher_exact']:
            fisher = stats['fisher_exact']
            output.append(f"**Fisher's exact test:** statistic = {fisher['statistic']:.3f}, p = {fisher['p_value']:.3f}")
            output.append(f"*Significant:* {'Yes' if fisher['significant'] else 'No'}")
            output.append("")
    
    # Distributions
    output.append("## Data Distributions")
    output.append("")
    
    output.append("### True Label Distribution")
    output.append("```")
    output.append(str(evaluation_results['true_label_distribution']))
    output.append("```")
    output.append("")
    
    output.append("### Classification Distribution")
    output.append("```")
    output.append(str(evaluation_results['category_distribution']))
    output.append("```")
    output.append("")
    
    # Metrics summary
    output.append("## Metrics Summary")
    output.append("")
    output.append("### Understanding the Metrics")
    output.append("")
    output.append("- **Precision**: Of the instances predicted as positive, what fraction was actually positive?")
    output.append("- **Recall**: Of the actual positive instances, what fraction was correctly predicted?")
    output.append("- **F1-Score**: Harmonic mean of precision and recall, providing a balanced measure.")
    output.append("")
    output.append("### Macro vs Weighted Averages")
    output.append("")
    output.append("- **Macro Average**: Simple average across all classes. Each class contributes equally regardless of size.")
    output.append("- **Weighted Average**: Average weighted by the number of true instances for each class. Larger classes have more influence.")
    output.append("")
    output.append("### When to Use Each")
    output.append("")
    output.append("- Use **Macro** when all classes are equally important, regardless of class imbalance.")
    output.append("- Use **Weighted** when you want to account for class imbalance and give more weight to larger classes.")
    output.append("")
    
    # Plot information
    plots_created = evaluation_results.get('plots_created', {})
    plot_info = evaluation_results.get('plot_info', {})
    
    if plots_created.get('confusion_matrix') or plots_created.get('metrics_summary'):
        output.append("## Plots Generated")
        output.append("")
        if plots_created.get('confusion_matrix'):
            if 'confusion_matrix' in plot_info:
                output.append(f"**Confusion Matrix:** ![Confusion Matrix]({plot_info['confusion_matrix']})")
            else:
                output.append("**Confusion Matrix:** Displayed (not saved)")
            output.append("")
        if plots_created.get('metrics_summary'):
            if 'metrics_summary' in plot_info:
                output.append(f"**Metrics Summary:** ![Metrics Summary]({plot_info['metrics_summary']})")
            else:
                output.append("**Metrics Summary:** Displayed (not saved)")
            output.append("")
    
    return "\n".join(output)


def _format_html(evaluation_results: Dict[str, Any], include_statistical_tests: bool) -> str:
    """Format results for HTML output."""
    output = []
    
    # HTML header
    output.append("<!DOCTYPE html>")
    output.append("<html>")
    output.append("<head>")
    output.append("<title>Classification Evaluation Report</title>")
    output.append("<style>")
    output.append("body { font-family: Arial, sans-serif; margin: 40px; }")
    output.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
    output.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    output.append("th { background-color: #f2f2f2; }")
    output.append(".metric { font-weight: bold; }")
    output.append(".section { margin: 30px 0; }")
    output.append("</style>")
    output.append("</head>")
    output.append("<body>")
    
    # Title
    report_title = evaluation_results.get('report_title', 'Classification Accuracy Evaluation Report')
    output.append(f"<h1>{report_title}</h1>")
    output.append(f"<p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
    
    # Input information
    output.append('<div class="section">')
    output.append("<h2>Input Information</h2>")
    output.append("<table>")
    output.append("<tr><th>Parameter</th><th>Value</th></tr>")
    input_source = evaluation_results.get('input_source', 'Not specified')
    if input_source != "in memory DataFrame":
        output.append(f"<tr><td>Data file</td><td>{input_source}</td></tr>")
    else:
        output.append(f"<tr><td>Data source</td><td>{input_source}</td></tr>")
    output.append(f"<tr><td>Classifications JSON</td><td>{evaluation_results.get('classifications_json_path', 'Not specified')}</td></tr>")
    output.append("</table>")
    output.append("</div>")
    
    input_params = evaluation_results.get('input_parameters', {})
    if input_params:
        output.append('<div class="section">')
        output.append("<h3>Parameters</h3>")
        output.append("<table>")
        output.append("<tr><th>Parameter</th><th>Value</th></tr>")
        for param, value in input_params.items():
            output.append(f"<tr><td>{param}</td><td>{value}</td></tr>")
        output.append("</table>")
        output.append("</div>")
    
    # Overall accuracy
    output.append('<div class="section">')
    output.append("<h2>Overall Performance</h2>")
    accuracy = evaluation_results['overall_accuracy']
    correct = evaluation_results['correct_predictions']
    total = evaluation_results['total_predictions']
    output.append(f'<p class="metric">Overall Accuracy: {accuracy:.1%} ({correct}/{total})</p>')
    output.append("</div>")
    
    # Additional metrics
    additional_metrics = evaluation_results['additional_metrics']
    output.append('<div class="section">')
    output.append("<h3>Macro Average Metrics</h3>")
    output.append("<table>")
    output.append("<tr><th>Metric</th><th>Value</th></tr>")
    output.append(f"<tr><td>Precision</td><td>{additional_metrics['macro']['precision']:.3f}</td></tr>")
    output.append(f"<tr><td>Recall</td><td>{additional_metrics['macro']['recall']:.3f}</td></tr>")
    output.append(f"<tr><td>F1-Score</td><td>{additional_metrics['macro']['f1']:.3f}</td></tr>")
    output.append("</table>")
    output.append("</div>")
    
    # Per-category accuracy
    output.append('<div class="section">')
    output.append("<h2>Per-Category Performance</h2>")
    output.append("<table>")
    output.append("<tr><th>Category</th><th>Accuracy</th><th>Correct/Total</th></tr>")
    for category, metrics in evaluation_results['category_metrics'].items():
        output.append(f"<tr><td>{category}</td><td>{metrics['accuracy']:.1%}</td><td>{metrics['correct']}/{metrics['total']}</td></tr>")
    output.append("</table>")
    output.append("</div>")
    
    # Misclassification patterns
    output.append('<div class="section">')
    output.append("<h2>Top Misclassification Patterns</h2>")
    output.append("<table>")
    output.append("<tr><th>True Label</th><th>Predicted Label</th><th>Count</th></tr>")
    for (true_label, pred_label), count in evaluation_results['misclassification_pairs'].head(10).items():
        output.append(f"<tr><td>{true_label}</td><td>{pred_label}</td><td>{count}</td></tr>")
    output.append("</table>")
    output.append("</div>")
    
    # Statistical tests
    if include_statistical_tests and evaluation_results['statistical_tests']:
        output.append('<div class="section">')
        output.append("<h2>Statistical Tests</h2>")
        stats = evaluation_results['statistical_tests']
        
        if 'chi_square' in stats and 'error' not in stats['chi_square']:
            chi2 = stats['chi_square']
            output.append(f"<p><strong>Chi-square test:</strong> χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.3f}</p>")
            output.append(f"<p><em>Significant:</em> {'Yes' if chi2['significant'] else 'No'}</p>")
        
        if 'fisher_exact' in stats and 'error' not in stats['fisher_exact']:
            fisher = stats['fisher_exact']
            output.append(f"<p><strong>Fisher's exact test:</strong> statistic = {fisher['statistic']:.3f}, p = {fisher['p_value']:.3f}</p>")
            output.append(f"<p><em>Significant:</em> {'Yes' if fisher['significant'] else 'No'}</p>")
        output.append("</div>")
    
    # Metrics summary
    output.append('<div class="section">')
    output.append("<h2>Metrics Summary</h2>")
    output.append("<h3>Understanding the Metrics</h3>")
    output.append("<ul>")
    output.append("<li><strong>Precision:</strong> Of the instances predicted as positive, what fraction was actually positive?</li>")
    output.append("<li><strong>Recall:</strong> Of the actual positive instances, what fraction was correctly predicted?</li>")
    output.append("<li><strong>F1-Score:</strong> Harmonic mean of precision and recall, providing a balanced measure.</li>")
    output.append("</ul>")
    output.append("<h3>Macro vs Weighted Averages</h3>")
    output.append("<ul>")
    output.append("<li><strong>Macro Average:</strong> Simple average across all classes. Each class contributes equally regardless of size.</li>")
    output.append("<li><strong>Weighted Average:</strong> Average weighted by the number of true instances for each class. Larger classes have more influence.</li>")
    output.append("</ul>")
    output.append("<h3>When to Use Each</h3>")
    output.append("<ul>")
    output.append("<li>Use <strong>Macro</strong> when all classes are equally important, regardless of class imbalance.</li>")
    output.append("<li>Use <strong>Weighted</strong> when you want to account for class imbalance and give more weight to larger classes.</li>")
    output.append("</ul>")
    output.append("</div>")
    
    # Plot information
    plots_created = evaluation_results.get('plots_created', {})
    plot_info = evaluation_results.get('plot_info', {})
    
    if plots_created.get('confusion_matrix') or plots_created.get('metrics_summary'):
        output.append('<div class="section">')
        output.append("<h2>Plots Generated</h2>")
        if plots_created.get('confusion_matrix'):
            if 'confusion_matrix' in plot_info:
                output.append(f"<h3>Confusion Matrix</h3>")
                output.append(f'<img src="{plot_info["confusion_matrix"]}" alt="Confusion Matrix" style="max-width: 100%; height: auto;">')
            else:
                output.append("<h3>Confusion Matrix</h3>")
                output.append("<p>Displayed (not saved)</p>")
        if plots_created.get('metrics_summary'):
            if 'metrics_summary' in plot_info:
                output.append(f"<h3>Metrics Summary</h3>")
                output.append(f'<img src="{plot_info["metrics_summary"]}" alt="Metrics Summary" style="max-width: 100%; height: auto;">')
            else:
                output.append("<h3>Metrics Summary</h3>")
                output.append("<p>Displayed (not saved)</p>")
        output.append("</div>")
    
    output.append("</body>")
    output.append("</html>")
    
    return "\n".join(output) 