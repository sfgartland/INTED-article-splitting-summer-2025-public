# Enhanced Classification Evaluation

This module provides comprehensive evaluation tools for classification tasks, with support for multiple metrics, statistical testing, enhanced visualization, and flexible output options.

## How to Use

### Required Data Format

Your data should be in one of these formats:

1. **DataFrame object** with the following columns:
   - **Prediction column**: Contains predicted category labels (e.g., 'Theory', 'Methodology')
   - **True label column**: Contains true/ground truth labels (e.g., 'theoretical_framework', 'methods')
   - **Probability distribution column**: Contains probability distributions for each prediction (mandatory for future probability-based evaluation)
   - **Section title column** (optional): For detailed misclassification analysis

2. **Pickle file path** containing a DataFrame with the same structure

### Classifications JSON Format

You need a JSON file that maps your true labels to the predicted categories:

```json
[
    {
        "category": "Theory",
        "true_label_equivalent": "theoretical_framework"
        ...
    },
    {
        "category": "Methodology", 
        "true_label_equivalent": "methods"
        ...
    },
    ...
]
```

### Basic Usage

```python
from section_classification.evaluation.evaluation import run_classification_evaluation

# Using DataFrame object
results = run_classification_evaluation(
    df=your_dataframe,
    pred_col='classification_highest_prob',
    true_label_col='hand_labels',
    section_title_col='section_title',
    probability_dist_col='classification_probabilities',
    classifications_json_path='path/to/classifications.json'
)

# Using pickle file
results = run_classification_evaluation(
    df='path/to/your_data.pkl',
    pred_col='classification_highest_prob',
    true_label_col='hand_labels',
    section_title_col='section_title',
    probability_dist_col='classification_probabilities',
    classifications_json_path='path/to/classifications.json'
)
```

**Note:** Using the pickle file input is recomended as it lets us include the data soruce name in the final report.

### Advanced Usage with Custom Report

```python
# Generate a comprehensive report with custom name
results = run_classification_evaluation(
    df=your_dataframe,
    pred_col='classification_highest_prob',
    true_label_col='hand_labels',
    section_title_col='section_title',
    probability_dist_col='classification_probabilities',
    classifications_json_path='path/to/classifications.json',
    save_report=True,
    output_format='markdown',
    report_name='My Custom Analysis Report',
    include_statistical_tests=True,
    create_summary_plots=True
)
```

This will create a folder named `My_Custom_Analysis_Report_YYYYMMDD_HHMMSS/` containing:
- `My_Custom_Analysis_Report_YYYYMMDD_HHMMSS.md` - The main report
- `confusion_matrix_YYYYMMDD_HHMMSS.png` - Confusion matrix visualization
- `metrics_summary_YYYYMMDD_HHMMSS.png` - Metrics summary plot

## Dependencies
The enhanced evaluation module requires:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Plotting
- `seaborn` - Enhanced plotting
- `scikit-learn` - Metrics calculation
- `scipy` - Statistical tests
- `pathlib` - Path handling
- `pydantic` - Data validation and model definition