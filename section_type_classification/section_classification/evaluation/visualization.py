import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List, Optional, Tuple, Union


def create_enhanced_confusion_matrix_plot(
    confusion_matrix: pd.DataFrame,
    title: str = 'Confusion Matrix - Classification Results',
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues',
    normalize: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Create an enhanced confusion matrix plot with multiple visualization options.
    
    Args:
        confusion_matrix: Confusion matrix DataFrame
        title: Plot title
        figsize: Figure size (width, height)
        cmap: Color map for the heatmap
        normalize: Whether to normalize the confusion matrix
        show_percentages: Whether to show percentages in addition to counts
        save_path: Path to save the plot (optional)
    """
    # Remove 'All' row and column for plotting
    cm_plot = confusion_matrix.iloc[:-1, :-1]
    
    if normalize:
        cm_plot = cm_plot.div(cm_plot.sum(axis=1), axis=0) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt='.1f' if normalize else '.0f',  # Use .0f instead of 'd' to handle floats
        cmap=cmap,
        cbar_kws={'label': 'Percentage' if normalize else 'Count'},
        ax=ax
    )
    
    # Customize plot
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_metrics_summary_plot(
    metrics: Dict[str, Any],
    labels: List[str],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create a summary plot showing precision, recall, and F1-score for each class.
    
    Args:
        metrics: Metrics dictionary from calculate_additional_metrics
        labels: List of class labels
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    per_class = metrics['per_class']
    
    # Ensure labels and metrics arrays have the same length
    n_metrics = len(per_class['precision'])
    n_labels = len(labels)
    
    if n_metrics != n_labels:
        print(f"Warning: Number of labels ({n_labels}) doesn't match number of metrics ({n_metrics})")
        # Use the minimum length to avoid broadcasting errors
        min_length = min(n_metrics, n_labels)
        labels = labels[:min_length]
        per_class['precision'] = per_class['precision'][:min_length]
        per_class['recall'] = per_class['recall'][:min_length]
        per_class['f1'] = per_class['f1'][:min_length]
        per_class['support'] = per_class['support'][:min_length]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Per-class metrics bar plot
    x = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.8)
    ax1.bar(x, per_class['recall'], width, label='Recall', alpha=0.8)
    ax1.bar(x + width, per_class['f1'], width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Support distribution
    ax2.bar(labels, per_class['support'], alpha=0.7, color='skyblue')
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Support (Number of samples)')
    ax2.set_title('Class Distribution')
    ax2.tick_params(axis='x', rotation=45)  # Removed ha='right'
    # Set horizontal alignment for tick labels
    for label in ax2.get_xticklabels():
        label.set_ha('right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_probability_density_plot(
    distributions: Union[Dict[str, float], List[Dict[str, float]]],
    *,
    title: str = "Probability Distribution",
    figsize: Tuple[int, int] = (10, 6),
    plot_type: str = "bar",
    color_palette: str = "viridis",
    show_entropy: bool = True,
    show_confidence_intervals: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    Create density plots for probability distributions.
    
    Args:
        distributions: Single distribution dict or list of distribution dicts
        title: Plot title
        figsize: Figure size (width, height)
        plot_type: Type of plot ('bar', 'line', 'heatmap', 'violin')
        color_palette: Color palette for the plot
        show_entropy: Whether to display entropy information
        show_confidence_intervals: Whether to show confidence intervals (if available)
        save_path: Path to save the plot (optional)
        dpi: DPI for saved plots
    """
    # Convert single distribution to list for consistent processing
    if isinstance(distributions, dict):
        distributions = [distributions]
    
    if not distributions:
        raise ValueError("At least one distribution must be provided")
    
    # Get all unique categories
    all_categories = set()
    for dist in distributions:
        all_categories.update(dist.keys())
    categories = sorted(list(all_categories))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == "bar":
        _create_bar_plot(ax, distributions, categories, color_palette, show_entropy)
    elif plot_type == "line":
        _create_line_plot(ax, distributions, categories, color_palette, show_entropy)
    elif plot_type == "heatmap":
        _create_heatmap_plot(fig, distributions, categories, color_palette, show_entropy)
    elif plot_type == "violin":
        _create_violin_plot(ax, distributions, categories, color_palette, show_entropy)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    
    # Customize plot
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add legend if multiple distributions
    if len(distributions) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    plt.show()


def _create_bar_plot(ax, distributions, categories, color_palette, show_entropy):
    """Create a bar plot for probability distributions."""
    colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(distributions)))
    
    x = np.arange(len(categories))
    width = 0.8 / len(distributions) if len(distributions) > 1 else 0.8
    
    for i, (dist, color) in enumerate(zip(distributions, colors)):
        probabilities = [dist.get(cat, 0) for cat in categories]
        offset = (i - len(distributions) / 2 + 0.5) * width
        
        bars = ax.bar(x + offset, probabilities, width, 
                     label=f'Distribution {i+1}', 
                     color=color, alpha=0.7)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            if prob > 0.05:  # Only show labels for significant probabilities
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add entropy information if requested
        if show_entropy and len(distributions) == 1:
            try:
                from .probability_analysis import calculate_probability_estimates
                estimates = calculate_probability_estimates(dist)
                entropy = estimates['entropy']['value']
                ax.text(0.02, 0.98, f'Entropy: {entropy:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            except Exception:
                pass
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)


def _create_line_plot(ax, distributions, categories, color_palette, show_entropy):
    """Create a line plot for probability distributions."""
    colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(distributions)))
    x = np.arange(len(categories))
    
    for i, (dist, color) in enumerate(zip(distributions, colors)):
        probabilities = [dist.get(cat, 0) for cat in categories]
        ax.plot(x, probabilities, 'o-', 
               label=f'Distribution {i+1}', 
               color=color, linewidth=2, markersize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)


def _create_heatmap_plot(fig, distributions, categories, color_palette, show_entropy):
    """Create a heatmap for multiple probability distributions."""
    # Create data matrix
    data = []
    for dist in distributions:
        row = [dist.get(cat, 0) for cat in categories]
        data.append(row)
    
    data = np.array(data)
    
    # Create heatmap
    im = plt.imshow(data, cmap=color_palette, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Probability', rotation=270, labelpad=15)
    
    # Set ticks and labels
    plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
    plt.yticks(range(len(distributions)), [f'Dist {i+1}' for i in range(len(distributions))])
    
    # Add text annotations
    for i in range(len(distributions)):
        for j in range(len(categories)):
            text = plt.text(j, i, f'{data[i, j]:.2f}',
                           ha="center", va="center", color="white" if data[i, j] > 0.5 else "black")


def _create_violin_plot(ax, distributions, categories, color_palette, show_entropy):
    """Create a violin plot for probability distributions."""
    # For violin plots, we need to create synthetic data points
    # based on the probability distributions
    data_for_violin = []
    labels_for_violin = []
    
    for i, dist in enumerate(distributions):
        for cat in categories:
            prob = dist.get(cat, 0)
            if prob > 0:
                # Create synthetic data points based on probability
                n_points = int(prob * 1000)  # Scale for better visualization
                data_for_violin.extend([prob] * n_points)
                labels_for_violin.extend([cat] * n_points)
    
    if data_for_violin:
        # Create violin plot
        violin_parts = ax.violinplot([data_for_violin], positions=range(len(categories)))
        
        # Color the violin plot
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(categories)))
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.1)


def create_distribution_comparison_plot(
    distributions: List[Dict[str, float]],
    categories: Optional[List[str]] = None,
    *,
    plot_types: List[str] = ["bar", "entropy", "mode_consistency"],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive comparison plot for multiple probability distributions.
    
    Args:
        distributions: List of probability distribution dictionaries
        categories: Optional list of all possible categories
        plot_types: Types of plots to include ('bar', 'entropy', 'mode_consistency', 'heatmap')
        figsize: Figure size (width, height)
        save_path: Path to save the plot (optional)
    """
    if not distributions:
        raise ValueError("At least one distribution must be provided")
    
    # Get all unique categories if not provided
    if categories is None:
        all_categories = set()
        for dist in distributions:
            all_categories.update(dist.keys())
        categories = sorted(list(all_categories))
    
    # Calculate number of subplots needed
    n_plots = len(plot_types)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    for i, plot_type in enumerate(plot_types):
        ax = axes[i]
        
        if plot_type == "bar":
            _create_bar_plot(ax, distributions, categories, "viridis", False)
            ax.set_title("Probability Distributions")
            
        elif plot_type == "entropy":
            # Calculate entropies
            entropies = []
            for dist in distributions:
                try:
                    from .probability_analysis import calculate_probability_estimates
                    estimates = calculate_probability_estimates(dist)
                    entropies.append(estimates['entropy']['value'])
                except Exception:
                    entropies.append(0)
            
            ax.bar(range(len(distributions)), entropies, color='skyblue', alpha=0.7)
            ax.set_title("Entropy Comparison")
            ax.set_xlabel("Distribution Index")
            ax.set_ylabel("Entropy")
            ax.set_xticks(range(len(distributions)))
            ax.set_xticklabels([f'Dist {i+1}' for i in range(len(distributions))])
            
        elif plot_type == "mode_consistency":
            # Calculate modes
            modes = []
            mode_probs = []
            for dist in distributions:
                try:
                    from .probability_analysis import calculate_probability_estimates
                    estimates = calculate_probability_estimates(dist)
                    modes.append(estimates['mode']['category'])
                    mode_probs.append(estimates['mode']['probability'])
                except Exception:
                    modes.append("Unknown")
                    mode_probs.append(0)
            
            # Count mode frequencies
            mode_counts = pd.Series(modes).value_counts()
            
            ax.bar(range(len(mode_counts)), mode_counts.values, 
                   color='lightcoral', alpha=0.7)
            ax.set_title("Mode Consistency")
            ax.set_xlabel("Category")
            ax.set_ylabel("Frequency as Mode")
            ax.set_xticks(range(len(mode_counts)))
            ax.set_xticklabels(mode_counts.index, rotation=45, ha='right')
            
        elif plot_type == "heatmap":
            _create_heatmap_plot(fig, distributions, categories, "viridis", False)
            ax.set_title("Probability Heatmap")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show() 