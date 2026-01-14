"""
Corruption Reporting System - Confusion Matrix Visualization
Version: 1.0.0
Description: Generate publication-quality confusion matrices

This module provides:
- Confusion matrix heatmaps
- Normalized and unnormalized versions
- Multi-class support
- Publication-ready formatting
- Customizable color schemes

Usage:
    from evaluation.visualizations.plot_confusion_matrix import (
        plot_confusion_matrix, plot_multi_confusion_matrix
    )
    
    # Single confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
    
    # Before/after comparison
    plot_multi_confusion_matrix([cm_before, cm_after], save_path='comparison.png')
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.visualizations.confusion_matrix')

# ============================================
# STYLE CONFIGURATION
# ============================================

# Color schemes
CMAP_DEFAULT = 'Blues'
CMAP_ERROR = 'Reds'
CMAP_DIVERGING = 'RdYlGn'

# ============================================
# CONFUSION MATRIX COMPUTATION
# ============================================

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label order (None = infer from data)
        
    Returns:
        Confusion matrix (rows=true, cols=pred)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique labels
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    # Initialize matrix
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Fill matrix
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_idx.get(true_label)
        pred_idx = label_to_idx.get(pred_label)
        if true_idx is not None and pred_idx is not None:
            cm[true_idx, pred_idx] += 1
    
    return cm

def normalize_confusion_matrix(
    cm: np.ndarray,
    mode: str = 'true'
) -> np.ndarray:
    """
    Normalize confusion matrix
    
    Args:
        cm: Confusion matrix
        mode: Normalization mode ('true', 'pred', 'all')
        
    Returns:
        Normalized confusion matrix
    """
    cm = cm.astype(float)
    
    if mode == 'true':
        # Normalize by true labels (rows)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm_normalized = cm / row_sums
    elif mode == 'pred':
        # Normalize by predicted labels (columns)
        col_sums = cm.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        cm_normalized = cm / col_sums
    elif mode == 'all':
        # Normalize by total
        total = cm.sum()
        cm_normalized = cm / total if total > 0 else cm
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return cm_normalized

# ============================================
# SINGLE CONFUSION MATRIX PLOTTING
# ============================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    cmap: str = CMAP_DEFAULT,
    show_values: bool = True,
    show_percentages: bool = False,
    **kwargs
) -> Optional[Figure]:
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names (None = infer)
        normalize: Normalize matrix
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: Figure DPI
        cmap: Colormap
        show_values: Show cell values
        show_percentages: Show percentages (if normalized)
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Compute confusion matrix
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        labels = [str(label) for label in labels]
    
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    
    # Normalize if requested
    cm_display = normalize_confusion_matrix(cm, mode='true') if normalize else cm
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, **kwargs)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(
        'Proportion' if normalize else 'Count',
        rotation=-90,
        va="bottom",
        fontsize=11,
        fontweight='bold'
    )
    
    # Set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add cell values
    if show_values:
        thresh = cm_display.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = cm_display[i, j]
                
                # Format text
                if normalize and show_percentages:
                    text = f"{value:.2%}\n({cm[i, j]})"
                elif normalize:
                    text = f"{value:.2f}\n({cm[i, j]})"
                else:
                    text = f"{int(value)}"
                
                # Determine text color
                color = "white" if value > thresh else "black"
                
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color=color,
                    fontsize=9
                )
    
    # Labels and title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# MULTIPLE CONFUSION MATRICES PLOTTING
# ============================================

def plot_multi_confusion_matrix(
    matrices: List[Dict[str, Any]],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    cmap: str = CMAP_DEFAULT,
    **kwargs
) -> Optional[Figure]:
    """
    Plot multiple confusion matrices side-by-side
    
    Args:
        matrices: List of dicts with 'y_true', 'y_pred', optionally 'labels'
        titles: List of subplot titles
        save_path: Path to save figure
        figsize: Figure size (None = auto)
        dpi: Figure DPI
        cmap: Colormap
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
        
    Example:
        matrices = [
            {'y_true': y_true, 'y_pred': pred_before, 'labels': ['Real', 'Fake']},
            {'y_true': y_true, 'y_pred': pred_after, 'labels': ['Real', 'Fake']}
        ]
        plot_multi_confusion_matrix(
            matrices,
            titles=['Before Counter-Evidence', 'After Counter-Evidence']
        )
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    n_matrices = len(matrices)
    
    # Default titles
    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(n_matrices)]
    
    # Auto figsize
    if figsize is None:
        figsize = (6 * n_matrices, 5)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize, dpi=dpi)
    
    # Handle single matrix case
    if n_matrices == 1:
        axes = [axes]
    
    # Plot each matrix
    for idx, (matrix_data, title, ax) in enumerate(zip(matrices, titles, axes)):
        y_true = matrix_data['y_true']
        y_pred = matrix_data['y_pred']
        labels = matrix_data.get('labels')
        normalize = matrix_data.get('normalize', False)
        
        # Compute confusion matrix
        if labels is None:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            labels = [str(label) for label in labels]
        
        cm = compute_confusion_matrix(y_true, y_pred, labels)
        cm_display = normalize_confusion_matrix(cm, mode='true') if normalize else cm
        
        # Plot heatmap
        im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap, **kwargs)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(
            'Proportion' if normalize else 'Count',
            rotation=-90,
            va="bottom",
            fontsize=10
        )
        
        # Set ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add cell values
        thresh = cm_display.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = cm_display[i, j]
                text = f"{value:.2f}" if normalize else f"{int(value)}"
                color = "white" if value > thresh else "black"
                
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color=color,
                    fontsize=9
                )
        
        # Labels and title
        if idx == 0:
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Multi confusion matrix saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# CONFUSION MATRIX WITH METRICS
# ============================================

def plot_confusion_matrix_with_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix with Metrics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot confusion matrix with additional metrics panel
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Compute confusion matrix
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        labels = [str(label) for label in labels]
    
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    cm_normalized = normalize_confusion_matrix(cm, mode='true')
    
    # Compute metrics
    total = cm.sum()
    accuracy = np.trace(cm) / total if total > 0 else 0
    
    # Per-class metrics
    precision = []
    recall = []
    f1_score = []
    
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot confusion matrix
    im = ax1.imshow(cm_normalized, interpolation='nearest', cmap=CMAP_DEFAULT)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom", fontsize=10)
    
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add values
    thresh = cm_normalized.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = cm_normalized[i, j]
            text = f"{value:.2f}\n({cm[i, j]})"
            color = "white" if value > thresh else "black"
            ax1.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot metrics table
    ax2.axis('tight')
    ax2.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['─' * 20, '─' * 10],
        ['Accuracy', f'{accuracy:.4f}'],
        ['', ''],
        ['Per-Class Metrics:', ''],
    ]
    
    for i, label in enumerate(labels):
        metrics_data.append([f'{label} Precision', f'{precision[i]:.4f}'])
        metrics_data.append([f'{label} Recall', f'{recall[i]:.4f}'])
        metrics_data.append([f'{label} F1-Score', f'{f1_score[i]:.4f}'])
        if i < len(labels) - 1:
            metrics_data.append(['', ''])
    
    table = ax2.table(
        cellText=metrics_data,
        cellLoc='left',
        loc='center',
        colWidths=[0.7, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax2.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Confusion matrix with metrics saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def save_confusion_matrices(
    results: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
    prefix: str = "",
    normalize: bool = True
) -> List[str]:
    """
    Save confusion matrices from evaluation results
    
    Args:
        results: Dictionary with 'y_true' and 'y_pred' for each model
        output_dir: Output directory
        prefix: Filename prefix
        normalize: Normalize matrices
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save individual matrices
    for model_name, data in results.items():
        if 'y_true' in data and 'y_pred' in data:
            filename = f"{prefix}{model_name}_cm.png"
            filepath = output_path / filename
            
            plot_confusion_matrix(
                data['y_true'],
                data['y_pred'],
                labels=data.get('labels'),
                normalize=normalize,
                title=f"Confusion Matrix - {model_name}",
                save_path=str(filepath)
            )
            
            saved_files.append(str(filepath))
    
    logger.info(f"Saved {len(saved_files)} confusion matrix/matrices")
    
    return saved_files

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'plot_confusion_matrix',
    'plot_multi_confusion_matrix',
    'plot_confusion_matrix_with_metrics',
    'compute_confusion_matrix',
    'normalize_confusion_matrix',
    'save_confusion_matrices',
    'HAS_MATPLOTLIB'
]
