"""
Corruption Reporting System - ROC Curve Visualization
Version: 1.0.0
Description: Generate publication-quality ROC curves and evaluation visualizations


This module provides:
- ROC curve plotting with multiple model comparison
- Precision-Recall curves
- Confusion matrices
- Score distributions
- Network graphs for coordination detection
- Consensus convergence plots
- Publication-ready formatting


Usage:
    from evaluation.visualizations.plot_roc import (
        plot_roc_curve, plot_multi_roc, generate_all_visualizations
    )
    
    # Single ROC curve
    plot_roc_curve(y_true, y_score, save_path='roc_curve.png')
    
    # Multiple models comparison
    plot_multi_roc(models_dict, save_path='roc_comparison.png')
    
    # Generate all visualizations from experiments
    generate_all_visualizations(experiment_results, output_dir)
"""


import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Import matplotlib and seaborn
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not available")


try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# Try to import sklearn for convenience
try:
    from sklearn.metrics import (
        roc_curve as sklearn_roc_curve,
        auc as sklearn_auc,
        precision_recall_curve,
        average_precision_score,
        confusion_matrix
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================
# LOGGING
# ============================================


logger = logging.getLogger('evaluation.visualizations.roc')


# ============================================
# STYLE CONFIGURATION
# ============================================


# Publication-ready style
PLOT_STYLE = {
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linestyle': '--'
}


# Color scheme for multiple curves
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']


# Apply global style
if HAS_MATPLOTLIB:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300
    })


# ============================================
# ROC CURVE COMPUTATION
# ============================================


def compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve points
    
    Args:
        y_true: Ground truth labels (0/1)
        y_score: Prediction scores (0-1)
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # Use sklearn if available for speed and accuracy
    if HAS_SKLEARN:
        fpr, tpr, thresholds = sklearn_roc_curve(y_true, y_score)
        return fpr, tpr, thresholds
    
    # Fallback: manual computation
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    # Get unique thresholds (sorted in descending order)
    thresholds = np.unique(y_score)[::-1]
    
    # Add boundary thresholds
    thresholds = np.concatenate(([thresholds[0] + 1], thresholds, [thresholds[-1] - 1]))
    
    # Compute TPR and FPR at each threshold
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        
        # Compute confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Compute TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return np.array(fpr_list), np.array(tpr_list), thresholds


def compute_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute area under curve using trapezoidal rule
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        
    Returns:
        AUC score
    """
    # Use sklearn if available
    if HAS_SKLEARN:
        return float(sklearn_auc(fpr, tpr))
    
    # Fallback: manual computation
    # Sort by fpr
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    
    # Compute area using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return float(auc)


def find_optimal_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[float, float, float]:
    """
    Find optimal threshold using Youden's J statistic
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
        
    Returns:
        Tuple of (optimal_threshold, optimal_fpr, optimal_tpr)
    """
    # Youden's J statistic = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    
    return (
        float(thresholds[optimal_idx]),
        float(fpr[optimal_idx]),
        float(tpr[optimal_idx])
    )


# ============================================
# SINGLE ROC CURVE PLOTTING
# ============================================


def plot_roc_curve(
    y_true: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    experiment_results: Optional[Dict[str, Any]] = None,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    show_optimal: bool = True,
    show_diagonal: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot ROC curve
    
    Args:
        y_true: Ground truth labels (optional if experiment_results provided)
        y_score: Prediction scores (optional if experiment_results provided)
        experiment_results: Experiment results dict with 'predictions' key
        title: Plot title
        save_path: Path to save figure (None = don't save)
        show_optimal: Show optimal threshold point
        show_diagonal: Show diagonal reference line
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Extract data from experiment_results if provided
    if experiment_results is not None:
        predictions = experiment_results.get('predictions', {})
        y_true = np.array(predictions.get('y_true', []))
        y_score = np.array(predictions.get('y_score', []))
    
    # Validate inputs
    if y_true is None or y_score is None:
        logger.error("Either provide y_true/y_score or experiment_results")
        return None
    
    if len(y_true) == 0 or len(y_score) == 0:
        logger.warning("No predictions found for ROC curve")
        return None
    
    # Apply style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Compute ROC curve
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_score)
    auc = compute_auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color=COLORS[0],
        linewidth=2.5,
        label=f'ROC Curve (AUC = {auc:.4f})',
        **kwargs
    )
    
    # Plot diagonal reference line
    if show_diagonal:
        ax.plot(
            [0, 1], [0, 1],
            color='navy',
            linestyle='--',
            linewidth=1.5,
            label='Random Classifier',
            alpha=0.5
        )
    
    # Mark optimal threshold
    if show_optimal:
        opt_threshold, opt_fpr, opt_tpr = find_optimal_threshold(fpr, tpr, thresholds)
        ax.plot(
            opt_fpr, opt_tpr,
            marker='o',
            markersize=8,
            color='red',
            label=f'Optimal (θ={opt_threshold:.3f})',
            zorder=10
        )
        
        # Add annotation
        ax.annotate(
            f'({opt_fpr:.3f}, {opt_tpr:.3f})',
            xy=(opt_fpr, opt_tpr),
            xytext=(opt_fpr + 0.1, opt_tpr - 0.1),
            fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red', lw=1)
        )
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"ROC curve saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig


# ============================================
# MULTIPLE ROC CURVES PLOTTING
# ============================================


def plot_multi_roc(
    models: Dict[str, Dict[str, np.ndarray]],
    title: str = "ROC Curve Comparison",
    save_path: Optional[str] = None,
    show_diagonal: bool = True,
    figsize: Tuple[int, int] = (10, 7),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot multiple ROC curves for model comparison
    
    Args:
        models: Dictionary mapping model_name -> {'y_true': ..., 'y_score': ...}
        title: Plot title
        save_path: Path to save figure
        show_diagonal: Show diagonal reference line
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
        
    Example:
        models = {
            'CLIP': {'y_true': y_true, 'y_score': clip_scores},
            'Ensemble': {'y_true': y_true, 'y_score': ensemble_scores}
        }
        plot_multi_roc(models, save_path='comparison.png')
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Apply style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot each model
    for idx, (model_name, data) in enumerate(models.items()):
        y_true = data['y_true']
        y_score = data['y_score']
        
        # Compute ROC curve
        fpr, tpr, _ = compute_roc_curve(y_true, y_score)
        auc = compute_auc(fpr, tpr)
        
        # Plot
        color = COLORS[idx % len(COLORS)]
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2.5,
            label=f'{model_name} (AUC = {auc:.4f})',
            **kwargs
        )
    
    # Plot diagonal reference line
    if show_diagonal:
        ax.plot(
            [0, 1], [0, 1],
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Random Classifier',
            alpha=0.5
        )
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Multi-ROC curve saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig


# ============================================
# ROC CURVE WITH CONFIDENCE BANDS
# ============================================


def plot_roc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve with Confidence Interval",
    save_path: Optional[str] = None,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot ROC curve with bootstrap confidence intervals
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        title: Plot title
        save_path: Path to save figure
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0-1)
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Apply style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Compute main ROC curve
    fpr_main, tpr_main, _ = compute_roc_curve(y_true, y_score)
    auc_main = compute_auc(fpr_main, tpr_main)
    
    # Bootstrap for confidence intervals
    n_samples = len(y_true)
    tpr_bootstrap = []
    auc_bootstrap = []
    
    np.random.seed(42)  # Reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        
        # Compute ROC
        fpr_boot, tpr_boot, _ = compute_roc_curve(y_true_boot, y_score_boot)
        auc_boot = compute_auc(fpr_boot, tpr_boot)
        
        # Interpolate to common FPR values
        tpr_interp = np.interp(fpr_main, fpr_boot, tpr_boot)
        
        tpr_bootstrap.append(tpr_interp)
        auc_bootstrap.append(auc_boot)
    
    # Compute confidence intervals
    tpr_bootstrap = np.array(tpr_bootstrap)
    alpha = 1 - confidence
    tpr_lower = np.percentile(tpr_bootstrap, alpha / 2 * 100, axis=0)
    tpr_upper = np.percentile(tpr_bootstrap, (1 - alpha / 2) * 100, axis=0)
    
    auc_lower = np.percentile(auc_bootstrap, alpha / 2 * 100)
    auc_upper = np.percentile(auc_bootstrap, (1 - alpha / 2) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot confidence band
    ax.fill_between(
        fpr_main, tpr_lower, tpr_upper,
        color=COLORS[0],
        alpha=0.2,
        label=f'{confidence*100:.0f}% CI'
    )
    
    # Plot main ROC curve
    ax.plot(
        fpr_main, tpr_main,
        color=COLORS[0],
        linewidth=2.5,
        label=f'ROC (AUC = {auc_main:.3f} [{auc_lower:.3f}-{auc_upper:.3f}])',
        **kwargs
    )
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'gray', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"ROC curve with CI saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig


# ============================================
# PRECISION-RECALL CURVE
# ============================================


def plot_precision_recall_curve(
    experiment_results: Dict[str, Any],
    output_path: Path,
    title: str = "Precision-Recall Curve"
) -> None:
    """Generate precision-recall curve from experiment results"""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return
    
    try:
        predictions = experiment_results.get('predictions', {})
        y_true = np.array(predictions.get('y_true', []))
        y_score = np.array(predictions.get('y_score', []))
        
        if len(y_true) == 0 or len(y_score) == 0:
            logger.warning("No predictions found for PR curve")
            return
        
        # Compute PR curve
        if HAS_SKLEARN:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)
        else:
            logger.error("sklearn required for PR curve computation")
            return
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate PR curve: {e}")


# ============================================
# CONFUSION MATRIX
# ============================================


def plot_confusion_matrix(
    experiment_results: Dict[str, Any],
    output_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """Generate confusion matrix heatmap from experiment results"""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return
    
    try:
        predictions = experiment_results.get('predictions', {})
        y_true = np.array(predictions.get('y_true', []))
        y_pred = np.array(predictions.get('y_pred', []))
        
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.warning("No predictions found for confusion matrix")
            return
        
        # Compute confusion matrix
        if HAS_SKLEARN:
            cm = confusion_matrix(y_true, y_pred)
        else:
            logger.error("sklearn required for confusion matrix")
            return
        
        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'],
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix: {e}")


# ============================================
# SCORE DISTRIBUTION
# ============================================


def plot_score_distribution(
    experiment_results: Dict[str, Any],
    output_path: Path,
    title: str = "Score Distribution"
) -> None:
    """Generate score distribution histogram from experiment results"""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return
    
    try:
        predictions = experiment_results.get('predictions', {})
        y_score = np.array(predictions.get('y_score', []))
        y_true = np.array(predictions.get('y_true', []))
        
        if len(y_score) == 0:
            logger.warning("No scores found for distribution plot")
            return
        
        # Separate by class
        real_scores = y_score[y_true == 0]
        fake_scores = y_score[y_true == 1]
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.hist(real_scores, bins=20, alpha=0.6, label='Real', color='green', edgecolor='black')
        plt.hist(fake_scores, bins=20, alpha=0.6, label='Fake', color='red', edgecolor='black')
        plt.xlabel('Credibility Score')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Score distribution saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate score distribution: {e}")


# ============================================
# NETWORK GRAPH
# ============================================


def plot_network_graph(
    experiment_results: Dict[str, Any],
    output_path: Path,
    title: str = "Coordination Network Graph"
) -> None:
    """Generate coordination detection network graph from experiment results"""
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        logger.error("matplotlib and networkx required for network graph")
        return
    
    try:
        scenarios = experiment_results.get('scenarios', [])
        
        if not scenarios:
            logger.warning("No scenarios found for network graph")
            return
        
        # Create graph from first coordinated scenario
        coordinated_scenario = next(
            (s for s in scenarios if s.get('is_coordinated')),
            None
        )
        
        if not coordinated_scenario:
            logger.warning("No coordinated scenario found")
            return
        
        # Build graph
        G = nx.Graph()
        submissions = coordinated_scenario['submissions']
        
        for i, sub1 in enumerate(submissions):
            for j, sub2 in enumerate(submissions[i+1:], i+1):
                # Add edge with similarity weight
                similarity = 0.8  # Simplified for visualization
                G.add_edge(sub1['pseudonym'], sub2['pseudonym'], weight=similarity)
        
        # Plot
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Network graph saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate network graph: {e}")


# ============================================
# CONVERGENCE PLOT
# ============================================


def plot_convergence(
    experiment_results: Dict[str, Any],
    output_path: Path,
    title: str = "Consensus Convergence Analysis"
) -> None:
    """Generate consensus convergence plot from experiment results"""
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib required for plotting")
        return
    
    try:
        convergence_times = experiment_results.get('convergence_times', [])
        agreement_rates = experiment_results.get('agreement_rates', [])
        
        if not convergence_times or not agreement_rates:
            logger.warning("No convergence data found")
            return
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Convergence times
        ax1.hist(convergence_times, bins=15, color='blue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(convergence_times), color='red', linestyle='--',
                   label=f'Mean: {np.mean(convergence_times):.2f}s')
        ax1.set_xlabel('Convergence Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Convergence Time Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        
        # Agreement rates
        ax2.hist(agreement_rates, bins=15, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(agreement_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(agreement_rates):.2%}')
        ax2.set_xlabel('Agreement Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Validator Agreement Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate convergence plot: {e}")


# ============================================
# GENERATE ALL VISUALIZATIONS
# ============================================


def generate_all_visualizations(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> List[Path]:
    """
    Generate all visualizations from experiment results
    
    Args:
        results_dict: Dictionary mapping experiment_name -> results
        output_dir: Output directory for figures
        
    Returns:
        List of generated file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_figures = []
    
    # ROC curves (for deepfake detection experiments)
    for exp_name, results in results_dict.items():
        if 'deepfake' in exp_name.lower():
            # ROC curve
            output_path = output_dir / f'roc_curve_{exp_name}.png'
            plot_roc_curve(experiment_results=results, save_path=str(output_path))
            generated_figures.append(output_path)
            
            # PR curve
            output_path = output_dir / f'pr_curve_{exp_name}.png'
            plot_precision_recall_curve(results, output_path)
            generated_figures.append(output_path)
            
            # Confusion matrix
            output_path = output_dir / f'confusion_matrix_{exp_name}.png'
            plot_confusion_matrix(results, output_path)
            generated_figures.append(output_path)
            
            # Score distribution
            output_path = output_dir / f'score_dist_{exp_name}.png'
            plot_score_distribution(results, output_path)
            generated_figures.append(output_path)
    
    # Network graph (for coordination detection)
    for exp_name, results in results_dict.items():
        if 'coordination' in exp_name.lower():
            output_path = output_dir / f'network_graph_{exp_name}.png'
            plot_network_graph(results, output_path)
            generated_figures.append(output_path)
    
    # Convergence plots (for consensus simulation)
    for exp_name, results in results_dict.items():
        if 'consensus' in exp_name.lower():
            output_path = output_dir / f'convergence_{exp_name}.png'
            plot_convergence(results, output_path)
            generated_figures.append(output_path)
    
    logger.info(f"Generated {len(generated_figures)} visualizations in {output_dir}")
    
    return generated_figures


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================


def save_roc_curves(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = ""
) -> List[str]:
    """
    Save multiple ROC curves from evaluation results
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save individual ROC curves
    for model_name, data in results.items():
        if 'y_true' in data and 'y_score' in data:
            filename = f"{prefix}{model_name}_roc.png"
            filepath = output_path / filename
            
            plot_roc_curve(
                data['y_true'],
                data['y_score'],
                title=f"ROC Curve - {model_name}",
                save_path=str(filepath)
            )
            
            saved_files.append(str(filepath))
    
    # Save comparison plot if multiple models
    if len(results) > 1:
        filename = f"{prefix}roc_comparison.png"
        filepath = output_path / filename
        
        plot_multi_roc(
            results,
            title="Model Comparison - ROC Curves",
            save_path=str(filepath)
        )
        
        saved_files.append(str(filepath))
    
    logger.info(f"Saved {len(saved_files)} ROC curve(s)")
    
    return saved_files


# ============================================
# PACKAGE EXPORTS
# ============================================


__all__ = [
    'plot_roc_curve',
    'plot_multi_roc',
    'plot_roc_with_ci',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_score_distribution',
    'plot_network_graph',
    'plot_convergence',
    'generate_all_visualizations',
    'compute_roc_curve',
    'compute_auc',
    'find_optimal_threshold',
    'save_roc_curves',
    'HAS_MATPLOTLIB',
    'HAS_SKLEARN',
    'HAS_NETWORKX'
]
