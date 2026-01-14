"""
Corruption Reporting System - Score Distribution Visualization
Version: 1.0.0
Description: Generate publication-quality score distribution plots

This module provides:
- Score histograms
- Distribution comparisons
- KDE (Kernel Density Estimation) plots
- Box plots and violin plots
- Publication-ready formatting

Usage:
    from evaluation.visualizations.plot_distributions import (
        plot_score_histogram, plot_score_comparison
    )
    
    # Plot score distribution
    plot_score_histogram(scores, save_path='histogram.png')
    
    # Compare multiple distributions
    plot_score_comparison(scores_dict, save_path='comparison.png')
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

# Try scipy for KDE
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available for KDE")

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.visualizations.distributions')

# ============================================
# STYLE CONFIGURATION
# ============================================

# Color schemes
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

# ============================================
# SCORE HISTOGRAM
# ============================================

def plot_score_histogram(
    scores: np.ndarray,
    bins: Union[int, str] = 'auto',
    title: str = "Score Distribution",
    xlabel: str = "Score",
    save_path: Optional[str] = None,
    show_stats: bool = True,
    show_kde: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot score histogram with optional KDE
    
    Args:
        scores: Array of scores
        bins: Number of bins or binning strategy
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure
        show_stats: Show statistics
        show_kde: Show KDE overlay
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Convert to numpy array
    scores = np.array(scores)
    
    # Remove NaN and infinite values
    scores = scores[np.isfinite(scores)]
    
    if len(scores) == 0:
        logger.warning("No valid scores to plot")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(
        scores,
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS[0],
        edgecolor='black',
        linewidth=1.2,
        **kwargs
    )
    
    # Plot KDE
    if show_kde and HAS_SCIPY and len(scores) > 1:
        try:
            kde = scipy_stats.gaussian_kde(scores)
            x_range = np.linspace(scores.min(), scores.max(), 200)
            kde_values = kde(x_range)
            
            ax.plot(
                x_range, kde_values,
                color='red',
                linewidth=2.5,
                label='KDE',
                alpha=0.8
            )
            ax.legend(loc='upper right', frameon=True, shadow=True)
        except Exception as e:
            logger.warning(f"KDE computation failed: {e}")
    
    # Compute statistics
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)
    
    # Add vertical lines for mean and median
    ax.axvline(
        mean,
        color='darkblue',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {mean:.3f}',
        alpha=0.7
    )
    ax.axvline(
        median,
        color='darkgreen',
        linestyle=':',
        linewidth=2,
        label=f'Median: {median:.3f}',
        alpha=0.7
    )
    
    # Add statistics text box
    if show_stats:
        stats_text = (
            f'Mean: {mean:.3f}\n'
            f'Median: {median:.3f}\n'
            f'Std Dev: {std:.3f}\n'
            f'Min: {scores.min():.3f}\n'
            f'Max: {scores.max():.3f}\n'
            f'N: {len(scores)}'
        )
        
        ax.text(
            0.98, 0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Score histogram saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# MULTIPLE DISTRIBUTIONS COMPARISON
# ============================================

def plot_score_comparison(
    scores_dict: Dict[str, np.ndarray],
    title: str = "Score Distribution Comparison",
    xlabel: str = "Score",
    save_path: Optional[str] = None,
    plot_type: str = 'histogram',
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Compare multiple score distributions
    
    Args:
        scores_dict: Dictionary mapping labels -> scores
        title: Plot title
        xlabel: X-axis label
        save_path: Path to save figure
        plot_type: Type of plot ('histogram', 'kde', 'box', 'violin')
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional matplotlib arguments
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if plot_type == 'histogram':
        # Overlapping histograms
        for idx, (label, scores) in enumerate(scores_dict.items()):
            scores = np.array(scores)
            scores = scores[np.isfinite(scores)]
            
            if len(scores) == 0:
                continue
            
            ax.hist(
                scores,
                bins='auto',
                alpha=0.6,
                label=label,
                color=COLORS[idx % len(COLORS)],
                edgecolor='black',
                linewidth=1.0,
                density=True,
                **kwargs
            )
    
    elif plot_type == 'kde':
        if not HAS_SCIPY:
            logger.error("scipy required for KDE plots")
            return None
        
        # KDE plots
        for idx, (label, scores) in enumerate(scores_dict.items()):
            scores = np.array(scores)
            scores = scores[np.isfinite(scores)]
            
            if len(scores) < 2:
                continue
            
            try:
                kde = scipy_stats.gaussian_kde(scores)
                x_range = np.linspace(scores.min(), scores.max(), 200)
                kde_values = kde(x_range)
                
                ax.plot(
                    x_range, kde_values,
                    linewidth=2.5,
                    label=label,
                    color=COLORS[idx % len(COLORS)],
                    alpha=0.8
                )
                ax.fill_between(
                    x_range, kde_values,
                    alpha=0.2,
                    color=COLORS[idx % len(COLORS)]
                )
            except Exception as e:
                logger.warning(f"KDE for {label} failed: {e}")
        
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    
    elif plot_type == 'box':
        # Box plots
        data = []
        labels = []
        
        for label, scores in scores_dict.items():
            scores = np.array(scores)
            scores = scores[np.isfinite(scores)]
            if len(scores) > 0:
                data.append(scores)
                labels.append(label)
        
        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
            **kwargs
        )
        
        # Color boxes
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(xlabel, fontsize=12, fontweight='bold')
    
    elif plot_type == 'violin':
        # Violin plots
        data = []
        labels = []
        
        for label, scores in scores_dict.items():
            scores = np.array(scores)
            scores = scores[np.isfinite(scores)]
            if len(scores) > 0:
                data.append(scores)
                labels.append(label)
        
        parts = ax.violinplot(
            data,
            showmeans=True,
            showmedians=True,
            **kwargs
        )
        
        # Color violins
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS[idx % len(COLORS)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(xlabel, fontsize=12, fontweight='bold')
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    # Formatting
    ax.set_xlabel(xlabel if plot_type in ['histogram', 'kde'] else 'Distribution', 
                  fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if plot_type in ['histogram', 'kde']:
        ax.legend(loc='best', frameon=True, shadow=True)
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Score comparison saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# BEFORE/AFTER COMPARISON
# ============================================

def plot_before_after_comparison(
    scores_before: np.ndarray,
    scores_after: np.ndarray,
    title: str = "Before vs After Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot before/after score comparison
    
    Args:
        scores_before: Scores before intervention
        scores_after: Scores after intervention
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
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Before histogram
    scores_before = np.array(scores_before)
    scores_before = scores_before[np.isfinite(scores_before)]
    
    axes[0].hist(
        scores_before,
        bins='auto',
        alpha=0.7,
        color=COLORS[1],
        edgecolor='black',
        linewidth=1.2,
        density=True,
        **kwargs
    )
    
    mean_before = np.mean(scores_before)
    axes[0].axvline(mean_before, color='darkred', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_before:.3f}')
    axes[0].set_xlabel('Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0].set_title('Before', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # After histogram
    scores_after = np.array(scores_after)
    scores_after = scores_after[np.isfinite(scores_after)]
    
    axes[1].hist(
        scores_after,
        bins='auto',
        alpha=0.7,
        color=COLORS[4],
        edgecolor='black',
        linewidth=1.2,
        density=True,
        **kwargs
    )
    
    mean_after = np.mean(scores_after)
    axes[1].axvline(mean_after, color='darkgreen', linestyle='--', linewidth=2,
                    label=f'Mean: {mean_after:.3f}')
    axes[1].set_xlabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[1].set_title('After', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    
    # Add change statistics
    mean_change = mean_after - mean_before
    change_text = f"Mean Change: {mean_change:+.3f}"
    fig.text(0.5, 0.02, change_text, ha='center', fontsize=11, style='italic')
    
    # Tight layout
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Before/after comparison saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# CONFIDENCE INTERVAL PLOT
# ============================================

def plot_confidence_intervals(
    scores_dict: Dict[str, np.ndarray],
    confidence: float = 0.95,
    title: str = "Score Confidence Intervals",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot confidence intervals for multiple score distributions
    
    Args:
        scores_dict: Dictionary mapping labels -> scores
        confidence: Confidence level (0-1)
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Compute statistics
    labels = []
    means = []
    ci_lowers = []
    ci_uppers = []
    
    for label, scores in scores_dict.items():
        scores = np.array(scores)
        scores = scores[np.isfinite(scores)]
        
        if len(scores) == 0:
            continue
        
        mean = np.mean(scores)
        
        # Compute confidence interval
        alpha = 1 - confidence
        lower = np.percentile(scores, alpha / 2 * 100)
        upper = np.percentile(scores, (1 - alpha / 2) * 100)
        
        labels.append(label)
        means.append(mean)
        ci_lowers.append(lower)
        ci_uppers.append(upper)
    
    # Plot
    y_pos = np.arange(len(labels))
    
    # Error bars
    errors = [
        [means[i] - ci_lowers[i] for i in range(len(means))],
        [ci_uppers[i] - means[i] for i in range(len(means))]
    ]
    
    ax.errorbar(
        means, y_pos,
        xerr=errors,
        fmt='o',
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
        color=COLORS[0],
        ecolor=COLORS[1],
        **kwargs
    )
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Add confidence level text
    ax.text(
        0.02, 0.98,
        f'{confidence*100:.0f}% Confidence Interval',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Confidence intervals saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def save_distributions(
    scores_dict: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = ""
) -> List[str]:
    """
    Save distribution plots
    
    Args:
        scores_dict: Dictionary mapping labels -> scores
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Individual histograms
    for label, scores in scores_dict.items():
        filename = f"{prefix}{label}_distribution.png"
        filepath = output_path / filename
        
        plot_score_histogram(
            scores,
            title=f"Score Distribution - {label}",
            save_path=str(filepath)
        )
        
        saved_files.append(str(filepath))
    
    # Comparison plot
    if len(scores_dict) > 1:
        filename = f"{prefix}distribution_comparison.png"
        filepath = output_path / filename
        
        plot_score_comparison(
            scores_dict,
            title="Score Distribution Comparison",
            save_path=str(filepath),
            plot_type='kde'
        )
        
        saved_files.append(str(filepath))
    
    logger.info(f"Saved {len(saved_files)} distribution plot(s)")
    
    return saved_files

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'plot_score_histogram',
    'plot_score_comparison',
    'plot_before_after_comparison',
    'plot_confidence_intervals',
    'save_distributions',
    'HAS_MATPLOTLIB',
    'HAS_SCIPY'
]
