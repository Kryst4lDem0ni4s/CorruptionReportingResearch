"""
Corruption Reporting System - Visualization Package
Version: 1.0.0
Description: Publication-quality research figure generation

This package provides:
- ROC curve plotting
- Confusion matrix visualization
- Network graph rendering
- Score distribution plots
- Matplotlib style configuration

Usage:
    from evaluation.visualizations import (
        plot_roc_curve,
        plot_confusion_matrix,
        plot_coordination_graph,
        plot_score_histogram
    )
    
    # Generate publication figures
    plot_roc_curve(y_true, y_score, save_path='figures/roc.png')
    plot_confusion_matrix(y_true, y_pred, save_path='figures/cm.png')
"""

import logging
from pathlib import Path
from typing import Optional

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.visualizations')

# ============================================
# MATPLOTLIB CONFIGURATION
# ============================================

def configure_matplotlib_style(style_file: Optional[str] = None):
    """
    Configure matplotlib with custom style
    
    Args:
        style_file: Path to .mplstyle file (None = use package default)
    """
    try:
        import matplotlib.pyplot as plt
        
        if style_file is None:
            # Use package default style
            style_path = Path(__file__).parent / 'style.mplstyle'
            if style_path.exists():
                plt.style.use(str(style_path))
                logger.info(f"Applied matplotlib style: {style_path}")
            else:
                logger.warning(f"Style file not found: {style_path}")
                # Apply basic defaults
                plt.rcParams.update({
                    'figure.figsize': (10, 6),
                    'figure.dpi': 100,
                    'font.size': 11,
                    'axes.labelsize': 12,
                    'axes.titlesize': 14,
                    'xtick.labelsize': 10,
                    'ytick.labelsize': 10,
                    'legend.fontsize': 10,
                    'lines.linewidth': 2,
                    'grid.alpha': 0.3
                })
        else:
            plt.style.use(style_file)
            logger.info(f"Applied custom style: {style_file}")
    
    except ImportError:
        logger.warning("matplotlib not available, skipping style configuration")
    except Exception as e:
        logger.error(f"Failed to configure matplotlib style: {e}")

# ============================================
# IMPORT VISUALIZATION MODULES
# ============================================

# ROC Curve plotting
try:
    from .plot_roc import (
        plot_roc_curve,
        plot_multi_roc,
        plot_roc_with_ci,
        compute_roc_curve,
        compute_auc,
        find_optimal_threshold,
        save_roc_curves,
        HAS_MATPLOTLIB as HAS_MATPLOTLIB_ROC
    )
    _HAS_ROC = True
except ImportError as e:
    logger.warning(f"ROC plotting unavailable: {e}")
    _HAS_ROC = False
    HAS_MATPLOTLIB_ROC = False

# Confusion Matrix plotting
try:
    from .plot_confusion_matrix import (
        plot_confusion_matrix,
        plot_multi_confusion_matrix,
        plot_confusion_matrix_with_metrics,
        compute_confusion_matrix,
        normalize_confusion_matrix,
        save_confusion_matrices,
        HAS_MATPLOTLIB as HAS_MATPLOTLIB_CM
    )
    _HAS_CM = True
except ImportError as e:
    logger.warning(f"Confusion matrix plotting unavailable: {e}")
    _HAS_CM = False
    HAS_MATPLOTLIB_CM = False

# Network Graph plotting
try:
    from .plot_network import (
        plot_coordination_graph,
        plot_community_detection,
        plot_attack_pattern,
        compute_layout,
        save_network_graphs,
        HAS_MATPLOTLIB as HAS_MATPLOTLIB_NET,
        HAS_NETWORKX
    )
    _HAS_NETWORK = True
except ImportError as e:
    logger.warning(f"Network plotting unavailable: {e}")
    _HAS_NETWORK = False
    HAS_MATPLOTLIB_NET = False
    HAS_NETWORKX = False

# Distribution plotting
try:
    from .plot_distributions import (
        plot_score_histogram,
        plot_score_comparison,
        plot_before_after_comparison,
        plot_confidence_intervals,
        save_distributions,
        HAS_MATPLOTLIB as HAS_MATPLOTLIB_DIST,
        HAS_SCIPY
    )
    _HAS_DIST = True
except ImportError as e:
    logger.warning(f"Distribution plotting unavailable: {e}")
    _HAS_DIST = False
    HAS_MATPLOTLIB_DIST = False
    HAS_SCIPY = False

# ============================================
# CHECK DEPENDENCIES
# ============================================

HAS_MATPLOTLIB = (
    HAS_MATPLOTLIB_ROC or 
    HAS_MATPLOTLIB_CM or 
    HAS_MATPLOTLIB_NET or 
    HAS_MATPLOTLIB_DIST
)

# ============================================
# AUTO-CONFIGURE STYLE
# ============================================

# Automatically apply style when package is imported
if HAS_MATPLOTLIB:
    configure_matplotlib_style()

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def save_all_figures(
    data: dict,
    output_dir: str,
    prefix: str = ""
) -> dict:
    """
    Save all available figure types
    
    Args:
        data: Dictionary with evaluation data
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dictionary mapping figure type -> file paths
    """
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ROC curves
    if _HAS_ROC and 'roc_data' in data:
        try:
            files = save_roc_curves(
                data['roc_data'],
                str(output_path),
                prefix=prefix
            )
            results['roc'] = files
            logger.info(f"Saved {len(files)} ROC curve(s)")
        except Exception as e:
            logger.error(f"Failed to save ROC curves: {e}")
    
    # Confusion matrices
    if _HAS_CM and 'cm_data' in data:
        try:
            files = save_confusion_matrices(
                data['cm_data'],
                str(output_path),
                prefix=prefix
            )
            results['confusion_matrix'] = files
            logger.info(f"Saved {len(files)} confusion matrix/matrices")
        except Exception as e:
            logger.error(f"Failed to save confusion matrices: {e}")
    
    # Network graphs
    if _HAS_NETWORK and 'graphs' in data:
        try:
            files = save_network_graphs(
                data['graphs'],
                str(output_path),
                prefix=prefix
            )
            results['network'] = files
            logger.info(f"Saved {len(files)} network graph(s)")
        except Exception as e:
            logger.error(f"Failed to save network graphs: {e}")
    
    # Distributions
    if _HAS_DIST and 'scores' in data:
        try:
            files = save_distributions(
                data['scores'],
                str(output_path),
                prefix=prefix
            )
            results['distributions'] = files
            logger.info(f"Saved {len(files)} distribution plot(s)")
        except Exception as e:
            logger.error(f"Failed to save distributions: {e}")
    
    return results

def check_dependencies() -> dict:
    """
    Check which visualization modules are available
    
    Returns:
        Dictionary with availability status
    """
    return {
        'matplotlib': HAS_MATPLOTLIB,
        'roc_curves': _HAS_ROC,
        'confusion_matrices': _HAS_CM,
        'network_graphs': _HAS_NETWORK and HAS_NETWORKX,
        'distributions': _HAS_DIST,
        'scipy_kde': HAS_SCIPY
    }

def print_dependencies():
    """Print dependency status"""
    deps = check_dependencies()
    
    print("\n" + "="*50)
    print("VISUALIZATION MODULE DEPENDENCIES")
    print("="*50)
    
    for name, available in deps.items():
        status = " Available" if available else " Missing"
        print(f"{name:25s}: {status}")
    
    print("="*50 + "\n")

# ============================================
# PACKAGE METADATA
# ============================================

__version__ = '1.0.0'
__author__ = 'Corruption Reporting System Team'
__description__ = 'Publication-quality research figure generation'

# ============================================
# PACKAGE EXPORTS
# ============================================

# Build exports list dynamically
__all__ = [
    # Configuration
    'configure_matplotlib_style',
    'check_dependencies',
    'print_dependencies',
    'save_all_figures',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    
    # Availability flags
    'HAS_MATPLOTLIB',
    'HAS_NETWORKX',
    'HAS_SCIPY'
]

# Add ROC plotting if available
if _HAS_ROC:
    __all__.extend([
        'plot_roc_curve',
        'plot_multi_roc',
        'plot_roc_with_ci',
        'compute_roc_curve',
        'compute_auc',
        'find_optimal_threshold',
        'save_roc_curves'
    ])

# Add confusion matrix plotting if available
if _HAS_CM:
    __all__.extend([
        'plot_confusion_matrix',
        'plot_multi_confusion_matrix',
        'plot_confusion_matrix_with_metrics',
        'compute_confusion_matrix',
        'normalize_confusion_matrix',
        'save_confusion_matrices'
    ])

# Add network plotting if available
if _HAS_NETWORK:
    __all__.extend([
        'plot_coordination_graph',
        'plot_community_detection',
        'plot_attack_pattern',
        'compute_layout',
        'save_network_graphs'
    ])

# Add distribution plotting if available
if _HAS_DIST:
    __all__.extend([
        'plot_score_histogram',
        'plot_score_comparison',
        'plot_before_after_comparison',
        'plot_confidence_intervals',
        'save_distributions'
    ])

# ============================================
# MODULE INITIALIZATION
# ============================================

logger.info(f"Visualization package v{__version__} initialized")
logger.info(f"Available modules: ROC={_HAS_ROC}, CM={_HAS_CM}, Network={_HAS_NETWORK}, Dist={_HAS_DIST}")

"""# Import package (auto-applies style)
from evaluation.visualizations import *

# Check dependencies
print_dependencies()
# Output:
# ==================================================
# VISUALIZATION MODULE DEPENDENCIES
# ==================================================
# matplotlib               :  Available
# roc_curves              :  Available
# confusion_matrices      :  Available
# network_graphs          :  Available
# distributions           :  Available
# scipy_kde               :  Available
# ==================================================

# Use visualization functions
plot_roc_curve(y_true, y_score, save_path='roc.png')
plot_confusion_matrix(y_true, y_pred, save_path='cm.png')

# Batch save all figures
data = {
    'roc_data': {'Model1': {'y_true': y_true, 'y_score': scores}},
    'cm_data': {'Model1': {'y_true': y_true, 'y_pred': preds}},
    'graphs': {'coordination': graph},
    'scores': {'credibility': score_array}
}
results = save_all_figures(data, 'figures/', prefix='experiment1_')
# Returns: {'roc': [...], 'confusion_matrix': [...], 'network': [...], 'distributions': [...]}

# Custom style configuration
configure_matplotlib_style('custom_style.mplstyle')
"""