"""
Corruption Reporting System - Network Graph Visualization
Version: 1.0.0
Description: Generate publication-quality coordination network graphs

This module provides:
- Coordination network visualization
- Community detection rendering
- Node attribute coloring
- Edge weight visualization
- Publication-ready formatting

Usage:
    from evaluation.visualizations.plot_network import (
        plot_coordination_graph, plot_community_detection
    )
    
    # Plot coordination network
    plot_coordination_graph(graph, save_path='network.png')
    
    # Plot with community detection
    plot_community_detection(graph, communities, save_path='communities.png')
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
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
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available")

# Import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not available")

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.visualizations.network')

# ============================================
# STYLE CONFIGURATION
# ============================================

# Color schemes
NODE_COLORS = {
    'normal': '#4A90E2',      # Blue
    'suspicious': '#F5A623',  # Orange
    'coordinated': '#D0021B',  # Red
    'default': '#7ED321'       # Green
}

EDGE_COLORS = {
    'weak': '#CCCCCC',
    'medium': '#888888',
    'strong': '#333333'
}

COMMUNITY_COLORS = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', 
    '#6A994E', '#BC4B51', '#8E7DBE', '#F77F00'
]

# ============================================
# GRAPH LAYOUT FUNCTIONS
# ============================================

def compute_layout(
    graph: 'nx.Graph',
    layout_type: str = 'spring',
    **kwargs
) -> Dict[Any, Tuple[float, float]]:
    """
    Compute graph layout positions
    
    Args:
        graph: NetworkX graph
        layout_type: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        **kwargs: Layout-specific parameters
        
    Returns:
        Dictionary mapping nodes to (x, y) positions
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX required for graph layout")
    
    layout_functions = {
        'spring': nx.spring_layout,
        'circular': nx.circular_layout,
        'kamada_kawai': nx.kamada_kawai_layout,
        'spectral': nx.spectral_layout,
        'shell': nx.shell_layout
    }
    
    if layout_type not in layout_functions:
        logger.warning(f"Unknown layout type '{layout_type}', using 'spring'")
        layout_type = 'spring'
    
    layout_func = layout_functions[layout_type]
    
    # Set default parameters
    if layout_type == 'spring':
        kwargs.setdefault('k', 1.0)
        kwargs.setdefault('iterations', 50)
        kwargs.setdefault('seed', 42)
    
    try:
        pos = layout_func(graph, **kwargs)
    except Exception as e:
        logger.error(f"Layout computation failed: {e}")
        # Fallback to spring layout
        pos = nx.spring_layout(graph, seed=42)
    
    return pos

# ============================================
# COORDINATION GRAPH PLOTTING
# ============================================

def plot_coordination_graph(
    graph: 'nx.Graph',
    node_labels: Optional[Dict[Any, str]] = None,
    node_colors: Optional[Dict[Any, str]] = None,
    node_sizes: Optional[Dict[Any, float]] = None,
    edge_weights: Optional[Dict[Tuple, float]] = None,
    title: str = "Coordination Network",
    save_path: Optional[str] = None,
    layout: str = 'spring',
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    show_labels: bool = True,
    **kwargs
) -> Optional[Figure]:
    """
    Plot coordination network graph
    
    Args:
        graph: NetworkX graph
        node_labels: Custom node labels
        node_colors: Node color mapping
        node_sizes: Node size mapping
        edge_weights: Edge weight mapping
        title: Plot title
        save_path: Path to save figure
        layout: Layout algorithm
        figsize: Figure size
        dpi: Figure DPI
        show_labels: Show node labels
        **kwargs: Additional parameters
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        raise ImportError("matplotlib and networkx required for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Compute layout
    pos = compute_layout(graph, layout_type=layout)
    
    # Prepare node attributes
    nodes = list(graph.nodes())
    
    # Node colors
    if node_colors is None:
        node_color_list = [NODE_COLORS['default']] * len(nodes)
    else:
        node_color_list = [
            node_colors.get(node, NODE_COLORS['default']) 
            for node in nodes
        ]
    
    # Node sizes
    if node_sizes is None:
        node_size_list = [300] * len(nodes)
    else:
        node_size_list = [
            node_sizes.get(node, 300) 
            for node in nodes
        ]
    
    # Draw edges
    edges = list(graph.edges())
    
    if edge_weights:
        # Categorize edge weights
        edge_colors_list = []
        edge_widths = []
        
        for edge in edges:
            weight = edge_weights.get(edge, edge_weights.get((edge[1], edge[0]), 0.5))
            
            if weight < 0.3:
                edge_colors_list.append(EDGE_COLORS['weak'])
                edge_widths.append(1.0)
            elif weight < 0.7:
                edge_colors_list.append(EDGE_COLORS['medium'])
                edge_widths.append(2.0)
            else:
                edge_colors_list.append(EDGE_COLORS['strong'])
                edge_widths.append(3.0)
        
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=edges,
            edge_color=edge_colors_list,
            width=edge_widths,
            alpha=0.6,
            ax=ax
        )
    else:
        nx.draw_networkx_edges(
            graph, pos,
            edge_color=EDGE_COLORS['medium'],
            width=1.5,
            alpha=0.5,
            ax=ax
        )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_color_list,
        node_size=node_size_list,
        alpha=0.9,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # Draw labels
    if show_labels:
        if node_labels is None:
            node_labels = {node: str(node) for node in nodes}
        
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
    
    # Legend
    legend_elements = [
        Patch(facecolor=NODE_COLORS['normal'], edgecolor='black', label='Normal'),
        Patch(facecolor=NODE_COLORS['suspicious'], edgecolor='black', label='Suspicious'),
        Patch(facecolor=NODE_COLORS['coordinated'], edgecolor='black', label='Coordinated')
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        frameon=True,
        shadow=True,
        fontsize=10
    )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add graph statistics
    stats_text = f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}"
    if graph.number_of_nodes() > 0:
        density = nx.density(graph)
        stats_text += f" | Density: {density:.3f}"
    
    ax.text(
        0.5, -0.05,
        stats_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic'
    )
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Coordination graph saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# COMMUNITY DETECTION PLOTTING
# ============================================

def plot_community_detection(
    graph: 'nx.Graph',
    communities: List[Set[Any]],
    title: str = "Community Detection",
    save_path: Optional[str] = None,
    layout: str = 'spring',
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    show_labels: bool = True,
    **kwargs
) -> Optional[Figure]:
    """
    Plot graph with community detection
    
    Args:
        graph: NetworkX graph
        communities: List of node sets (communities)
        title: Plot title
        save_path: Path to save figure
        layout: Layout algorithm
        figsize: Figure size
        dpi: Figure DPI
        show_labels: Show node labels
        **kwargs: Additional parameters
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        raise ImportError("matplotlib and networkx required for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Compute layout
    pos = compute_layout(graph, layout_type=layout)
    
    # Assign community colors
    node_to_community = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = comm_id
    
    # Prepare node colors
    nodes = list(graph.nodes())
    node_color_list = [
        COMMUNITY_COLORS[node_to_community.get(node, 0) % len(COMMUNITY_COLORS)]
        for node in nodes
    ]
    
    # Draw edges
    nx.draw_networkx_edges(
        graph, pos,
        edge_color='gray',
        width=1.5,
        alpha=0.3,
        ax=ax
    )
    
    # Draw nodes by community
    for comm_id, community in enumerate(communities):
        community_nodes = list(community)
        color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
        
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=community_nodes,
            node_color=color,
            node_size=400,
            alpha=0.9,
            edgecolors='black',
            linewidths=2.0,
            label=f'Community {comm_id + 1} ({len(community)} nodes)',
            ax=ax
        )
    
    # Draw labels
    if show_labels:
        node_labels = {node: str(node) for node in nodes}
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
    
    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        shadow=True,
        fontsize=9
    )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add statistics
    modularity = nx.algorithms.community.modularity(graph, communities) if communities else 0
    stats_text = f"Communities: {len(communities)} | Modularity: {modularity:.3f}"
    
    ax.text(
        0.5, -0.05,
        stats_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic'
    )
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Community detection graph saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# ATTACK PATTERN VISUALIZATION
# ============================================

def plot_attack_pattern(
    graph: 'nx.Graph',
    attack_nodes: Set[Any],
    normal_nodes: Set[Any],
    title: str = "Coordination Attack Pattern",
    save_path: Optional[str] = None,
    layout: str = 'spring',
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 300,
    **kwargs
) -> Optional[Figure]:
    """
    Plot graph highlighting attack patterns
    
    Args:
        graph: NetworkX graph
        attack_nodes: Set of nodes involved in attack
        normal_nodes: Set of normal nodes
        title: Plot title
        save_path: Path to save figure
        layout: Layout algorithm
        figsize: Figure size
        dpi: Figure DPI
        **kwargs: Additional parameters
        
    Returns:
        Figure object (if save_path is None)
    """
    if not HAS_MATPLOTLIB or not HAS_NETWORKX:
        raise ImportError("matplotlib and networkx required for plotting")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Compute layout
    pos = compute_layout(graph, layout_type=layout)
    
    # Draw edges
    # Highlight edges between attack nodes
    attack_edges = [
        (u, v) for u, v in graph.edges()
        if u in attack_nodes and v in attack_nodes
    ]
    normal_edges = [
        (u, v) for u, v in graph.edges()
        if (u, v) not in attack_edges
    ]
    
    # Draw normal edges
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=normal_edges,
        edge_color='gray',
        width=1.0,
        alpha=0.3,
        ax=ax
    )
    
    # Draw attack edges
    if attack_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=attack_edges,
            edge_color='red',
            width=3.0,
            alpha=0.8,
            ax=ax
        )
    
    # Draw normal nodes
    if normal_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(normal_nodes),
            node_color=NODE_COLORS['normal'],
            node_size=300,
            alpha=0.7,
            edgecolors='black',
            linewidths=1.0,
            label='Normal',
            ax=ax
        )
    
    # Draw attack nodes
    if attack_nodes:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=list(attack_nodes),
            node_color=NODE_COLORS['coordinated'],
            node_size=500,
            alpha=0.9,
            edgecolors='black',
            linewidths=2.0,
            label='Coordinated Attack',
            ax=ax
        )
    
    # Draw labels
    labels = {node: str(node) for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Legend
    ax.legend(
        loc='upper right',
        frameon=True,
        shadow=True,
        fontsize=10
    )
    
    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Add statistics
    attack_size = len(attack_nodes)
    attack_density = nx.density(graph.subgraph(attack_nodes)) if attack_size > 1 else 0
    stats_text = f"Attack Nodes: {attack_size} | Attack Density: {attack_density:.3f}"
    
    ax.text(
        0.5, -0.05,
        stats_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic',
        color='red'
    )
    
    # Tight layout
    fig.tight_layout()
    
    # Save or return
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Attack pattern graph saved to: {save_path}")
        plt.close(fig)
        return None
    else:
        return fig

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def save_network_graphs(
    graphs: Dict[str, 'nx.Graph'],
    output_dir: str,
    prefix: str = "",
    layout: str = 'spring'
) -> List[str]:
    """
    Save multiple network graphs
    
    Args:
        graphs: Dictionary mapping name -> graph
        output_dir: Output directory
        prefix: Filename prefix
        layout: Layout algorithm
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for name, graph in graphs.items():
        filename = f"{prefix}{name}_network.png"
        filepath = output_path / filename
        
        plot_coordination_graph(
            graph,
            title=f"Network - {name}",
            save_path=str(filepath),
            layout=layout
        )
        
        saved_files.append(str(filepath))
    
    logger.info(f"Saved {len(saved_files)} network graph(s)")
    
    return saved_files

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'plot_coordination_graph',
    'plot_community_detection',
    'plot_attack_pattern',
    'compute_layout',
    'save_network_graphs',
    'HAS_MATPLOTLIB',
    'HAS_NETWORKX'
]
