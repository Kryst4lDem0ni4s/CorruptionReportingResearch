"""
Graph Utils - NetworkX graph operations for coordination detection

Provides:
- Graph construction from submissions
- Community detection (Louvain)
- Graph similarity measures
- Centrality calculations
- Subgraph extraction
- Graph visualization utilities
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)


class GraphUtils:
    """
    Graph utilities for coordination detection.
    
    Features:
    - Graph construction from submission data
    - Community detection (Louvain algorithm)
    - Similarity-based edge weighting
    - Centrality measures
    - Subgraph analysis
    - Graph metrics
    """
    
    @staticmethod
    def create_submission_graph(
        submissions: List[Dict],
        similarity_threshold: float = 0.3
    ) -> nx.Graph:
        """
        Create graph from submissions.
        
        Nodes represent submissions, edges represent similarity.
        
        Args:
            submissions: List of submission dictionaries
            similarity_threshold: Minimum similarity for edge creation
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Add nodes
        for submission in submissions:
            submission_id = submission.get('submission_id', submission.get('id'))
            
            G.add_node(
                submission_id,
                timestamp=submission.get('timestamp'),
                pseudonym=submission.get('pseudonym'),
                evidence_type=submission.get('evidence_type'),
                submission_data=submission
            )
        
        # Add edges based on similarity
        submission_ids = [s.get('submission_id', s.get('id')) for s in submissions]
        
        for i, sub1 in enumerate(submissions):
            for sub2 in submissions[i+1:]:
                id1 = sub1.get('submission_id', sub1.get('id'))
                id2 = sub2.get('submission_id', sub2.get('id'))
                
                # Calculate similarity (placeholder - use actual similarity function)
                similarity = GraphUtils._calculate_submission_similarity(sub1, sub2)
                
                if similarity >= similarity_threshold:
                    G.add_edge(id1, id2, weight=similarity, similarity=similarity)
        
        logger.info(
            f"Graph created: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    @staticmethod
    def _calculate_submission_similarity(sub1: Dict, sub2: Dict) -> float:
        """
        Calculate similarity between two submissions.
        
        Combines temporal, stylometric, and content similarity.
        
        Args:
            sub1: First submission
            sub2: Second submission
            
        Returns:
            float: Similarity score (0-1)
        """
        similarities = []
        
        # Temporal similarity (if timestamps available)
        ts1 = sub1.get('timestamp', 0)
        ts2 = sub2.get('timestamp', 0)
        
        if ts1 and ts2:
            time_diff = abs(ts1 - ts2)
            # Decay function: similar if within 1 hour
            temporal_sim = np.exp(-time_diff / 3600)
            similarities.append(temporal_sim)
        
        # Pseudonym similarity (same pseudonym = suspicious)
        if sub1.get('pseudonym') == sub2.get('pseudonym'):
            similarities.append(0.8)
        
        # Style similarity (if stylometric features available)
        style1 = sub1.get('stylometric_features', {})
        style2 = sub2.get('stylometric_features', {})
        
        if style1 and style2:
            # Calculate feature similarity
            style_sim = GraphUtils._calculate_feature_similarity(style1, style2)
            similarities.append(style_sim)
        
        # Content similarity (if text available)
        text1 = sub1.get('narrative', '')
        text2 = sub2.get('narrative', '')
        
        if text1 and text2:
            # Use simple token overlap
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if tokens1 and tokens2:
                content_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                similarities.append(content_sim)
        
        # Average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        
        return 0.0
    
    @staticmethod
    def _calculate_feature_similarity(features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature dictionaries."""
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        differences = []
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            # Only compare numerical features
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize difference
                max_val = max(abs(val1), abs(val2), 1e-6)
                diff = abs(val1 - val2) / max_val
                differences.append(1 - diff)  # Convert to similarity
        
        if differences:
            return sum(differences) / len(differences)
        
        return 0.0
    
    @staticmethod
    def detect_communities(
        G: nx.Graph,
        resolution: float = 1.0
    ) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter (higher = more communities)
            
        Returns:
            dict: node_id → community_id
        """
        try:
            # Use Louvain community detection
            import community as community_louvain
            
            communities = community_louvain.best_partition(G, resolution=resolution)
            
            num_communities = len(set(communities.values()))
            logger.info(f"Detected {num_communities} communities")
            
            return communities
            
        except ImportError:
            # Fallback to greedy modularity if python-louvain not available
            logger.warning("python-louvain not available, using greedy modularity")
            
            communities_generator = nx.community.greedy_modularity_communities(G)
            communities_list = list(communities_generator)
            
            # Convert to node → community_id format
            communities = {}
            for i, community in enumerate(communities_list):
                for node in community:
                    communities[node] = i
            
            logger.info(f"Detected {len(communities_list)} communities (greedy)")
            
            return communities
    
    @staticmethod
    def calculate_centrality(G: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        Calculate multiple centrality measures.
        
        Args:
            G: NetworkX graph
            
        Returns:
            dict: Centrality measures for each node
        """
        centrality = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(G)
        
        # Betweenness centrality
        betweenness_cent = nx.betweenness_centrality(G)
        
        # Closeness centrality
        closeness_cent = nx.closeness_centrality(G)
        
        # Eigenvector centrality (if connected)
        try:
            eigenvector_cent = nx.eigenvector_centrality(G, max_iter=100)
        except:
            eigenvector_cent = {node: 0.0 for node in G.nodes()}
        
        # Combine into single dict
        for node in G.nodes():
            centrality[node] = {
                'degree': degree_cent.get(node, 0.0),
                'betweenness': betweenness_cent.get(node, 0.0),
                'closeness': closeness_cent.get(node, 0.0),
                'eigenvector': eigenvector_cent.get(node, 0.0)
            }
        
        return centrality
    
    @staticmethod
    def find_suspicious_patterns(
        G: nx.Graph,
        communities: Dict[str, int]
    ) -> List[Dict]:
        """
        Find suspicious coordination patterns.
        
        Args:
            G: NetworkX graph
            communities: Community assignments
            
        Returns:
            List of suspicious patterns
        """
        patterns = []
        
        # Pattern 1: Dense communities (potential coordinated attacks)
        community_sizes = {}
        for node, comm_id in communities.items():
            community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1
        
        for comm_id, size in community_sizes.items():
            if size >= 3:  # At least 3 submissions
                # Get nodes in community
                comm_nodes = [n for n, c in communities.items() if c == comm_id]
                
                # Calculate subgraph density
                subgraph = G.subgraph(comm_nodes)
                density = nx.density(subgraph)
                
                if density > 0.5:  # High density = suspicious
                    patterns.append({
                        'type': 'dense_community',
                        'community_id': comm_id,
                        'nodes': comm_nodes,
                        'size': size,
                        'density': density,
                        'suspicion_score': density * (size / 10)  # Scale by size
                    })
        
        # Pattern 2: Star patterns (one node connected to many)
        centrality = GraphUtils.calculate_centrality(G)
        
        for node, cent in centrality.items():
            if cent['degree'] > 0.5:  # High degree centrality
                neighbors = list(G.neighbors(node))
                
                if len(neighbors) >= 3:
                    patterns.append({
                        'type': 'star_pattern',
                        'center_node': node,
                        'connected_nodes': neighbors,
                        'num_connections': len(neighbors),
                        'suspicion_score': cent['degree'] * len(neighbors)
                    })
        
        # Pattern 3: Temporal clustering
        # Group by timestamp
        temporal_groups = {}
        for node in G.nodes():
            timestamp = G.nodes[node].get('timestamp', 0)
            if timestamp:
                # Group by hour
                hour_key = int(timestamp // 3600)
                if hour_key not in temporal_groups:
                    temporal_groups[hour_key] = []
                temporal_groups[hour_key].append(node)
        
        for hour_key, nodes in temporal_groups.items():
            if len(nodes) >= 3:
                # Check if they're connected
                subgraph = G.subgraph(nodes)
                if subgraph.number_of_edges() > 0:
                    patterns.append({
                        'type': 'temporal_cluster',
                        'time_window': hour_key,
                        'nodes': nodes,
                        'num_nodes': len(nodes),
                        'num_edges': subgraph.number_of_edges(),
                        'suspicion_score': len(nodes) * subgraph.number_of_edges() / 10
                    })
        
        # Sort by suspicion score
        patterns.sort(key=lambda x: x.get('suspicion_score', 0), reverse=True)
        
        logger.info(f"Found {len(patterns)} suspicious patterns")
        
        return patterns
    
    @staticmethod
    def calculate_graph_metrics(G: nx.Graph) -> Dict:
        """
        Calculate graph-level metrics.
        
        Args:
            G: NetworkX graph
            
        Returns:
            dict: Graph metrics
        """
        metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'num_connected_components': nx.number_connected_components(G),
        }
        
        # Average clustering coefficient
        try:
            metrics['avg_clustering'] = nx.average_clustering(G)
        except:
            metrics['avg_clustering'] = 0.0
        
        # Diameter (if connected)
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            metrics['diameter'] = None
            metrics['avg_shortest_path'] = None
        
        # Degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        if degrees:
            metrics['avg_degree'] = np.mean(degrees)
            metrics['max_degree'] = max(degrees)
            metrics['min_degree'] = min(degrees)
        
        return metrics
    
    @staticmethod
    def extract_subgraph(
        G: nx.Graph,
        nodes: List[str]
    ) -> nx.Graph:
        """
        Extract subgraph containing specified nodes.
        
        Args:
            G: NetworkX graph
            nodes: List of node IDs
            
        Returns:
            Subgraph
        """
        subgraph = G.subgraph(nodes).copy()
        
        logger.debug(
            f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, "
            f"{subgraph.number_of_edges()} edges"
        )
        
        return subgraph
    
    @staticmethod
    def get_node_neighbors(
        G: nx.Graph,
        node: str,
        depth: int = 1
    ) -> Set[str]:
        """
        Get neighbors of node up to specified depth.
        
        Args:
            G: NetworkX graph
            node: Node ID
            depth: Neighborhood depth
            
        Returns:
            Set of neighbor node IDs
        """
        if node not in G:
            return set()
        
        neighbors = {node}
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                next_level.update(G.neighbors(n))
            neighbors.update(next_level)
            current_level = next_level
        
        neighbors.discard(node)  # Remove the original node
        
        return neighbors
    
    @staticmethod
    def to_dict(G: nx.Graph) -> Dict:
        """
        Convert graph to dictionary for serialization.
        
        Args:
            G: NetworkX graph
            
        Returns:
            dict: Graph data
        """
        return nx.node_link_data(G)
    
    @staticmethod
    def from_dict(data: Dict) -> nx.Graph:
        """
        Create graph from dictionary.
        
        Args:
            data: Graph data dictionary
            
        Returns:
            NetworkX graph
        """
        return nx.node_link_graph(data)


# Convenience functions

def create_graph_from_submissions(submissions: List[Dict]) -> nx.Graph:
    """Create coordination graph from submissions."""
    return GraphUtils.create_submission_graph(submissions)


def detect_coordination(G: nx.Graph) -> Tuple[Dict, List]:
    """Detect coordination patterns in graph."""
    communities = GraphUtils.detect_communities(G)
    patterns = GraphUtils.find_suspicious_patterns(G, communities)
    return communities, patterns
