"""
Layer 3: Coordination Detection System
Detects coordinated attacks using graph analysis and stylometric features.

Input: Submission with credibility scores
Output: Coordination flags and community detection results
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from networkx.algorithms import community
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Initialize logger
logger = logging.getLogger(__name__)


class Layer3Coordination:
    """
    Layer 3: Coordination Detection System
    
    Implements:
    - Graph construction with multiple edge types (style, temporal, content)
    - Stylometric feature extraction (TTR, lexical density, POS patterns)
    - Louvain community detection
    - One-Class SVM for anomaly detection
    - Coordination flagging with confidence scores
    """
    
    def __init__(
        self,
        storage_service,
        text_utils,
        graph_utils,
        min_similarity: float = 0.7,
        time_window_hours: int = 24,
        min_community_size: int = 3
    ):
        """
        Initialize Layer 3 with configuration.
        
        Args:
            storage_service: Storage service for accessing submissions
            text_utils: Text utilities for stylometric analysis
            graph_utils: Graph utilities for NetworkX operations
            min_similarity: Minimum similarity threshold for edges
            time_window_hours: Time window for temporal edges
            min_community_size: Minimum size for flagged communities
        """
        self.storage = storage_service
        self.text_utils = text_utils
        self.graph_utils = graph_utils
        
        self.min_similarity = min_similarity
        self.time_window = timedelta(hours=time_window_hours)
        self.min_community_size = min_community_size
        
        # Initialize One-Class SVM for anomaly detection
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        logger.info(
            f"Layer 3 (Coordination) initialized "
            f"(similarity≥{min_similarity}, window={time_window_hours}h)"
        )
    
    def process(
        self,
        submission_id: str,
        text_narrative: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Process submission through coordination detection.
        
        Args:
            submission_id: Unique submission identifier
            text_narrative: Text narrative for stylometric analysis
            timestamp: Submission timestamp
            
        Returns:
            dict: Coordination detection results
            
        Raises:
            ValueError: If detection fails
        """
        logger.info(f"Layer 3 processing submission {submission_id}")
        
        try:
            # Step 1: Get all recent submissions
            recent_submissions = self._get_recent_submissions(timestamp)
            logger.debug(f"Found {len(recent_submissions)} recent submissions")
            
            # Step 2: Extract stylometric features for current submission
            current_features = self._extract_stylometric_features(
                submission_id,
                text_narrative
            )
            
            # Step 3: Build coordination graph
            graph = self._build_coordination_graph(
                recent_submissions,
                submission_id,
                current_features,
                text_narrative
            )
            
            # Step 4: Detect communities
            communities = self._detect_communities(graph)
            logger.debug(f"Detected {len(communities)} communities")
            
            # Step 5: Check if current submission is in suspicious community
            coordination_info = self._check_coordination(
                submission_id,
                communities,
                graph,
                current_features,
                recent_submissions
            )
            
            # Step 6: Anomaly detection (if enough data)
            anomaly_score = self._detect_anomaly(
                current_features,
                recent_submissions
            )
            
            result = {
                "submission_id": submission_id,
                "flagged": coordination_info['flagged'],
                "confidence": coordination_info['confidence'],
                "community_id": coordination_info.get('community_id'),
                "community_size": coordination_info.get('community_size'),
                "similarity_scores": coordination_info.get('similarity_scores', {}),
                "anomaly_score": anomaly_score,
                "graph_metrics": {
                    "total_nodes": graph.number_of_nodes(),
                    "total_edges": graph.number_of_edges(),
                    "num_communities": len(communities)
                },
                "stylometric_features": current_features,
                "layer3_status": "completed",
                "timestamp_analyzed": datetime.utcnow().isoformat()
            }
            
            if result['flagged']:
                logger.warning(
                    f"Coordination detected for {submission_id} "
                    f"(confidence={result['confidence']:.3f}, "
                    f"community_size={result['community_size']})"
                )
            else:
                logger.info(f"No coordination detected for {submission_id}")
            
            return result
            
        except Exception as e:
            logger.error(
                f"Layer 3 processing failed for {submission_id}: {e}",
                exc_info=True
            )
            raise ValueError(f"Coordination detection failed: {str(e)}")
    
    def _get_recent_submissions(
        self,
        timestamp: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get submissions within time window.
        
        Args:
            timestamp: Reference timestamp
            
        Returns:
            list: Recent submission data
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        cutoff_time = timestamp - self.time_window
        
        # Get all submissions from storage
        all_submissions = self.storage.get_all_submissions()
        
        # Filter by time window
        recent = []
        for sub in all_submissions:
            sub_time_str = sub.get('timestamp_submission')
            if sub_time_str:
                try:
                    sub_time = datetime.fromisoformat(sub_time_str.replace('Z', ''))
                    if sub_time >= cutoff_time:
                        recent.append(sub)
                except Exception as e:
                    logger.warning(f"Invalid timestamp format: {e}")
        
        return recent
    
    def _extract_stylometric_features(
        self,
        submission_id: str,
        text: Optional[str]
    ) -> Dict:
        """
        Extract stylometric features from text.
        
        Features:
        - Type-Token Ratio (TTR)
        - Lexical density
        - Average word length
        - Sentence length statistics
        - POS tag distribution (simplified)
        
        Args:
            submission_id: Submission identifier
            text: Text to analyze
            
        Returns:
            dict: Stylometric features
        """
        if not text or len(text.strip()) < 10:
            # Return neutral features for short/missing text
            return {
                "ttr": 0.5,
                "lexical_density": 0.5,
                "avg_word_length": 5.0,
                "avg_sentence_length": 10.0,
                "text_length": 0,
                "has_text": False
            }
        
        try:
            # Use text_utils for feature extraction
            features = self.text_utils.extract_stylometric_features(text)
            features['has_text'] = True
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return {
                "ttr": 0.5,
                "lexical_density": 0.5,
                "avg_word_length": 5.0,
                "avg_sentence_length": 10.0,
                "text_length": len(text),
                "has_text": True
            }
    
    def _build_coordination_graph(
        self,
        submissions: List[Dict],
        current_id: str,
        current_features: Dict,
        current_text: Optional[str]
    ) -> nx.Graph:
        """
        Build coordination graph with multiple edge types.
        
        Edge types:
        - style: Stylometric similarity
        - temporal: Submitted close in time
        - content: Semantic similarity
        
        Args:
            submissions: List of recent submissions
            current_id: Current submission ID
            current_features: Current submission features
            current_text: Current submission text
            
        Returns:
            nx.Graph: Coordination graph
        """
        G = nx.Graph()
        
        # Add current submission as node
        G.add_node(
            current_id,
            features=current_features,
            text=current_text,
            timestamp=datetime.utcnow()
        )
        
        # Add other submissions as nodes and compute edges
        for sub in submissions:
            sub_id = sub.get('id')
            if not sub_id or sub_id == current_id:
                continue
            
            sub_text = sub.get('text_narrative', '')
            sub_features = sub.get('stylometric_features')
            
            # Extract features if not cached
            if not sub_features:
                sub_features = self._extract_stylometric_features(
                    sub_id,
                    sub_text
                )
            
            # Add node
            sub_time_str = sub.get('timestamp_submission')
            sub_time = datetime.utcnow()
            if sub_time_str:
                try:
                    sub_time = datetime.fromisoformat(sub_time_str.replace('Z', ''))
                except Exception:
                    pass
            
            G.add_node(
                sub_id,
                features=sub_features,
                text=sub_text,
                timestamp=sub_time
            )
            
            # Compute similarities
            style_sim = self._compute_style_similarity(
                current_features,
                sub_features
            )
            
            temporal_sim = self._compute_temporal_similarity(
                datetime.utcnow(),
                sub_time
            )
            
            content_sim = self._compute_content_similarity(
                current_text,
                sub_text
            )
            
            # Weighted overall similarity
            overall_sim = (
                0.4 * style_sim +
                0.3 * temporal_sim +
                0.3 * content_sim
            )
            
            # Add edge if similarity exceeds threshold
            if overall_sim >= self.min_similarity:
                G.add_edge(
                    current_id,
                    sub_id,
                    weight=overall_sim,
                    style=style_sim,
                    temporal=temporal_sim,
                    content=content_sim
                )
                logger.debug(
                    f"Edge added: {current_id} <-> {sub_id} "
                    f"(sim={overall_sim:.3f})"
                )
        
        logger.debug(
            f"Graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def _compute_style_similarity(
        self,
        features1: Dict,
        features2: Dict
    ) -> float:
        """
        Compute stylometric similarity between two submissions.
        
        Args:
            features1: Features of first submission
            features2: Features of second submission
            
        Returns:
            float: Similarity score [0, 1]
        """
        if not features1.get('has_text') or not features2.get('has_text'):
            return 0.0
        
        # Compare feature vectors
        keys = ['ttr', 'lexical_density', 'avg_word_length', 'avg_sentence_length']
        
        diffs = []
        for key in keys:
            val1 = features1.get(key, 0.5)
            val2 = features2.get(key, 0.5)
            
            # Normalize difference
            max_val = max(val1, val2) + 1e-6
            diff = abs(val1 - val2) / max_val
            diffs.append(diff)
        
        # Similarity = 1 - average difference
        similarity = 1.0 - np.mean(diffs)
        
        return max(0.0, min(1.0, similarity))
    
    def _compute_temporal_similarity(
        self,
        time1: datetime,
        time2: datetime
    ) -> float:
        """
        Compute temporal similarity (submissions close in time).
        
        Args:
            time1: First timestamp
            time2: Second timestamp
            
        Returns:
            float: Similarity score [0, 1]
        """
        time_diff = abs((time1 - time2).total_seconds())
        
        # Exponential decay: similarity = exp(-diff / half_life)
        # half_life = 6 hours (submissions 6 hours apart have sim=0.5)
        half_life = 6 * 3600  # 6 hours in seconds
        
        similarity = np.exp(-time_diff / half_life)
        
        return float(similarity)
    
    def _compute_content_similarity(
        self,
        text1: Optional[str],
        text2: Optional[str]
    ) -> float:
        """
        Compute semantic content similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score [0, 1]
        """
        if not text1 or not text2:
            return 0.0
        
        if len(text1) < 10 or len(text2) < 10:
            return 0.0
        
        try:
            # Use text_utils for similarity computation
            similarity = self.text_utils.compute_text_similarity(text1, text2)
            return similarity
            
        except Exception as e:
            logger.warning(f"Content similarity computation failed: {e}")
            return 0.0
    
    def _detect_communities(self, graph: nx.Graph) -> List[Set[str]]:
        """
        Detect communities using Louvain algorithm.
        
        Args:
            graph: Coordination graph
            
        Returns:
            list: List of communities (sets of node IDs)
        """
        if graph.number_of_nodes() < 2:
            return []
        
        if graph.number_of_edges() == 0:
            # No edges = no communities
            return []
        
        try:
            # Use Louvain community detection
            communities_generator = community.louvain_communities(
                graph,
                seed=42,
                weight='weight'
            )
            
            communities = list(communities_generator)
            
            # Filter out single-node communities
            communities = [c for c in communities if len(c) >= 2]
            
            logger.debug(f"Detected {len(communities)} multi-node communities")
            
            return communities
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return []
    
    def _check_coordination(
        self,
        submission_id: str,
        communities: List[Set[str]],
        graph: nx.Graph,
        current_features: Dict,
        all_submissions: List[Dict]
    ) -> Dict:
        """
        Check if submission is part of coordinated attack.
        
        Args:
            submission_id: Current submission ID
            communities: Detected communities
            graph: Coordination graph
            current_features: Current submission features
            all_submissions: All recent submissions
            
        Returns:
            dict: Coordination check results
        """
        # Find community containing current submission
        current_community = None
        community_id = None
        
        for idx, comm in enumerate(communities):
            if submission_id in comm:
                current_community = comm
                community_id = idx
                break
        
        if not current_community:
            # Not in any community
            return {
                'flagged': False,
                'confidence': 0.0,
                'community_id': None,
                'community_size': 0,
                'similarity_scores': {}
            }
        
        community_size = len(current_community)
        
        # Flag if community is large enough
        flagged = community_size >= self.min_community_size
        
        # Calculate confidence based on:
        # 1. Community size
        # 2. Edge weights (similarity)
        # 3. Graph density
        
        # Get edges within community
        community_edges = []
        for node1 in current_community:
            for node2 in current_community:
                if node1 < node2 and graph.has_edge(node1, node2):
                    community_edges.append((node1, node2))
        
        if community_edges:
            avg_similarity = np.mean([
                graph[u][v]['weight'] for u, v in community_edges
            ])
        else:
            avg_similarity = 0.0
        
        # Confidence calculation
        # Factor 1: Community size (normalized)
        size_factor = min(1.0, community_size / 10)
        
        # Factor 2: Average similarity
        sim_factor = avg_similarity
        
        # Factor 3: Temporal clustering
        if len(current_community) >= 2:
            timestamps = []
            for node_id in current_community:
                if graph.nodes[node_id].get('timestamp'):
                    timestamps.append(graph.nodes[node_id]['timestamp'])
            
            if len(timestamps) >= 2:
                time_diffs = []
                timestamps_sorted = sorted(timestamps)
                for i in range(len(timestamps_sorted) - 1):
                    diff = (timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds()
                    time_diffs.append(diff)
                
                avg_time_diff = np.mean(time_diffs)
                # Closer in time = higher confidence
                temporal_factor = np.exp(-avg_time_diff / 3600)  # 1 hour half-life
            else:
                temporal_factor = 0.5
        else:
            temporal_factor = 0.5
        
        # Weighted confidence
        confidence = (
            0.4 * size_factor +
            0.4 * sim_factor +
            0.2 * temporal_factor
        )
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Get similarity scores to community members
        similarity_scores = {}
        if graph.has_node(submission_id):
            for neighbor in graph.neighbors(submission_id):
                if neighbor in current_community:
                    similarity_scores[neighbor] = graph[submission_id][neighbor]['weight']
        
        return {
            'flagged': flagged,
            'confidence': confidence,
            'community_id': community_id,
            'community_size': community_size,
            'similarity_scores': similarity_scores,
            'avg_similarity': avg_similarity,
            'temporal_clustering': temporal_factor
        }
    
    def _detect_anomaly(
        self,
        current_features: Dict,
        recent_submissions: List[Dict]
    ) -> float:
        """
        Detect if submission is anomalous using One-Class SVM.
        
        Args:
            current_features: Current submission features
            recent_submissions: Recent submissions for training
            
        Returns:
            float: Anomaly score (higher = more anomalous)
        """
        # Need at least 10 samples to train
        if len(recent_submissions) < 10:
            return 0.0
        
        if not current_features.get('has_text'):
            return 0.0
        
        try:
            # Extract feature vectors
            feature_vectors = []
            for sub in recent_submissions:
                sub_features = sub.get('stylometric_features')
                if not sub_features:
                    sub_text = sub.get('text_narrative', '')
                    sub_features = self._extract_stylometric_features(
                        sub.get('id'),
                        sub_text
                    )
                
                if sub_features.get('has_text'):
                    vec = self._feature_dict_to_vector(sub_features)
                    feature_vectors.append(vec)
            
            if len(feature_vectors) < 10:
                return 0.0
            
            # Train One-Class SVM
            X_train = np.array(feature_vectors)
            X_train = self.scaler.fit_transform(X_train)
            
            svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
            svm.fit(X_train)
            
            # Predict on current submission
            current_vec = self._feature_dict_to_vector(current_features)
            current_vec = self.scaler.transform([current_vec])
            
            # Decision function: negative = anomaly
            decision = svm.decision_function(current_vec)[0]
            
            # Convert to [0, 1] score (higher = more anomalous)
            # Normalize decision function output
            anomaly_score = 1.0 / (1.0 + np.exp(decision))
            
            return float(anomaly_score)
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return 0.0
    
    def _feature_dict_to_vector(self, features: Dict) -> np.ndarray:
        """
        Convert feature dictionary to numpy vector.
        
        Args:
            features: Feature dictionary
            
        Returns:
            np.ndarray: Feature vector
        """
        keys = ['ttr', 'lexical_density', 'avg_word_length', 'avg_sentence_length']
        vector = [features.get(key, 0.5) for key in keys]
        return np.array(vector)
    
    def get_coordination_graph_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_similarity: Optional[float] = None
    ) -> Dict:
        """
        Get coordination graph data for visualization.
        
        Args:
            start_date: Filter start date
            end_date: Filter end date
            min_similarity: Minimum edge similarity
            
        Returns:
            dict: Graph data with nodes, edges, communities
        """
        try:
            # Get submissions in date range
            all_submissions = self.storage.get_all_submissions()
            
            filtered_submissions = []
            for sub in all_submissions:
                sub_time_str = sub.get('timestamp_submission')
                if sub_time_str:
                    try:
                        sub_time = datetime.fromisoformat(sub_time_str.replace('Z', ''))
                        
                        if start_date and sub_time < start_date:
                            continue
                        if end_date and sub_time > end_date:
                            continue
                        
                        filtered_submissions.append(sub)
                    except Exception:
                        pass
            
            # Build graph with all filtered submissions
            G = nx.Graph()
            
            for sub in filtered_submissions:
                sub_id = sub.get('id')
                G.add_node(
                    sub_id,
                    pseudonym=sub.get('pseudonym', 'N/A'),
                    score=sub.get('credibility', {}).get('final_score', 0.5),
                    flagged=sub.get('coordination', {}).get('flagged', False)
                )
            
            # Add edges based on cached coordination data
            threshold = min_similarity if min_similarity else self.min_similarity
            
            for sub in filtered_submissions:
                sub_id = sub.get('id')
                coord_data = sub.get('coordination', {})
                similarity_scores = coord_data.get('similarity_scores', {})
                
                for other_id, sim in similarity_scores.items():
                    if sim >= threshold and G.has_node(other_id):
                        if not G.has_edge(sub_id, other_id):
                            G.add_edge(sub_id, other_id, weight=sim)
            
            # Detect communities
            communities = self._detect_communities(G)
            
            # Convert to serializable format
            nodes = []
            for node_id in G.nodes():
                node_data = G.nodes[node_id]
                nodes.append({
                    'id': node_id,
                    'label': node_data.get('pseudonym', node_id[:8]),
                    'score': node_data.get('score', 0.5),
                    'flagged': node_data.get('flagged', False)
                })
            
            edges = []
            for u, v in G.edges():
                edges.append({
                    'source': u,
                    'target': v,
                    'weight': G[u][v]['weight']
                })
            
            community_data = []
            for idx, comm in enumerate(communities):
                community_data.append({
                    'id': idx,
                    'members': list(comm),
                    'size': len(comm)
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'communities': community_data,
                'total_submissions': len(nodes),
                'flagged_submissions': sum(1 for n in nodes if n['flagged'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph data: {e}")
            return {
                'nodes': [],
                'edges': [],
                'communities': [],
                'total_submissions': 0,
                'flagged_submissions': 0
            }


"""from backend.utils.text_utils import TextUtils, extract_features
from backend.utils.graph_utils import GraphUtils, detect_coordination

class CoordinationDetectionLayer:
    def analyze_submissions(self, submissions: List[Dict]):
        
        # Extract stylometric features
        for submission in submissions:
            if 'narrative' in submission:
                features = extract_features(submission['narrative'])
                submission['stylometric_features'] = features
        
        # Create graph
        G = GraphUtils.create_submission_graph(
            submissions,
            similarity_threshold=0.3
        )
        
        # Detect communities and patterns
        communities, patterns = detect_coordination(G)
        
        # Calculate graph metrics
        metrics = GraphUtils.calculate_graph_metrics(G)
        
        return {
            'graph': G,
            'communities': communities,
            'suspicious_patterns': patterns,
            'metrics': metrics
        }
"""