"""
Sentence Transformer Model - Text embedding for semantic similarity

Uses sentence-transformers/all-MiniLM-L6-v2 for:
- Text embedding generation
- Semantic similarity calculation
- Stylometric analysis support
- Coordination detection
"""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from backend.models.base_model import BaseModel
from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class SentenceTransformerModel(BaseModel):
    """
    Sentence transformer for text embeddings.
    
    Model: sentence-transformers/all-MiniLM-L6-v2
    Size: ~80MB
    
    Features:
    - Text to 384-dim embeddings
    - Cosine similarity computation
    - Batch processing
    - GPU acceleration support
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    MAX_LENGTH = 256
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize sentence transformer model.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision for faster inference
        """
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            use_fp16=use_fp16
        )
        
        self.embedding_dim = self.EMBEDDING_DIM
        self.max_length = self.MAX_LENGTH
    
    def _load_model(self):
        """Load sentence transformer model and tokenizer."""
        logger.info(f"Loading sentence transformer: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.max_length
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested and available
            if self.use_fp16 and self.device != 'cpu':
                self.model.half()
            
            logger.info(
                f"Sentence transformer loaded successfully on {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise
    
    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling to get sentence embeddings.
        
        Args:
            model_output: Model output tensor
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        
        # Sum embeddings
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum mask
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Mean pooling
        embeddings = sum_embeddings / sum_mask
        
        return embeddings
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Normalize embeddings to unit length
            
        Returns:
            np.ndarray: Embeddings array [num_texts, embedding_dim]
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
                # Mean pooling
                embeddings = self._mean_pooling(
                    model_output,
                    encoded_input['attention_mask']
                )
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(
                        embeddings,
                        p=2,
                        dim=1
                    )
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings_np)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Return single embedding if single input
        if single_input:
            return all_embeddings[0]
        
        return all_embeddings
    
    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between two texts or embeddings.
        
        Args:
            text1: First text or embedding
            text2: Second text or embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            
        Returns:
            float: Similarity score
        """
        # Get embeddings if texts provided
        if isinstance(text1, str):
            emb1 = self.encode(text1)
        else:
            emb1 = text1
        
        if isinstance(text2, str):
            emb2 = self.encode(text2)
        else:
            emb2 = text2
        
        # Calculate similarity
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
        
        elif metric == 'euclidean':
            # Euclidean distance (inverted to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        
        elif metric == 'dot':
            # Dot product
            similarity = np.dot(emb1, emb2)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return float(similarity)
    
    def batch_similarity(
        self,
        texts: List[str],
        query: str,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Calculate similarity between query and multiple texts.
        
        Args:
            texts: List of texts
            query: Query text
            metric: Similarity metric
            
        Returns:
            np.ndarray: Similarity scores
        """
        # Encode all texts including query
        all_texts = texts + [query]
        embeddings = self.encode(all_texts)
        
        # Split query embedding
        text_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1]
        
        # Calculate similarities
        if metric == 'cosine':
            # Batch cosine similarity
            similarities = np.dot(text_embeddings, query_embedding)
        
        elif metric == 'euclidean':
            # Batch euclidean distance
            distances = np.linalg.norm(
                text_embeddings - query_embedding,
                axis=1
            )
            similarities = 1 / (1 + distances)
        
        elif metric == 'dot':
            # Batch dot product
            similarities = np.dot(text_embeddings, query_embedding)
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def find_most_similar(
        self,
        texts: List[str],
        query: str,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to query.
        
        Args:
            texts: List of texts to search
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (index, score, text) tuples
        """
        # Calculate similarities
        similarities = self.batch_similarity(texts, query)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = [
            (idx, similarities[idx], texts[idx])
            for idx in top_indices
        ]
        
        return results
    
    def cluster_texts(
        self,
        texts: List[str],
        num_clusters: int = 5
    ) -> np.ndarray:
        """
        Cluster texts using embeddings.
        
        Args:
            texts: List of texts
            num_clusters: Number of clusters
            
        Returns:
            np.ndarray: Cluster labels
        """
        from sklearn.cluster import KMeans
        
        # Encode texts
        embeddings = self.encode(texts)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        return labels
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model info
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'max_length': self.max_length,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'use_fp16': self.use_fp16,
            'model_size_mb': 80  # Approximate
        }


"""Usage Examples
Example 1: Text Embedding
python
from backend.models.model_cache import get_sentence_transformer

# Get model
model = get_sentence_transformer()

# Single text
text = "This is a corruption allegation narrative."
embedding = model.encode(text)
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch encoding
texts = ["First narrative", "Second narrative", "Third narrative"]
embeddings = model.encode(texts)
print(f"Batch embeddings: {embeddings.shape}")  # (3, 384)
Example 2: Similarity Analysis
python
# Compare narratives
narrative1 = "Official accepted bribe from contractor"
narrative2 = "Contractor paid money to government official"
narrative3 = "Weather report for today"

similarity_12 = model.similarity(narrative1, narrative2)
similarity_13 = model.similarity(narrative1, narrative3)

print(f"Similar narratives: {similarity_12:.3f}")  # High score
print(f"Different topics: {similarity_13:.3f}")    # Low score
Example 3: Finding Similar Submissions
python
# Find coordinated submissions
submissions = [
    "Narrative about official X accepting bribe",
    "Similar story about official X taking money",
    "Unrelated complaint about road conditions",
    "Another report about official X and bribery"
]

query = "Investigation into official X corruption"

results = model.find_most_similar(submissions, query, top_k=3)

for idx, score, text in results:
    print(f"Score: {score:.3f} - {text[:50]}...")
Example 4: Model Cache Management
python
from backend.models.model_cache import (
    get_cache_stats,
    clear_model_cache
)

# Get statistics
stats = get_cache_stats()
print(f"Loaded models: {stats['loaded_models']}")
print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
print(f"Embedding cache: {stats['embedding_cache_size']} items")

# Clear cache if needed
clear_model_cache()"""