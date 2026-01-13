"""
Model Cache - In-memory caching for ML models

Provides:
- Singleton model instances
- LRU caching for embeddings
- Memory management
- Lazy loading
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Any

import numpy as np

from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class ModelCache:
    """
    Singleton cache for ML models.
    
    Features:
    - Lazy loading of models
    - Singleton pattern for shared instances
    - LRU caching for embeddings
    - Memory-efficient storage
    """
    
    _instance = None
    _models: Dict[str, Any] = {}
    _embedding_cache: Dict[str, np.ndarray] = {}
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache."""
        if self._initialized:
            return
        
        self._initialized = True
        self.max_cache_size = 1000  # Max cached embeddings
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info("ModelCache initialized")
    
    def get_clip_model(self, device: Optional[str] = None):
        """
        Get or create CLIP model instance.
        
        Args:
            device: Device to use
            
        Returns:
            CLIPModel instance
        """
        if 'clip' not in self._models:
            logger.info("Loading CLIP model (first access)")
            from backend.models.clip_model import CLIPModel
            
            self._models['clip'] = CLIPModel(device=device)
            self._models['clip'].load()
        
        return self._models['clip']
    
    def get_wav2vec_model(self, device: Optional[str] = None):
        """
        Get or create Wav2Vec2 model instance.
        
        Args:
            device: Device to use
            
        Returns:
            Wav2Vec2Model instance
        """
        if 'wav2vec' not in self._models:
            logger.info("Loading Wav2Vec2 model (first access)")
            from backend.models.wav2vec_model import Wav2Vec2Model
            
            self._models['wav2vec'] = Wav2Vec2Model(device=device)
            self._models['wav2vec'].load()
        
        return self._models['wav2vec']
    
    def get_blip_model(self, device: Optional[str] = None):
        """
        Get or create BLIP model instance.
        
        Args:
            device: Device to use
            
        Returns:
            BLIPModel instance
        """
        if 'blip' not in self._models:
            logger.info("Loading BLIP model (first access)")
            from backend.models.blip_model import BLIPModel
            
            self._models['blip'] = BLIPModel(device=device)
            self._models['blip'].load()
        
        return self._models['blip']
    
    def get_sentence_transformer(self, device: Optional[str] = None):
        """
        Get or create Sentence Transformer instance.
        
        Args:
            device: Device to use
            
        Returns:
            SentenceTransformerModel instance
        """
        if 'sentence_transformer' not in self._models:
            logger.info("Loading Sentence Transformer (first access)")
            from backend.models.sentence_transformer import SentenceTransformerModel
            
            self._models['sentence_transformer'] = SentenceTransformerModel(device=device)
            self._models['sentence_transformer'].load()
        
        return self._models['sentence_transformer']
    
    def cache_embedding(self, key: str, embedding: np.ndarray):
        """
        Cache embedding with key.
        
        Args:
            key: Cache key (e.g., hash of text)
            embedding: Embedding array
        """
        # Evict oldest if cache is full
        if len(self._embedding_cache) >= self.max_cache_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
            logger.debug(f"Evicted embedding from cache: {oldest_key}")
        
        self._embedding_cache[key] = embedding
    
    def get_cached_embedding(self, key: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.
        
        Args:
            key: Cache key
            
        Returns:
            Embedding array or None if not cached
        """
        if key in self._embedding_cache:
            self.cache_hits += 1
            return self._embedding_cache[key]
        
        self.cache_misses += 1
        return None
    
    def clear_embedding_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def unload_model(self, model_name: str):
        """
        Unload a specific model.
        
        Args:
            model_name: Model identifier
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    def unload_all_models(self):
        """Unload all models."""
        self._models.clear()
        logger.info("All models unloaded")
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache stats
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'loaded_models': list(self._models.keys()),
            'num_loaded_models': len(self._models),
            'embedding_cache_size': len(self._embedding_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': hit_ratio
        }
    
    def get_memory_usage(self) -> dict:
        """
        Estimate memory usage.
        
        Returns:
            dict: Memory usage info
        """
        # Estimate model sizes (approximate)
        model_sizes = {
            'clip': 350,  # MB
            'wav2vec': 360,
            'blip': 500,
            'sentence_transformer': 80
        }
        
        total_model_memory = sum(
            model_sizes.get(name, 100)
            for name in self._models.keys()
        )
        
        # Estimate embedding cache size
        embedding_memory = 0
        for embedding in self._embedding_cache.values():
            embedding_memory += embedding.nbytes / (1024 ** 2)  # MB
        
        return {
            'total_model_memory_mb': total_model_memory,
            'embedding_cache_memory_mb': embedding_memory,
            'total_memory_mb': total_model_memory + embedding_memory
        }


# Global cache instance
_cache = ModelCache()


# Convenience functions

def get_clip_model(device: Optional[str] = None):
    """Get CLIP model from cache."""
    return _cache.get_clip_model(device)


def get_wav2vec_model(device: Optional[str] = None):
    """Get Wav2Vec2 model from cache."""
    return _cache.get_wav2vec_model(device)


def get_blip_model(device: Optional[str] = None):
    """Get BLIP model from cache."""
    return _cache.get_blip_model(device)


def get_sentence_transformer(device: Optional[str] = None):
    """Get Sentence Transformer from cache."""
    return _cache.get_sentence_transformer(device)


def clear_model_cache():
    """Clear all models from cache."""
    _cache.unload_all_models()
    _cache.clear_embedding_cache()


def get_cache_stats() -> dict:
    """Get cache statistics."""
    return _cache.get_cache_stats()


@lru_cache(maxsize=128)
def get_text_hash(text: str) -> str:
    """
    Get hash of text for caching.
    
    Args:
        text: Input text
        
    Returns:
        str: Hash string
    """
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]
