"""
Models Package - ML model management and registry

Provides centralized access to all ML models:
- CLIP: Deepfake detection
- Wav2Vec2: Audio feature extraction
- BLIP: Image captioning
- Sentence Transformer: Text embeddings
- Model Cache: Efficient model loading and caching

Usage:
    from backend.models import get_clip_model, get_sentence_transformer
    
    clip = get_clip_model()
    score = clip.predict_deepfake('image.jpg')
"""

import logging
from typing import Optional, Dict, Any

from backend.models.base_model import BaseModel
from backend.models.clip_model import CLIPModel
from backend.models.wav2vec_model import Wav2Vec2Model
from backend.models.blip_model import BLIPModel
from backend.models.sentence_transformer import SentenceTransformerModel
from backend.models.model_cache import (
    ModelCache,
    get_clip_model,
    get_wav2vec_model,
    get_blip_model,
    get_sentence_transformer,
    clear_model_cache,
    get_cache_stats,
    get_text_hash
)

# Initialize logger
logger = logging.getLogger(__name__)


# ==================== MODEL REGISTRY ====================

# Registry of all available models
MODEL_REGISTRY: Dict[str, type] = {
    'clip': CLIPModel,
    'wav2vec': Wav2Vec2Model,
    'blip': BLIPModel,
    'sentence_transformer': SentenceTransformerModel
}


def get_model_class(model_name: str) -> Optional[type]:
    """
    Get model class by name.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Model class or None if not found
    """
    return MODEL_REGISTRY.get(model_name.lower())


def list_available_models() -> list:
    """
    List all available models.
    
    Returns:
        List of model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_name: Model identifier
        
    Returns:
        dict: Model information
    """
    model_class = get_model_class(model_name)
    
    if model_class is None:
        return {
            'error': f'Model {model_name} not found',
            'available_models': list_available_models()
        }
    
    # Get model sizes from constants
    from backend.constants import MODEL_SIZES
    
    info = {
        'name': model_name,
        'class': model_class.__name__,
        'size_mb': MODEL_SIZES.get(model_name, 'unknown'),
        'description': model_class.__doc__.strip().split('\n')[0] if model_class.__doc__ else 'No description'
    }
    
    # Add model-specific info if available
    if hasattr(model_class, 'MODEL_NAME'):
        info['huggingface_name'] = model_class.MODEL_NAME
    
    return info


def get_all_models_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all models.
    
    Returns:
        dict: Information for all models
    """
    return {
        model_name: get_model_info(model_name)
        for model_name in MODEL_REGISTRY.keys()
    }


def preload_models(
    models: Optional[list] = None,
    device: Optional[str] = None
) -> Dict[str, bool]:
    """
    Preload specified models into cache.
    
    Args:
        models: List of model names to load (loads all if None)
        device: Device to load models on
        
    Returns:
        dict: Loading status for each model
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    
    results = {}
    
    for model_name in models:
        try:
            logger.info(f"Preloading model: {model_name}")
            
            if model_name == 'clip':
                get_clip_model(device)
            elif model_name == 'wav2vec':
                get_wav2vec_model(device)
            elif model_name == 'blip':
                get_blip_model(device)
            elif model_name == 'sentence_transformer':
                get_sentence_transformer(device)
            else:
                logger.warning(f"Unknown model: {model_name}")
                results[model_name] = False
                continue
            
            results[model_name] = True
            logger.info(f"Successfully preloaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to preload {model_name}: {e}")
            results[model_name] = False
    
    return results


def unload_all_models():
    """
    Unload all models from cache.
    """
    logger.info("Unloading all models")
    clear_model_cache()
    logger.info("All models unloaded")


def get_total_memory_usage() -> Dict[str, Any]:
    """
    Get total memory usage of all loaded models.
    
    Returns:
        dict: Memory usage information
    """
    cache = ModelCache()
    memory_info = cache.get_memory_usage()
    
    # Add cache statistics
    cache_stats = get_cache_stats()
    
    return {
        'models': memory_info,
        'cache': cache_stats,
        'total_memory_mb': memory_info.get('total_memory_mb', 0)
    }


# ==================== EXPORT API ====================

__all__ = [
    # Base classes
    'BaseModel',
    
    # Model classes
    'CLIPModel',
    'Wav2Vec2Model',
    'BLIPModel',
    'SentenceTransformerModel',
    
    # Cache management
    'ModelCache',
    'get_clip_model',
    'get_wav2vec_model',
    'get_blip_model',
    'get_sentence_transformer',
    'clear_model_cache',
    'get_cache_stats',
    'get_text_hash',
    
    # Registry functions
    'get_model_class',
    'list_available_models',
    'get_model_info',
    'get_all_models_info',
    'preload_models',
    'unload_all_models',
    'get_total_memory_usage',
    
    # Registry constant
    'MODEL_REGISTRY'
]


# ==================== MODULE INITIALIZATION ====================

logger.info(f"Models package initialized with {len(MODEL_REGISTRY)} models")
logger.debug(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
