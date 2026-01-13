"""
Base Model - Abstract interface for ML models

Provides:
- Common model loading/unloading interface
- Device management (CPU/GPU)
- Memory optimization utilities
- Standard error handling
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch

from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Provides common functionality:
    - Model loading/unloading
    - Device management
    - Memory optimization
    - Error handling
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize base model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision for faster inference
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        # Auto-detect device if not specified
        if device is None:
            self.device = self._auto_detect_device()
        else:
            self.device = device
        
        # Model and processor will be loaded lazily
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        self._is_loaded = False
        
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(device: {self.device}, fp16: {use_fp16})"
        )
    
    def _auto_detect_device(self) -> str:
        """
        Auto-detect best available device.
        
        Returns:
            str: Device identifier ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("CUDA not available, using CPU")
        
        return device
    
    @abstractmethod
    def _load_model(self):
        """
        Load model and associated components.
        
        Must be implemented by subclasses.
        """
        pass
    
    def load(self):
        """
        Load model if not already loaded.
        """
        if self._is_loaded:
            logger.debug(f"{self.__class__.__name__} already loaded")
            return
        
        try:
            logger.info(f"Loading {self.__class__.__name__}...")
            self._load_model()
            self._is_loaded = True
            logger.info(f"{self.__class__.__name__} loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__}: {e}")
            raise
    
    def unload(self):
        """
        Unload model to free memory.
        """
        if not self._is_loaded:
            logger.debug(f"{self.__class__.__name__} not loaded")
            return
        
        try:
            # Move model to CPU and delete references
            if self.model is not None:
                self.model.cpu()
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            self._is_loaded = False
            logger.info(f"{self.__class__.__name__} unloaded")
        
        except Exception as e:
            logger.error(f"Failed to unload {self.__class__.__name__}: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_device(self) -> str:
        """Get current device."""
        return self.device
    
    def to(self, device: str):
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu' or 'cuda')
        """
        if not self._is_loaded:
            logger.warning("Model not loaded, cannot move device")
            return
        
        try:
            self.device = device
            
            if self.model is not None:
                self.model.to(device)
            
            logger.info(f"{self.__class__.__name__} moved to {device}")
        
        except Exception as e:
            logger.error(f"Failed to move model to {device}: {e}")
            raise
    
    def get_memory_usage(self) -> dict:
        """
        Get approximate memory usage.
        
        Returns:
            dict: Memory usage information
        """
        if not self._is_loaded:
            return {
                'is_loaded': False,
                'device': self.device,
                'memory_mb': 0
            }
        
        # Estimate model size
        total_params = 0
        if self.model is not None:
            for param in self.model.parameters():
                total_params += param.numel()
        
        # Calculate memory (assuming fp32 or fp16)
        bytes_per_param = 2 if self.use_fp16 else 4
        memory_bytes = total_params * bytes_per_param
        memory_mb = memory_bytes / (1024 ** 2)
        
        result = {
            'is_loaded': True,
            'device': self.device,
            'total_parameters': total_params,
            'memory_mb': memory_mb,
            'precision': 'fp16' if self.use_fp16 else 'fp32'
        }
        
        # Add CUDA memory info if using GPU
        if self.device == 'cuda' and torch.cuda.is_available():
            result['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            result['cuda_cached_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"loaded={self._is_loaded})"
        )
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._is_loaded:
            try:
                self.unload()
            except:
                pass
