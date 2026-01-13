"""
Wav2Vec2 Model - Audio feature extraction

Uses facebook/wav2vec2-base for:
- Audio feature extraction
- Speech representation learning
- Audio authenticity analysis
- Cross-modal verification
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from backend.models.base_model import BaseModel
from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class Wav2Vec2Model(BaseModel):
    """
    Wav2Vec2 model for audio feature extraction.
    
    Model: facebook/wav2vec2-base
    Size: ~360MB
    
    Features:
    - Audio to embeddings
    - Feature extraction from speech
    - Audio authenticity analysis
    - Support for various audio formats
    """
    
    MODEL_NAME = "facebook/wav2vec2-base"
    SAMPLE_RATE = 16000  # Required sample rate
    EMBEDDING_DIM = 768  # Base model embedding dimension
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize Wav2Vec2 model.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision for faster inference
        """
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            use_fp16=use_fp16
        )
        
        self.sample_rate = self.SAMPLE_RATE
        self.embedding_dim = self.EMBEDDING_DIM
    
    def _load_model(self):
        """Load Wav2Vec2 model and processor."""
        logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
        
        try:
            # Load processor (handles audio preprocessing)
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.model_name
            )
            
            # Load model
            self.model = Wav2Vec2Model.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested and available
            if self.use_fp16 and self.device != 'cpu':
                self.model.half()
            
            logger.info(
                f"Wav2Vec2 model loaded successfully on {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {e}")
            raise
    
    def extract_features(
        self,
        audio_path: Union[str, Path, np.ndarray],
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """
        Extract features from audio file or array.
        
        Args:
            audio_path: Path to audio file or audio array
            aggregate: How to aggregate features ('mean', 'max', 'last', 'all')
            
        Returns:
            np.ndarray: Feature embeddings
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load and preprocess audio
        if isinstance(audio_path, (str, Path)):
            from backend.utils.audio_utils import load_audio
            
            # Load audio at correct sample rate
            audio_array, sr = load_audio(
                audio_path,
                target_sr=self.sample_rate
            )
        else:
            audio_array = audio_path
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get last hidden state
            hidden_states = outputs.last_hidden_state  # [batch, time, features]
        
        # Aggregate features
        if aggregate == 'mean':
            # Mean pooling over time dimension
            features = hidden_states.mean(dim=1)
        
        elif aggregate == 'max':
            # Max pooling over time dimension
            features = hidden_states.max(dim=1)[0]
        
        elif aggregate == 'last':
            # Last timestep
            features = hidden_states[:, -1, :]
        
        elif aggregate == 'all':
            # Return all timesteps
            features = hidden_states
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        # Convert to numpy
        features_np = features.cpu().numpy()
        
        # Return single vector if batch size is 1
        if features_np.shape[0] == 1 and aggregate != 'all':
            return features_np[0]
        
        return features_np
    
    def compute_audio_similarity(
        self,
        audio1: Union[str, Path, np.ndarray],
        audio2: Union[str, Path, np.ndarray]
    ) -> float:
        """
        Compute similarity between two audio samples.
        
        Args:
            audio1: First audio (path or array)
            audio2: Second audio (path or array)
            
        Returns:
            float: Cosine similarity score [0, 1]
        """
        # Extract features
        features1 = self.extract_features(audio1, aggregate='mean')
        features2 = self.extract_features(audio2, aggregate='mean')
        
        # Compute cosine similarity
        similarity = np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )
        
        return float(similarity)
    
    def analyze_audio_authenticity(
        self,
        audio_path: Union[str, Path]
    ) -> dict:
        """
        Analyze audio for authenticity indicators.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Analysis results with authenticity scores
        """
        from backend.utils.audio_utils import (
            load_audio,
            compute_spectral_features,
            detect_audio_artifacts
        )
        
        # Load audio
        audio_array, sr = load_audio(
            audio_path,
            target_sr=self.sample_rate
        )
        
        # Extract deep features
        deep_features = self.extract_features(audio_array, aggregate='mean')
        
        # Compute spectral features
        spectral_features = compute_spectral_features(audio_array, sr)
        
        # Detect artifacts
        artifacts = detect_audio_artifacts(audio_array, sr)
        
        # Compute feature statistics
        feature_stats = {
            'mean': float(deep_features.mean()),
            'std': float(deep_features.std()),
            'max': float(deep_features.max()),
            'min': float(deep_features.min())
        }
        
        # Authenticity score (heuristic-based)
        # Lower artifact count and normal feature stats indicate authenticity
        authenticity_score = 1.0 - (artifacts['total_artifacts'] / 100)
        authenticity_score = max(0.0, min(1.0, authenticity_score))
        
        return {
            'authenticity_score': authenticity_score,
            'feature_stats': feature_stats,
            'spectral_features': spectral_features,
            'artifacts': artifacts,
            'embedding_dim': deep_features.shape[0]
        }
    
    def batch_extract_features(
        self,
        audio_paths: list,
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            aggregate: Aggregation method
            
        Returns:
            np.ndarray: Batch of features [num_audios, embedding_dim]
        """
        features_list = []
        
        for audio_path in audio_paths:
            try:
                features = self.extract_features(audio_path, aggregate)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features from {audio_path}: {e}")
                # Add zero vector as placeholder
                features_list.append(np.zeros(self.embedding_dim))
        
        return np.vstack(features_list)
    
    def get_temporal_features(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int]:
        """
        Get temporal features (time-series) from audio.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (features, num_timesteps)
        """
        # Extract all temporal features
        features = self.extract_features(audio_path, aggregate='all')
        
        # features shape: [1, time, features] or [time, features]
        if len(features.shape) == 3:
            features = features[0]  # Remove batch dimension
        
        num_timesteps = features.shape[0]
        
        return features, num_timesteps
    
    def detect_voice_activity(
        self,
        audio_path: Union[str, Path],
        threshold: float = 0.5
    ) -> dict:
        """
        Detect voice activity in audio.
        
        Args:
            audio_path: Path to audio file
            threshold: Energy threshold for voice detection
            
        Returns:
            dict: Voice activity detection results
        """
        from backend.utils.audio_utils import load_audio
        
        # Load audio
        audio_array, sr = load_audio(
            audio_path,
            target_sr=self.sample_rate
        )
        
        # Extract temporal features
        features, num_timesteps = self.get_temporal_features(audio_path)
        
        # Compute energy per timestep
        energy = np.linalg.norm(features, axis=1)
        
        # Normalize energy
        energy_normalized = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)
        
        # Detect voice segments
        voice_mask = energy_normalized > threshold
        
        # Calculate statistics
        voice_ratio = voice_mask.sum() / len(voice_mask)
        num_segments = np.diff(voice_mask.astype(int)).sum() // 2
        
        return {
            'voice_ratio': float(voice_ratio),
            'num_voice_segments': int(num_segments),
            'total_timesteps': num_timesteps,
            'voice_mask': voice_mask.tolist(),
            'energy_profile': energy_normalized.tolist()
        }
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model info
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'sample_rate': self.sample_rate,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'use_fp16': self.use_fp16,
            'model_size_mb': 360  # Approximate
        }
