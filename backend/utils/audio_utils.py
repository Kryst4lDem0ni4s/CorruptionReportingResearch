"""
Audio Utils - Audio processing and feature extraction

Provides:
- Audio loading and resampling
- Feature extraction for Wav2Vec2
- Audio normalization
- Format conversion
- Basic audio analysis
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize logger
logger = logging.getLogger(__name__)


class AudioUtils:
    """
    Audio utilities for ML preprocessing.
    
    Features:
    - Audio loading (WAV, MP3 via scipy)
    - Resampling for Wav2Vec2 (16kHz)
    - Audio normalization
    - Feature extraction helpers
    - Quality checks
    """
    
    # Standard sample rate for speech models
    TARGET_SAMPLE_RATE = 16000
    
    @staticmethod
    def load_audio(
        audio_path: Path,
        target_sr: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (resamples if different)
            
        Returns:
            tuple: (audio_array, sample_rate) or None if load fails
        """
        try:
            # Try loading with scipy (supports WAV)
            sample_rate, audio = wavfile.read(audio_path)
            
            # Convert to float32 and normalize to [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                audio = audio.astype(np.float32)
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample if target sample rate specified
            if target_sr is not None and sample_rate != target_sr:
                audio = AudioUtils.resample_audio(audio, sample_rate, target_sr)
                sample_rate = target_sr
            
            logger.debug(
                f"Audio loaded: {audio_path.name} "
                f"(duration={len(audio)/sample_rate:.2f}s, sr={sample_rate}Hz)"
            )
            
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return None
    
    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr == target_sr:
            return audio
        
        # Calculate resampling ratio
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        
        # Resample using scipy
        resampled = signal.resample(audio, target_length)
        
        logger.debug(f"Audio resampled: {orig_sr}Hz → {target_sr}Hz")
        
        return resampled.astype(np.float32)
    
    @staticmethod
    def normalize_audio(
        audio: np.ndarray,
        target_level: float = -20.0
    ) -> np.ndarray:
        """
        Normalize audio amplitude.
        
        Args:
            audio: Audio array
            target_level: Target RMS level in dB
            
        Returns:
            Normalized audio array
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms == 0:
            return audio
        
        # Calculate target RMS from dB
        target_rms = 10 ** (target_level / 20)
        
        # Normalize
        normalized = audio * (target_rms / rms)
        
        # Clip to [-1, 1]
        normalized = np.clip(normalized, -1.0, 1.0)
        
        logger.debug(f"Audio normalized: RMS {rms:.4f} → {target_rms:.4f}")
        
        return normalized
    
    @staticmethod
    def preprocess_for_wav2vec(
        audio_path: Path
    ) -> Optional[torch.Tensor]:
        """
        Preprocess audio for Wav2Vec2 model.
        
        Wav2Vec2 expects:
        - 16kHz sample rate
        - Single channel (mono)
        - Float32 values in [-1, 1]
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor or None
        """
        # Load audio at 16kHz
        result = AudioUtils.load_audio(
            audio_path,
            target_sr=AudioUtils.TARGET_SAMPLE_RATE
        )
        
        if result is None:
            return None
        
        audio, sample_rate = result
        
        # Normalize
        audio = AudioUtils.normalize_audio(audio)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        logger.debug(f"Audio preprocessed for Wav2Vec2: {audio_tensor.shape}")
        
        return audio_tensor
    
    @staticmethod
    def extract_features(
        audio: np.ndarray,
        sample_rate: int
    ) -> dict:
        """
        Extract basic audio features.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            dict: Audio features
        """
        # Duration
        duration = len(audio) / sample_rate
        
        # Energy/amplitude features
        rms = np.sqrt(np.mean(audio ** 2))
        peak_amplitude = np.max(np.abs(audio))
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        zero_crossing_rate = zero_crossings / len(audio)
        
        # Spectral features (basic)
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)[:len(fft)//2]
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(fft)//2]
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        features = {
            'duration': float(duration),
            'sample_rate': sample_rate,
            'num_samples': len(audio),
            'rms': float(rms),
            'peak_amplitude': float(peak_amplitude),
            'zero_crossing_rate': float(zero_crossing_rate),
            'spectral_centroid': float(spectral_centroid)
        }
        
        return features
    
    @staticmethod
    def check_audio_quality(
        audio: np.ndarray,
        sample_rate: int
    ) -> dict:
        """
        Check audio quality and detect issues.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            dict: Quality metrics
        """
        features = AudioUtils.extract_features(audio, sample_rate)
        
        # Detect issues
        is_silent = features['rms'] < 0.001
        is_clipping = features['peak_amplitude'] > 0.99
        is_too_short = features['duration'] < 0.5
        is_low_quality = sample_rate < 8000
        
        quality = {
            **features,
            'is_silent': is_silent,
            'is_clipping': is_clipping,
            'is_too_short': is_too_short,
            'is_low_quality': is_low_quality,
            'has_issues': is_silent or is_clipping or is_too_short or is_low_quality
        }
        
        return quality
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        threshold: float = 0.01
    ) -> np.ndarray:
        """
        Trim leading and trailing silence.
        
        Args:
            audio: Audio array
            threshold: Silence threshold
            
        Returns:
            Trimmed audio array
        """
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            # All silent
            return audio
        
        # Find start and end indices
        indices = np.where(non_silent)[0]
        start_idx = indices[0]
        end_idx = indices[-1] + 1
        
        # Trim
        trimmed = audio[start_idx:end_idx]
        
        logger.debug(f"Audio trimmed: {len(audio)} → {len(trimmed)} samples")
        
        return trimmed
    
    @staticmethod
    def split_audio(
        audio: np.ndarray,
        sample_rate: int,
        chunk_duration: float = 30.0,
        overlap: float = 5.0
    ) -> list:
        """
        Split long audio into chunks.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            chunk_duration: Chunk duration in seconds
            overlap: Overlap duration in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunks = []
        start = 0
        
        while start < len(audio):
            end = start + chunk_samples
            chunk = audio[start:end]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            chunks.append(chunk)
            start += step_samples
        
        logger.debug(f"Audio split into {len(chunks)} chunks")
        
        return chunks
    
    @staticmethod
    def to_tensor(audio: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array to torch tensor.
        
        Args:
            audio: Audio array
            
        Returns:
            Audio tensor
        """
        return torch.from_numpy(audio).float()
    
    @staticmethod
    def to_numpy(audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array.
        
        Args:
            audio_tensor: Audio tensor
            
        Returns:
            Audio array
        """
        return audio_tensor.cpu().numpy()


# Convenience functions

def load_and_preprocess_for_wav2vec(audio_path: Path) -> Optional[torch.Tensor]:
    """Load and preprocess audio for Wav2Vec2."""
    return AudioUtils.preprocess_for_wav2vec(audio_path)


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds."""
    result = AudioUtils.load_audio(audio_path)
    if result is None:
        return None
    audio, sample_rate = result
    return len(audio) / sample_rate
