"""
Corruption Reporting System - Dataset Loader
Version: 1.0.0
Description: Load and preprocess evaluation datasets

This module provides:
- Dataset loading from disk
- Data preprocessing and normalization
- Train/test/validation splits
- Batch generation
- Data augmentation

Supported Datasets:
- FaceForensics++ (images)
- Celeb-DF (videos → frames)
- Synthetic coordinated attacks

Usage:
    from evaluation.datasets import DatasetLoader
    
    # Load dataset
    loader = DatasetLoader('faceforensics')
    dataset = loader.load(split='test', max_samples=100)
    
    # Iterate batches
    for batch in loader.batch_generator(batch_size=32):
        images, labels = batch
        # Process batch
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Generator, Union
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import dataset package
try:
    from evaluation.datasets import (
        get_dataset_path,
        get_dataset_info,
        is_dataset_downloaded,
        validate_dataset,
        logger
    )
except ImportError as e:
    print(f"Error importing datasets package: {e}")
    sys.exit(1)

# ============================================
# CONSTANTS
# ============================================

# Supported file extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
TEXT_EXTENSIONS = ['.txt', '.json']

# Default splits
DEFAULT_SPLITS = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

# ============================================
# DATASET LOADER CLASS
# ============================================

class DatasetLoader:
    """Dataset loader for evaluation experiments"""
    
    def __init__(
        self,
        dataset_name: str,
        seed: int = 42,
        verbose: bool = False
    ):
        """
        Initialize dataset loader
        
        Args:
            dataset_name: Name of dataset to load
            seed: Random seed for reproducibility
            verbose: Enable verbose logging
            
        Raises:
            ValueError: If dataset not found or not downloaded
        """
        self.dataset_name = dataset_name
        self.seed = seed
        self.verbose = verbose
        
        # Setup logging
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Validate dataset
        if not is_dataset_downloaded(dataset_name):
            raise ValueError(
                f"Dataset {dataset_name} not downloaded. "
                f"Run download_datasets.py first."
            )
        
        # Get dataset info
        self.dataset_info = get_dataset_info(dataset_name)
        self.dataset_path = get_dataset_path(dataset_name)
        
        # Set random seed
        random.seed(seed)
        
        # Cache
        self._file_index = None
        self._splits = None
        
        logger.info(f"Initialized loader for {dataset_name}")
    
    # ========================================
    # MAIN LOADING METHODS
    # ========================================
    
    def load(
        self,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True
    ) -> Dict[str, Any]:
        """
        Load dataset
        
        Args:
            split: Data split ('train', 'val', 'test', None=all)
            max_samples: Maximum number of samples to load
            shuffle: Shuffle samples
            
        Returns:
            Dataset dictionary with samples and metadata
        """
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        # Build file index
        if self._file_index is None:
            self._build_file_index()
        
        # Get samples for split
        if split:
            samples = self._get_split_samples(split)
        else:
            samples = self._file_index['all']
        
        # Shuffle
        if shuffle:
            random.shuffle(samples)
        
        # Limit samples
        if max_samples:
            samples = samples[:max_samples]
        
        # Load dataset
        dataset = {
            'name': self.dataset_name,
            'split': split,
            'num_samples': len(samples),
            'samples': samples,
            'metadata': self.dataset_info
        }
        
        logger.info(f"Loaded {len(samples)} samples")
        
        return dataset
    
    def load_sample(self, sample_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load individual sample
        
        Args:
            sample_path: Path to sample file
            
        Returns:
            Sample data dictionary
        """
        sample_path = Path(sample_path)
        
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample not found: {sample_path}")
        
        # Determine sample type
        extension = sample_path.suffix.lower()
        
        if extension in IMAGE_EXTENSIONS:
            return self._load_image_sample(sample_path)
        elif extension in VIDEO_EXTENSIONS:
            return self._load_video_sample(sample_path)
        elif extension in TEXT_EXTENSIONS:
            return self._load_text_sample(sample_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    # ========================================
    # BATCH GENERATION
    # ========================================
    
    def batch_generator(
        self,
        batch_size: int = 32,
        split: Optional[str] = None,
        shuffle: bool = True
    ) -> Generator[Tuple[List[Dict], List[Any]], None, None]:
        """
        Generate batches of samples
        
        Args:
            batch_size: Number of samples per batch
            split: Data split to use
            shuffle: Shuffle samples
            
        Yields:
            Tuple of (samples, labels)
        """
        # Load dataset
        dataset = self.load(split=split, shuffle=shuffle)
        samples = dataset['samples']
        
        # Generate batches
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # Load batch data
            batch_data = []
            batch_labels = []
            
            for sample in batch_samples:
                try:
                    sample_data = self.load_sample(sample['path'])
                    batch_data.append(sample_data)
                    batch_labels.append(sample.get('label'))
                except Exception as e:
                    logger.warning(f"Error loading sample {sample['path']}: {e}")
                    continue
            
            if batch_data:
                yield batch_data, batch_labels
    
    # ========================================
    # FILE INDEX BUILDING
    # ========================================
    
    def _build_file_index(self):
        """Build index of dataset files"""
        logger.info("Building file index...")
        
        if self.dataset_name == 'faceforensics':
            self._file_index = self._index_faceforensics()
        elif self.dataset_name == 'celebdf':
            self._file_index = self._index_celebdf()
        elif self.dataset_name == 'real-and-fake-face-detection':
            self._file_index = self._index_real_and_fake_face_detection()
        elif self.dataset_name == 'synthetic_attacks':
            self._file_index = self._index_synthetic_attacks()
        else:
            # Generic indexing
            self._file_index = self._index_generic()
        
        logger.info(f"Indexed {len(self._file_index['all'])} files")
    
    def _index_faceforensics(self) -> Dict[str, List[Dict]]:
        """Index FaceForensics++ dataset"""
        index = {'all': [], 'real': [], 'fake': []}
        
        # Real images
        real_dir = self.dataset_path / 'real'
        if real_dir.exists():
            for img_path in real_dir.glob('*'):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    sample = {
                        'path': img_path,
                        'label': 'real',
                        'label_id': 0,
                        'type': 'image'
                    }
                    index['all'].append(sample)
                    index['real'].append(sample)
        
        # Fake images
        fake_dir = self.dataset_path / 'fake'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*'):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    sample = {
                        'path': img_path,
                        'label': 'fake',
                        'label_id': 1,
                        'type': 'image'
                    }
                    index['all'].append(sample)
                    index['fake'].append(sample)
        
        return index
    
    def _index_celebdf(self) -> Dict[str, List[Dict]]:
        """Index Celeb-DF dataset"""
        index = {'all': [], 'real': [], 'fake': []}
        
        # Real vids
        real_dirs = ['Celeb-real', 'YouTube-real', 'real']
        for dir_name in real_dirs:
            real_dir = self.dataset_path / dir_name
            if real_dir.exists():
                for video_path in real_dir.glob('*'):
                    if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                        sample = {
                            'path': video_path,
                            'label': 'real',
                            'label_id': 0,
                            'type': 'video',
                            'source': dir_name
                        }
                        index['all'].append(sample)
                        index['real'].append(sample)
                    elif video_path.suffix.lower() in IMAGE_EXTENSIONS:
                        sample = {
                            'path': video_path,
                            'label': 'real',
                            'label_id': 0,
                            'type': 'image',
                            'source': dir_name
                        }
                        index['all'].append(sample)
                        index['real'].append(sample)
        
        # Fake vids
        fake_dirs = ['Celeb-synthesis', 'fake']
        for dir_name in fake_dirs:
            fake_dir = self.dataset_path / dir_name
            if fake_dir.exists():
                for video_path in fake_dir.glob('*'):
                    if video_path.suffix.lower() in VIDEO_EXTENSIONS:
                        sample = {
                            'path': video_path,
                            'label': 'fake',
                            'label_id': 1,
                            'type': 'video',
                            'source': dir_name
                        }
                        index['all'].append(sample)
                        index['fake'].append(sample)
                    elif video_path.suffix.lower() in IMAGE_EXTENSIONS:
                        sample = {
                            'path': video_path,
                            'label': 'fake',
                            'label_id': 1,
                            'type': 'image',
                            'source': dir_name
                        }
                        index['all'].append(sample)
                        index['fake'].append(sample)
        
        return index
    
    def _index_real_and_fake_face_detection(self) -> Dict[str, List[Dict]]:
        """Index Real and Fake Face dataset"""
        index = {'all': [], 'real': [], 'fake': []}
        
        # Determine directory structure
        # Check if nested 'real_and_fake_face' exists
        nested_dir = self.dataset_path / 'real_and_fake_face'
        base_dir = nested_dir if nested_dir.exists() else self.dataset_path

        # Real images
        real_dirs = ['training_real', 'real']
        for dir_name in real_dirs:
            real_dir = base_dir / dir_name
            if real_dir.exists():
                for img_path in real_dir.glob('*'):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        sample = {
                            'path': img_path,
                            'label': 'real',
                            'label_id': 0,
                            'type': 'image'
                        }
                        index['all'].append(sample)
                        index['real'].append(sample)
                        
        # Fake images
        fake_dirs = ['training_fake', 'fake', 'easy', 'hard']
        for dir_name in fake_dirs:
            fake_dir = base_dir / dir_name
            if fake_dir.exists():
                for img_path in fake_dir.glob('*'):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        sample = {
                            'path': img_path,
                            'label': 'fake',
                            'label_id': 1,
                            'type': 'image'
                        }
                        index['all'].append(sample)
                        index['fake'].append(sample)
                        
        return index
    
    def _index_synthetic_attacks(self) -> Dict[str, List[Dict]]:
        """Index synthetic attacks dataset"""
        index = {'all': []}
        
        # Load synthetic attacks JSON
        json_path = self.dataset_path / 'synthetic_attacks.json'
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                attacks = json.load(f)
            
            for attack in attacks:
                sample = {
                    'path': json_path,  # All in one file
                    'data': attack,
                    'label': 'coordinated_attack',
                    'type': 'attack_scenario',
                    'attack_id': attack.get('id')
                }
                index['all'].append(sample)
        
        return index
    
    def _index_generic(self) -> Dict[str, List[Dict]]:
        """Generic file indexing"""
        index = {'all': []}
        
        # Recursively find all files
        for file_path in self.dataset_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in (IMAGE_EXTENSIONS + VIDEO_EXTENSIONS + TEXT_EXTENSIONS):
                sample = {
                    'path': file_path,
                    'type': self._get_file_type(file_path),
                    'label': None
                }
                index['all'].append(sample)
        
        return index
    
    # ========================================
    # SPLIT MANAGEMENT
    # ========================================
    
    def _get_split_samples(self, split: str) -> List[Dict]:
        """Get samples for a data split"""
        if self._splits is None:
            self._create_splits()
        
        if split not in self._splits:
            raise ValueError(f"Unknown split: {split}")
        
        return self._splits[split]
    
    def _create_splits(self):
        """Create train/val/test splits"""
        all_samples = self._file_index['all'].copy()
        random.shuffle(all_samples)
        
        n = len(all_samples)
        train_end = int(n * DEFAULT_SPLITS['train'])
        val_end = train_end + int(n * DEFAULT_SPLITS['val'])
        
        self._splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end],
            'test': all_samples[val_end:]
        }
        
        logger.info(f"Created splits - train: {len(self._splits['train'])}, "
                   f"val: {len(self._splits['val'])}, test: {len(self._splits['test'])}")
    
    # ========================================
    # SAMPLE LOADING
    # ========================================
    
    def _load_image_sample(self, image_path: Path) -> Dict[str, Any]:
        """Load image sample"""
        try:
            # Lazy import to avoid dependency if not needed
            from PIL import Image
            
            # Load image
            image = Image.open(image_path)
            
            return {
                'path': str(image_path),
                'type': 'image',
                'format': image.format,
                'size': image.size,
                'mode': image.mode,
                'data': image  # PIL Image object
            }
        except ImportError:
            logger.warning("PIL not available, returning path only")
            return {
                'path': str(image_path),
                'type': 'image',
                'data': None
            }
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def _load_video_sample(self, video_path: Path) -> Dict[str, Any]:
        """Load video sample (metadata only, no frames)"""
        # Return video metadata
        # For actual frame extraction, would need opencv-python
        return {
            'path': str(video_path),
            'type': 'video',
            'size_bytes': video_path.stat().st_size,
            'note': 'Use opencv-python for frame extraction'
        }
    
    def _load_text_sample(self, text_path: Path) -> Dict[str, Any]:
        """Load text sample"""
        if text_path.suffix == '.json':
            with open(text_path, 'r') as f:
                data = json.load(f)
            return {
                'path': str(text_path),
                'type': 'json',
                'data': data
            }
        else:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return {
                'path': str(text_path),
                'type': 'text',
                'data': text
            }
    
    # ========================================
    # UTILITIES
    # ========================================
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return 'image'
        elif ext in VIDEO_EXTENSIONS:
            return 'video'
        elif ext in TEXT_EXTENSIONS:
            return 'text'
        else:
            return 'unknown'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if self._file_index is None:
            self._build_file_index()
        
        stats = {
            'total_samples': len(self._file_index['all']),
            'splits': {}
        }
        
        # Count by label
        if 'real' in self._file_index:
            stats['real_samples'] = len(self._file_index['real'])
        if 'fake' in self._file_index:
            stats['fake_samples'] = len(self._file_index['fake'])
        
        # Split statistics
        if self._splits:
            for split_name, samples in self._splits.items():
                stats['splits'][split_name] = len(samples)
        
        return stats

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def load_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Convenience function to load dataset
    
    Args:
        dataset_name: Name of dataset
        split: Data split
        max_samples: Maximum samples
        seed: Random seed
        
    Returns:
        Dataset dictionary
    """
    loader = DatasetLoader(dataset_name, seed=seed)
    return loader.load(split=split, max_samples=max_samples)

def get_loader(dataset_name: str, **kwargs) -> DatasetLoader:
    """
    Get dataset loader instance
    
    Args:
        dataset_name: Name of dataset
        **kwargs: Additional loader arguments
        
    Returns:
        DatasetLoader instance
    """
    return DatasetLoader(dataset_name, **kwargs)

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'DatasetLoader',
    'load_dataset',
    'get_loader',
    'IMAGE_EXTENSIONS',
    'VIDEO_EXTENSIONS',
    'TEXT_EXTENSIONS',
    'DEFAULT_SPLITS'
]
