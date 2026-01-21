#!/usr/bin/env python3
"""
Corruption Reporting System - Dataset Download Script
Version: 1.0.0
Description: Download and prepare evaluation datasets


This script downloads:
- Real and Fake Faces Dataset (Kaggle - 140k images, ~300MB)
- CIFAKE Dataset (Real vs AI-Generated, ~1GB)
- Synthetic coordinated attack scenarios
- Creates necessary directory structure


Usage:
    # Download all datasets
    python download_datasets.py
    
    # Download specific dataset
    python download_datasets.py --dataset faceforensics
    
    # Download with sample limit
    python download_datasets.py --dataset faceforensics --samples 100
    
    # Dry run (check availability)
    python download_datasets.py --dry-run


Dependencies:
    pip install kaggle gdown requests tqdm pillow
"""


import argparse
import sys
import os
import json
import hashlib
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import urllib.request
import urllib.error


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Import optional dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not available")


try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Import dataset package
try:
    from evaluation.datasets import (
        DATASETS_DIR,
        get_dataset_path,
        get_dataset_info,
        list_datasets,
        is_dataset_downloaded,
        get_dataset_statistics,
        logger
    )
except ImportError as e:
    print(f"Error importing datasets package: {e}")
    sys.exit(1)


# ============================================
# CONFIGURATION
# ============================================


# Public datasets for deepfake detection (all under 10GB)
PUBLIC_DATASETS = {
    # 'faceforensics': {
    #     'name': 'Real and Fake Faces (Kaggle)',
    #     'source': 'kaggle', # you must pip install and do it manually
    #     'kaggle_dataset': 'ciplab/real-and-fake-face-detection',
    #     'size_mb': 300,
    #     'num_samples': 1200,  # 600 real + 600 fake (subset)
    #     'description': 'Real faces and StyleGAN-generated fake faces',
    #     'categories': ['real', 'fake']
    # },
    'celebdf': {
        'name': 'CIFAKE: Real and AI-Generated Images',
        'source': 'direct',
        'url': 'https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images',
        'size_mb': 1000,
        'num_samples': 1000,  # 500 real + 500 fake (subset)
        'description': 'CIFAR-10 real images and AI-generated counterparts',
        'categories': ['real', 'fake']
    },
    'synthetic_attacks': {
        'name': 'Synthetic Coordinated Attacks',
        'source': 'generated',
        'size_mb': 10,
        'num_samples': 100,
        'description': 'Generated coordinated attack scenarios for testing',
        'categories': ['coordinated', 'independent']
    }
}


# Direct download URLs (public, no auth required)
DIRECT_DOWNLOAD_URLS = {
    'sample_real_faces': 'https://github.com/NVlabs/ffhq-dataset/releases/download/v1.0/thumbnails128x128.zip',
}


# ============================================
# UTILITY FUNCTIONS
# ============================================


def download_file(url: str, destination: Path, show_progress: bool = True) -> bool:
    """
    Download file from URL with progress bar
    
    Args:
        url: Download URL
        destination: Destination file path
        show_progress: Show progress bar
        
    Returns:
        True if successful
    """
    try:
        if HAS_REQUESTS:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                if HAS_TQDM and show_progress:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            # Fallback to urllib
            urllib.request.urlretrieve(url, destination)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract zip or tar archive
    
    Args:
        archive_path: Path to archive file
        extract_to: Extraction directory
        
    Returns:
        True if successful
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def verify_file_hash(file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """
    Verify file integrity using hash
    
    Args:
        file_path: File to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm (sha256, md5)
        
    Returns:
        True if hash matches
    """
    try:
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        computed_hash = hasher.hexdigest()
        return computed_hash.lower() == expected_hash.lower()
        
    except Exception as e:
        logger.error(f"Failed to verify hash: {e}")
        return False


# ============================================
# DOWNLOADER CLASS
# ============================================


class DatasetDownloader:
    """Dataset download manager"""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize downloader
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Download statistics
        self.stats = {
            'downloaded': [],
            'skipped': [],
            'failed': []
        }
        
        # Ensure base directories exist
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # MAIN DOWNLOAD METHODS
    # ========================================
    
    def download_all(self, sample_size: Optional[int] = None, force: bool = False) -> Dict[str, bool]:
        """
        Download all datasets
        
        Args:
            sample_size: Number of samples per dataset
            force: Force re-download
            
        Returns:
            Dictionary mapping dataset_name -> success
        """
        results = {}
        
        for dataset_name in PUBLIC_DATASETS.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            success = self.download_dataset(dataset_name, sample_size, force)
            results[dataset_name] = success
        
        return results
    
    def download_dataset(
        self,
        dataset_name: str,
        sample_size: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        Download a dataset
        
        Args:
            dataset_name: Name of dataset
            sample_size: Number of samples to download (None = all)
            force: Force re-download
            
        Returns:
            True if successful
        """
        logger.info(f"Downloading {dataset_name}...")
        
        # Check if already downloaded
        if is_dataset_downloaded(dataset_name) and not force:
            logger.info(f"Dataset {dataset_name} already downloaded (use --force to re-download)")
            self.stats['skipped'].append(dataset_name)
            return True
        
        # Get dataset config
        if dataset_name not in PUBLIC_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            self.stats['failed'].append(dataset_name)
            return False
        
        dataset_config = PUBLIC_DATASETS[dataset_name]
        
        # Handle different sources
        source = dataset_config['source']
        
        if source == 'kaggle':
            success = self._download_from_kaggle(dataset_name, dataset_config, sample_size)
        elif source == 'direct':
            success = self._download_direct(dataset_name, dataset_config, sample_size)
        elif source == 'generated':
            success = self._generate_synthetic_dataset(dataset_name, dataset_config, sample_size)
        else:
            logger.error(f"Unsupported source: {source}")
            success = False
        
        if success:
            self._mark_downloaded(dataset_name)
            self.stats['downloaded'].append(dataset_name)
            logger.info(f" Successfully downloaded {dataset_name}")
        else:
            self.stats['failed'].append(dataset_name)
            logger.error(f" Failed to download {dataset_name}")
        
        return success
    
    # ========================================
    # KAGGLE DOWNLOAD
    # ========================================
    
    def _download_from_kaggle(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        sample_size: Optional[int]
    ) -> bool:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_size: Number of samples
            
        Returns:
            True if successful
        """
        try:
            import kaggle
            HAS_KAGGLE = True
        except ImportError:
            HAS_KAGGLE = False
            logger.error("Kaggle API not available. Install with: pip install kaggle")
            logger.info("Also configure Kaggle API: https://www.kaggle.com/docs/api")
            return self._download_kaggle_fallback(dataset_name, config, sample_size)
        
        try:
            kaggle_dataset = config['kaggle_dataset']
            dataset_path = get_dataset_path(dataset_name)
            
            logger.info(f"Downloading from Kaggle: {kaggle_dataset}")
            
            # # Download dataset
            # kaggle.api.dataset_download_files(
            #     kaggle_dataset,
            #     path=dataset_path,
            #     unzip=True,
            #     quiet=False
            # )
            
            # Organize files
            success = self._organize_kaggle_files(dataset_name, dataset_path, sample_size)
            
            return success
            
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
            logger.info("Trying fallback method...")
            return self._download_kaggle_fallback(dataset_name, config, sample_size)
    
    def _download_kaggle_fallback(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        sample_size: Optional[int]
    ) -> bool:
        """
        Fallback: Manual Kaggle dataset instructions
        
        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_size: Number of samples
            
        Returns:
            True if user confirms manual download
        """
        logger.info("="*60)
        logger.info(f"Manual Download Instructions: {config['name']}")
        logger.info("="*60)
        logger.info("")
        logger.info("Kaggle API is not configured. Please follow these steps:")
        logger.info("")
        logger.info("1. Install Kaggle API: pip install kaggle")
        logger.info("2. Create Kaggle account: https://www.kaggle.com/")
        logger.info("3. Get API credentials:")
        logger.info("   - Go to: https://www.kaggle.com/settings")
        logger.info("   - Click 'Create New API Token'")
        logger.info("   - Place kaggle.json in: ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        logger.info("")
        logger.info(f"4. Download dataset: {config['kaggle_dataset']}")
        logger.info(f"   URL: https://www.kaggle.com/datasets/{config['kaggle_dataset']}")
        logger.info("")
        logger.info(f"5. Extract to: {get_dataset_path(dataset_name)}")
        logger.info("")
        logger.info("="*60)
        
        # Create placeholder structure
        dataset_path = get_dataset_path(dataset_name)
        for category in config.get('categories', ['real', 'fake']):
            (dataset_path / category).mkdir(parents=True, exist_ok=True)
        
        # Create README
        readme_path = dataset_path / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"Dataset: {config['name']}\n")
            f.write(f"Source: Kaggle - {config['kaggle_dataset']}\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"\nPlease download manually and extract here.\n")
        
        return False
    
    def _organize_kaggle_files(
        self,
        dataset_name: str,
        dataset_path: Path,
        sample_size: Optional[int]
    ) -> bool:
        """
        Organize Kaggle dataset files into expected structure
        
        Args:
            dataset_name: Dataset identifier
            dataset_path: Dataset directory
            sample_size: Number of samples to keep
            
        Returns:
            True if successful
        """
        try:
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            all_files = []
            
            for ext in image_extensions:
                all_files.extend(dataset_path.rglob(f'*{ext}'))
                all_files.extend(dataset_path.rglob(f'*{ext.upper()}'))
            
            logger.info(f"Found {len(all_files)} images")
            
            # Organize into real/fake categories
            real_dir = dataset_path / 'real'
            fake_dir = dataset_path / 'fake'
            real_dir.mkdir(exist_ok=True)
            fake_dir.mkdir(exist_ok=True)
            
            real_count = 0
            fake_count = 0
            max_per_category = sample_size // 2 if sample_size else 10000
            
            for file_path in all_files:
                # Determine if real or fake based on path/name
                file_str = str(file_path).lower()
                
                is_fake = any(keyword in file_str for keyword in ['fake', 'generated', 'synthetic', 'gan', 'deepfake'])
                
                if is_fake and fake_count < max_per_category:
                    dest = fake_dir / f"fake_{fake_count:05d}{file_path.suffix}"
                    shutil.copy2(file_path, dest)
                    fake_count += 1
                elif not is_fake and real_count < max_per_category:
                    dest = real_dir / f"real_{real_count:05d}{file_path.suffix}"
                    shutil.copy2(file_path, dest)
                    real_count += 1
                
                if real_count >= max_per_category and fake_count >= max_per_category:
                    break
            
            logger.info(f"Organized: {real_count} real, {fake_count} fake images")
            
            # Create metadata
            self._create_metadata(dataset_name, dataset_path, real_count, fake_count)
            
            return real_count > 0 or fake_count > 0
            
        except Exception as e:
            logger.error(f"Failed to organize files: {e}")
            return False
    
    # ========================================
    # DIRECT DOWNLOAD
    # ========================================
    
    def _download_direct(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        sample_size: Optional[int]
    ) -> bool:
        """
        Download dataset from direct URL
        
        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_size: Number of samples
            
        Returns:
            True if successful
        """
        try:
            url = config['url']
            dataset_path = get_dataset_path(dataset_name)
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Download archive
            archive_name = url.split('/')[-1]
            if '?' in archive_name:
                archive_name = archive_name.split('?')[0]
            
            if not archive_name.endswith(('.zip', '.tar', '.gz', '.tgz')):
                archive_name += '.zip'
            
            archive_path = dataset_path / archive_name
            
            logger.info(f"Downloading from: {url}")
            success = download_file(url, archive_path, show_progress=True)
            
            if not success:
                logger.warning("Direct download failed. Using alternative method...")
                return self._download_alternative_dataset(dataset_name, config, sample_size)
            
            # Extract
            logger.info("Extracting archive...")
            extract_success = extract_archive(archive_path, dataset_path)
            
            if extract_success:
                # Organize files
                self._organize_kaggle_files(dataset_name, dataset_path, sample_size)
                
                # Clean up archive
                archive_path.unlink()
            
            return extract_success
            
        except Exception as e:
            logger.error(f"Direct download failed: {e}")
            return self._download_alternative_dataset(dataset_name, config, sample_size)
    
    def _download_alternative_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        sample_size: Optional[int]
    ) -> bool:
        """
        Alternative: Use public sample dataset
        
        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_size: Number of samples
            
        Returns:
            True if successful
        """
        logger.info(f"Using alternative sample dataset for {dataset_name}")
        
        dataset_path = get_dataset_path(dataset_name)
        
        # Create structure with sample instructions
        for category in config.get('categories', ['real', 'fake']):
            (dataset_path / category).mkdir(parents=True, exist_ok=True)
        
        # Create README with alternative sources
        readme_path = dataset_path / 'README.txt'
        with open(readme_path, 'w') as f:
            f.write(f"Dataset: {config['name']}\n")
            f.write(f"Description: {config['description']}\n\n")
            f.write("Alternative public datasets for deepfake detection:\n\n")
            f.write("1. Real and Fake Face Detection (Kaggle)\n")
            f.write("   https://www.kaggle.com/ciplab/real-and-fake-face-detection\n")
            f.write("   Size: ~300MB, 1200 images\n\n")
            f.write("2. 140k Real and Fake Faces (Kaggle)\n")
            f.write("   https://www.kaggle.com/xhlulu/140k-real-and-fake-faces\n")
            f.write("   Size: ~400MB, subset recommended\n\n")
            f.write("3. DeeperForensics-1.0 (Academic)\n")
            f.write("   https://github.com/EndlessSora/DeeperForensics-1.0\n")
            f.write("   Size: Variable, requires request\n\n")
            f.write(f"Please download and extract to: {dataset_path}\n")
            f.write("Organize as: real/ and fake/ subdirectories\n")
        
        logger.info(f"Created dataset structure at: {dataset_path}")
        logger.info(f"See {readme_path} for download instructions")
        
        return False
    
    # ========================================
    # SYNTHETIC DATASET GENERATION
    # ========================================
    
    def _generate_synthetic_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        sample_size: Optional[int]
    ) -> bool:
        """
        Generate synthetic dataset (for coordination detection)
        
        Args:
            dataset_name: Dataset identifier
            config: Dataset configuration
            sample_size: Number of samples
            
        Returns:
            True if successful
        """
        try:
            from evaluation.datasets.generate_synthetic import SyntheticAttackGenerator
            
            logger.info(f"Generating synthetic dataset: {dataset_name}")
            
            dataset_path = get_dataset_path(dataset_name)
            num_samples = sample_size or config['num_samples']
            
            generator = SyntheticAttackGenerator()
            scenarios = generator.generate_attacks(
                num_groups=num_samples,
                # output_dir=dataset_path
            )
            
            logger.info(f"Generated {len(scenarios)} synthetic scenarios")
            
            # Create metadata
            metadata = {
                'name': config['name'],
                'description': config['description'],
                'num_scenarios': len(scenarios),
                'categories': config['categories'],
                'generated_at': datetime.now().isoformat()
            }
            
            metadata_path = dataset_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            return False
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def _create_metadata(
        self,
        dataset_name: str,
        dataset_path: Path,
        real_count: int,
        fake_count: int
    ) -> None:
        """Create dataset metadata file"""
        metadata = {
            'name': dataset_name,
            'description': PUBLIC_DATASETS[dataset_name]['description'],
            'num_real': real_count,
            'num_fake': fake_count,
            'total_samples': real_count + fake_count,
            'categories': ['real', 'fake'],
            'downloaded_at': datetime.now().isoformat()
        }
        
        metadata_path = dataset_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _mark_downloaded(self, dataset_name: str) -> None:
        """Mark dataset as downloaded"""
        dataset_path = get_dataset_path(dataset_name)
        marker_file = dataset_path / '.downloaded'
        
        with open(marker_file, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def print_statistics(self) -> None:
        """Print download statistics"""
        logger.info("\n" + "="*60)
        logger.info("Download Statistics")
        logger.info("="*60)
        logger.info(f"Downloaded: {len(self.stats['downloaded'])} datasets")
        for ds in self.stats['downloaded']:
            logger.info(f"   {ds}")
        
        if self.stats['skipped']:
            logger.info(f"\nSkipped: {len(self.stats['skipped'])} datasets")
            for ds in self.stats['skipped']:
                logger.info(f"  - {ds}")
        
        if self.stats['failed']:
            logger.info(f"\nFailed: {len(self.stats['failed'])} datasets")
            for ds in self.stats['failed']:
                logger.info(f"   {ds}")
        
        logger.info("="*60)


# ============================================
# COMMAND-LINE INTERFACE
# ============================================


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Download evaluation datasets for corruption reporting system',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(PUBLIC_DATASETS.keys()) + ['all'],
        default='all',
        help='Dataset to download (default: all)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=1000,
        help='Number of samples to download per dataset (default: 1000)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without downloading'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\nAvailable Datasets:")
        print("="*60)
        for name, config in PUBLIC_DATASETS.items():
            status = "" if is_dataset_downloaded(name) else ""
            print(f"\n{status} {name}")
            print(f"  Name: {config['name']}")
            print(f"  Description: {config['description']}")
            print(f"  Size: ~{config['size_mb']}MB")
            print(f"  Samples: ~{config['num_samples']}")
        print("\n" + "="*60)
        return 0
    
    # Dry run
    if args.dry_run:
        print("\nDry Run - Would download:")
        print("="*60)
        datasets_to_download = [args.dataset] if args.dataset != 'all' else list(PUBLIC_DATASETS.keys())
        for name in datasets_to_download:
            config = PUBLIC_DATASETS[name]
            print(f"\n{name}:")
            print(f"  {config['name']}")
            print(f"  Size: ~{config['size_mb']}MB")
            print(f"  Samples: {args.samples}")
        print("\n" + "="*60)
        return 0
    
    # Initialize downloader
    downloader = DatasetDownloader(verbose=args.verbose)
    
    # Download datasets
    if args.dataset == 'all':
        logger.info("Downloading all datasets...")
        downloader.download_all(sample_size=args.samples, force=args.force)
    else:
        logger.info(f"Downloading dataset: {args.dataset}")
        downloader.download_dataset(args.dataset, sample_size=args.samples, force=args.force)
    
    # Print statistics
    downloader.print_statistics()
    
    return 0 if not downloader.stats['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
