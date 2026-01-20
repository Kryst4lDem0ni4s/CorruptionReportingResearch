#!/usr/bin/env python3
"""
Corruption Reporting System - Model Download Script
Version: 1.0.0
Description: Pre-download ML models for offline usage

This script downloads all required pre-trained models:
- openai/clip-vit-base-patch32 (~350MB)
- sentence-transformers/all-MiniLM-L6-v2 (~80MB)
- facebook/wav2vec2-base (~360MB) - Optional
- Salesforce/blip-image-captioning-base (~1GB) - Optional

Usage:
    python scripts/download_models.py [--all] [--cache-dir PATH]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

REQUIRED_MODELS = {
    'clip': {
        'name': 'openai/clip-vit-base-patch32',
        'description': 'CLIP image classifier for deepfake detection',
        'size': '~350MB',
        'required': True,
        'library': 'transformers'
    },
    'sentence_transformer': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'description': 'Sentence embeddings for text analysis',
        'size': '~80MB',
        'required': True,
        'library': 'sentence-transformers'
    }
}

OPTIONAL_MODELS = {
    'wav2vec': {
        'name': 'facebook/wav2vec2-base',
        'description': 'Audio feature extraction',
        'size': '~360MB',
        'required': False,
        'library': 'transformers'
    },
    'blip': {
        'name': 'Salesforce/blip-image-captioning-base',
        'description': 'Image captioning',
        'size': '~1GB',
        'required': False,
        'library': 'transformers'
    }
}

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def check_dependencies():
    """Check if required libraries are installed"""
    try:
        import torch
        import transformers
        from sentence_transformers import SentenceTransformer
        logger.info(" All required libraries are installed")
        return True
    except ImportError as e:
        logger.error(f" Missing dependency: {e}")
        logger.error("Please install requirements: pip install -r requirements.txt")
        return False

def download_clip_model(cache_dir: Optional[str] = None) -> bool:
    """
    Download CLIP model
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("Downloading CLIP Model")
        logger.info("=" * 70)
        logger.info(f"Model: {REQUIRED_MODELS['clip']['name']}")
        logger.info(f"Size: {REQUIRED_MODELS['clip']['size']}")
        logger.info("Purpose: Zero-shot deepfake detection")
        logger.info("")
        
        from transformers import CLIPProcessor, CLIPModel
        
        model_name = REQUIRED_MODELS['clip']['name']
        
        logger.info("Downloading processor...")
        processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("Downloading model...")
        model = CLIPModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info(" CLIP model downloaded successfully")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f" Failed to download CLIP model: {e}")
        return False

def download_sentence_transformer_model(cache_dir: Optional[str] = None) -> bool:
    """
    Download Sentence Transformer model
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("Downloading Sentence Transformer Model")
        logger.info("=" * 70)
        logger.info(f"Model: {REQUIRED_MODELS['sentence_transformer']['name']}")
        logger.info(f"Size: {REQUIRED_MODELS['sentence_transformer']['size']}")
        logger.info("Purpose: Text embeddings for coordination detection")
        logger.info("")
        
        from sentence_transformers import SentenceTransformer
        
        model_name = REQUIRED_MODELS['sentence_transformer']['name']
        
        logger.info("Downloading model...")
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir
        )
        
        logger.info(" Sentence Transformer model downloaded successfully")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f" Failed to download Sentence Transformer model: {e}")
        return False

def download_wav2vec_model(cache_dir: Optional[str] = None) -> bool:
    """
    Download Wav2Vec2 model (optional)
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("Downloading Wav2Vec2 Model (Optional)")
        logger.info("=" * 70)
        logger.info(f"Model: {OPTIONAL_MODELS['wav2vec']['name']}")
        logger.info(f"Size: {OPTIONAL_MODELS['wav2vec']['size']}")
        logger.info("Purpose: Audio feature extraction")
        logger.info("")
        
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        
        model_name = OPTIONAL_MODELS['wav2vec']['name']
        
        logger.info("Downloading processor...")
        processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("Downloading model...")
        model = Wav2Vec2Model.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info(" Wav2Vec2 model downloaded successfully")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f" Failed to download Wav2Vec2 model: {e}")
        return False

def download_blip_model(cache_dir: Optional[str] = None) -> bool:
    """
    Download BLIP model (optional)
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        True if successful
    """
    try:
        logger.info("=" * 70)
        logger.info("Downloading BLIP Model (Optional)")
        logger.info("=" * 70)
        logger.info(f"Model: {OPTIONAL_MODELS['blip']['name']}")
        logger.info(f"Size: {OPTIONAL_MODELS['blip']['size']}")
        logger.info("Purpose: Image captioning")
        logger.info("")
        
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        model_name = OPTIONAL_MODELS['blip']['name']
        
        logger.info("Downloading processor...")
        processor = BlipProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info("Downloading model...")
        model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        logger.info(" BLIP model downloaded successfully")
        logger.info("")
        
        return True
    
    except Exception as e:
        logger.error(f" Failed to download BLIP model: {e}")
        return False

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_models(cache_dir: Optional[str] = None) -> bool:
    """
    Verify that models can be loaded
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        True if all required models load successfully
    """
    logger.info("=" * 70)
    logger.info("Verifying Downloaded Models")
    logger.info("=" * 70)
    
    all_ok = True
    
    # Verify CLIP
    try:
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(
            REQUIRED_MODELS['clip']['name'],
            cache_dir=cache_dir
        )
        logger.info(" CLIP model loads successfully")
    except Exception as e:
        logger.error(f" CLIP model verification failed: {e}")
        all_ok = False
    
    # Verify Sentence Transformer
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            REQUIRED_MODELS['sentence_transformer']['name'],
            cache_folder=cache_dir
        )
        logger.info(" Sentence Transformer model loads successfully")
    except Exception as e:
        logger.error(f" Sentence Transformer model verification failed: {e}")
        all_ok = False
    
    logger.info("")
    
    if all_ok:
        logger.info(" All required models verified successfully")
    else:
        logger.error(" Some models failed verification")
    
    return all_ok

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Download ML models for Corruption Reporting System'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all models including optional ones'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Custom cache directory for models'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing models without downloading'
    )
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip model verification after download'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Model Download")
    print("=" * 70)
    print("")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Cannot proceed without required dependencies")
        return 1
    
    # Verify only mode
    if args.verify_only:
        success = verify_models(args.cache_dir)
        return 0 if success else 1
    
    # Set cache directory
    cache_dir = args.cache_dir
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache directory: {cache_dir}")
        logger.info("")
    else:
        logger.info("Using default cache directory (~/.cache/huggingface)")
        logger.info("")
    
    # Calculate total size
    total_size = "~430MB"
    if args.all:
        total_size = "~1.8GB"
    
    logger.info(f"Total download size: {total_size}")
    logger.info("This may take several minutes depending on your internet connection.")
    logger.info("")
    
    # Confirm download
    try:
        response = input("Continue with download? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            logger.info("Download cancelled by user")
            return 0
    except (KeyboardInterrupt, EOFError):
        logger.info("\nDownload cancelled by user")
        return 0
    
    print("")
    
    # Download required models
    results = []
    
    logger.info("Downloading required models...")
    logger.info("")
    
    results.append(("CLIP", download_clip_model(cache_dir)))
    results.append(("Sentence Transformer", download_sentence_transformer_model(cache_dir)))
    
    # Download optional models if requested
    if args.all:
        logger.info("Downloading optional models...")
        logger.info("")
        
        results.append(("Wav2Vec2", download_wav2vec_model(cache_dir)))
        results.append(("BLIP", download_blip_model(cache_dir)))
    
    # Verify models unless skipped
    if not args.skip_verification:
        verification_success = verify_models(cache_dir)
        results.append(("Verification", verification_success))
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Download Summary")
    logger.info("=" * 70)
    
    for model_name, success in results:
        status = " SUCCESS" if success else " FAILED"
        logger.info(f"{model_name:30s}: {status}")
    
    logger.info("")
    
    # Check overall success
    all_required_ok = all(success for name, success in results if name in ["CLIP", "Sentence Transformer"])
    
    if all_required_ok:
        logger.info(" All required models downloaded successfully!")
        logger.info("")
        logger.info("Models are ready to use. The system will automatically load them when needed.")
        return 0
    else:
        logger.error(" Some required models failed to download")
        logger.error("Please check your internet connection and try again")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
