"""
Image Utils - Image preprocessing and manipulation

Provides:
- Image loading and resizing for ML models
- Normalization for CLIP, BLIP models
- Test-time augmentation
- Image quality checks
- Format conversion
- Batch preprocessing
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image, ImageEnhance, ImageOps

# Initialize logger
logger = logging.getLogger(__name__)


class ImageUtils:
    """
    Image utilities for ML preprocessing.
    
    Features:
    - Image loading and validation
    - Resizing and normalization for CLIP/BLIP
    - Test-time augmentation (TTA)
    - Image quality assessment
    - Tensor conversion
    """
    
    # Standard ImageNet normalization (used by most pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # CLIP normalization (slightly different)
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    @staticmethod
    def load_image(
        image_path: Path,
        convert_rgb: bool = True
    ) -> Optional[Image.Image]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image
            convert_rgb: Convert to RGB mode
            
        Returns:
            PIL Image or None if load fails
        """
        try:
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if convert_rgb and img.mode != 'RGB':
                if img.mode == 'RGBA':
                    # Create white background for transparency
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                    img = background
                else:
                    img = img.convert('RGB')
            
            logger.debug(f"Image loaded: {image_path.name} ({img.size}, {img.mode})")
            
            return img
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    @staticmethod
    def resize_image(
        img: Image.Image,
        target_size: Union[int, Tuple[int, int]],
        method: str = 'lanczos'
    ) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            img: PIL Image
            target_size: Target size (int or (width, height))
            method: Resampling method (lanczos/bilinear/bicubic)
            
        Returns:
            Resized PIL Image
        """
        # Determine target dimensions
        if isinstance(target_size, int):
            # Resize shortest side to target_size
            width, height = img.size
            if width < height:
                new_width = target_size
                new_height = int(height * target_size / width)
            else:
                new_height = target_size
                new_width = int(width * target_size / height)
            target_size = (new_width, new_height)
        
        # Select resampling method
        resample_methods = {
            'lanczos': Image.Resampling.LANCZOS,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
        }
        resample = resample_methods.get(method.lower(), Image.Resampling.LANCZOS)
        
        # Resize
        resized = img.resize(target_size, resample)
        
        logger.debug(f"Image resized: {img.size} → {resized.size}")
        
        return resized
    
    @staticmethod
    def center_crop(
        img: Image.Image,
        crop_size: Union[int, Tuple[int, int]]
    ) -> Image.Image:
        """
        Center crop image.
        
        Args:
            img: PIL Image
            crop_size: Crop size (int or (width, height))
            
        Returns:
            Cropped PIL Image
        """
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        width, height = img.size
        crop_width, crop_height = crop_size
        
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        right = left + crop_width
        bottom = top + crop_height
        
        cropped = img.crop((left, top, right, bottom))
        
        logger.debug(f"Image center-cropped: {img.size} → {cropped.size}")
        
        return cropped
    
    @staticmethod
    def normalize_image(
        img: Union[Image.Image, torch.Tensor],
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        use_clip_norm: bool = False
    ) -> torch.Tensor:
        """
        Normalize image for ML model input.
        
        Args:
            img: PIL Image or torch Tensor
            mean: Normalization mean (defaults to ImageNet)
            std: Normalization std (defaults to ImageNet)
            use_clip_norm: Use CLIP normalization instead of ImageNet
            
        Returns:
            Normalized torch Tensor (C, H, W) in range [-1, 1]
        """
        # Convert PIL to tensor if needed
        if isinstance(img, Image.Image):
            img = ImageUtils.pil_to_tensor(img)
        
        # Select normalization parameters
        if mean is None or std is None:
            if use_clip_norm:
                mean = ImageUtils.CLIP_MEAN
                std = ImageUtils.CLIP_STD
            else:
                mean = ImageUtils.IMAGENET_MEAN
                std = ImageUtils.IMAGENET_STD
        
        # Normalize
        mean_tensor = torch.tensor(mean).view(3, 1, 1)
        std_tensor = torch.tensor(std).view(3, 1, 1)
        
        normalized = (img - mean_tensor) / std_tensor
        
        return normalized
    
    @staticmethod
    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """
        Convert PIL Image to torch Tensor.
        
        Args:
            img: PIL Image
            
        Returns:
            torch Tensor (C, H, W) in range [0, 1]
        """
        # Convert to numpy array
        img_array = np.array(img)
        
        # Handle different image modes
        if len(img_array.shape) == 2:
            # Grayscale
            img_array = np.expand_dims(img_array, axis=2)
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # Normalize to [0, 1]
        tensor = tensor / 255.0
        
        return tensor
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """
        Convert torch Tensor to PIL Image.
        
        Args:
            tensor: torch Tensor (C, H, W)
            
        Returns:
            PIL Image
        """
        # Denormalize if needed (assume range [0, 1] or [-1, 1])
        if tensor.min() < 0:
            # Range [-1, 1] -> [0, 1]
            tensor = (tensor + 1) / 2
        
        # Clamp to [0, 1]
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy (C, H, W) -> (H, W, C)
        img_array = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Scale to [0, 255]
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to PIL
        img = Image.fromarray(img_array)
        
        return img
    
    @staticmethod
    def preprocess_for_clip(
        img: Image.Image,
        size: int = 224
    ) -> torch.Tensor:
        """
        Preprocess image for CLIP model.
        
        Args:
            img: PIL Image
            size: Target size (CLIP uses 224x224)
            
        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Resize shortest side to size
        img = ImageUtils.resize_image(img, size)
        
        # Center crop to square
        img = ImageUtils.center_crop(img, size)
        
        # Convert to tensor and normalize
        tensor = ImageUtils.normalize_image(img, use_clip_norm=True)
        
        logger.debug(f"Image preprocessed for CLIP: {tensor.shape}")
        
        return tensor
    
    @staticmethod
    def preprocess_for_blip(
        img: Image.Image,
        size: int = 384
    ) -> torch.Tensor:
        """
        Preprocess image for BLIP model.
        
        Args:
            img: PIL Image
            size: Target size (BLIP typically uses 384x384)
            
        Returns:
            Preprocessed tensor (C, H, W)
        """
        # Resize and crop
        img = ImageUtils.resize_image(img, size)
        img = ImageUtils.center_crop(img, size)
        
        # Convert to tensor and normalize (ImageNet normalization)
        tensor = ImageUtils.normalize_image(img, use_clip_norm=False)
        
        logger.debug(f"Image preprocessed for BLIP: {tensor.shape}")
        
        return tensor
    
    @staticmethod
    def augment_image(
        img: Image.Image,
        augmentation: str
    ) -> Image.Image:
        """
        Apply single augmentation to image.
        
        Args:
            img: PIL Image
            augmentation: Augmentation type
            
        Returns:
            Augmented PIL Image
        """
        aug_img = img.copy()
        
        if augmentation == 'horizontal_flip':
            aug_img = ImageOps.mirror(aug_img)
        
        elif augmentation == 'vertical_flip':
            aug_img = ImageOps.flip(aug_img)
        
        elif augmentation == 'rotate_90':
            aug_img = aug_img.rotate(90, expand=True)
        
        elif augmentation == 'rotate_180':
            aug_img = aug_img.rotate(180, expand=True)
        
        elif augmentation == 'rotate_270':
            aug_img = aug_img.rotate(270, expand=True)
        
        elif augmentation == 'brightness_up':
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(1.2)
        
        elif augmentation == 'brightness_down':
            enhancer = ImageEnhance.Brightness(aug_img)
            aug_img = enhancer.enhance(0.8)
        
        elif augmentation == 'contrast_up':
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(1.2)
        
        elif augmentation == 'contrast_down':
            enhancer = ImageEnhance.Contrast(aug_img)
            aug_img = enhancer.enhance(0.8)
        
        elif augmentation == 'saturation_up':
            enhancer = ImageEnhance.Color(aug_img)
            aug_img = enhancer.enhance(1.2)
        
        return aug_img
    
    @staticmethod
    def test_time_augmentation(
        img: Image.Image,
        num_augmentations: int = 10
    ) -> List[Image.Image]:
        """
        Generate test-time augmentations.
        
        Creates multiple augmented versions for ensemble prediction.
        
        Args:
            img: PIL Image
            num_augmentations: Number of augmentations to generate
            
        Returns:
            List of augmented images (includes original)
        """
        augmentations = [
            'horizontal_flip',
            'brightness_up',
            'brightness_down',
            'contrast_up',
            'contrast_down',
            'saturation_up',
            'rotate_90',
            'rotate_180',
            'rotate_270'
        ]
        
        # Start with original
        augmented_images = [img.copy()]
        
        # Generate augmentations
        for i, aug_type in enumerate(augmentations):
            if len(augmented_images) >= num_augmentations:
                break
            
            aug_img = ImageUtils.augment_image(img, aug_type)
            augmented_images.append(aug_img)
        
        logger.debug(f"Generated {len(augmented_images)} augmented images")
        
        return augmented_images
    
    @staticmethod
    def check_image_quality(img: Image.Image) -> dict:
        """
        Check basic image quality metrics.
        
        Args:
            img: PIL Image
            
        Returns:
            dict: Quality metrics
        """
        width, height = img.size
        
        # Convert to numpy for analysis
        img_array = np.array(img)
        
        # Calculate statistics
        if len(img_array.shape) == 3:
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
        else:
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
        
        # Detect potential issues
        is_too_dark = mean_brightness < 30
        is_too_bright = mean_brightness > 225
        is_low_contrast = std_brightness < 20
        
        quality = {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'total_pixels': width * height,
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'is_low_contrast': is_low_contrast,
            'has_issues': is_too_dark or is_too_bright or is_low_contrast
        }
        
        return quality
    
    @staticmethod
    def batch_preprocess(
        images: List[Image.Image],
        target_size: int = 224,
        normalize: bool = True,
        use_clip_norm: bool = False
    ) -> torch.Tensor:
        """
        Batch preprocess images.
        
        Args:
            images: List of PIL Images
            target_size: Target size
            normalize: Apply normalization
            use_clip_norm: Use CLIP normalization
            
        Returns:
            Batched tensor (B, C, H, W)
        """
        processed = []
        
        for img in images:
            # Resize and crop
            img = ImageUtils.resize_image(img, target_size)
            img = ImageUtils.center_crop(img, target_size)
            
            # Convert to tensor
            tensor = ImageUtils.pil_to_tensor(img)
            
            # Normalize if requested
            if normalize:
                tensor = ImageUtils.normalize_image(tensor, use_clip_norm=use_clip_norm)
            
            processed.append(tensor)
        
        # Stack into batch
        batch = torch.stack(processed)
        
        logger.debug(f"Batch preprocessed: {batch.shape}")
        
        return batch


# Convenience functions

def load_and_preprocess_for_clip(image_path: Path) -> Optional[torch.Tensor]:
    """Load and preprocess image for CLIP."""
    img = ImageUtils.load_image(image_path)
    if img is None:
        return None
    return ImageUtils.preprocess_for_clip(img)


def load_and_preprocess_for_blip(image_path: Path) -> Optional[torch.Tensor]:
    """Load and preprocess image for BLIP."""
    img = ImageUtils.load_image(image_path)
    if img is None:
        return None
    return ImageUtils.preprocess_for_blip(img)
