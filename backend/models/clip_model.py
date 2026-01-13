"""
CLIP Model - Zero-shot deepfake detection

Uses openai/clip-vit-base-patch32 for:
- Image classification
- Zero-shot deepfake detection
- Image-text similarity
- Visual feature extraction
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel

from backend.models.base_model import BaseModel
from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class CLIPModel(BaseModel):
    """
    CLIP model for zero-shot deepfake detection.
    
    Model: openai/clip-vit-base-patch32
    Size: ~350MB
    
    Features:
    - Zero-shot classification
    - Image-text similarity
    - Deepfake detection
    - Feature extraction
    """
    
    MODEL_NAME = "openai/clip-vit-base-patch32"
    
    # Default prompts for deepfake detection
    DEEPFAKE_PROMPTS = [
        "a real photograph",
        "an authentic image",
        "a genuine photo",
        "a fake image",
        "a manipulated photograph",
        "a deepfake image",
        "an AI generated image",
        "a synthetic photo"
    ]
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize CLIP model.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision for faster inference
        """
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            use_fp16=use_fp16
        )
    
    def _load_model(self):
        """Load CLIP model and processor."""
        logger.info(f"Loading CLIP model: {self.model_name}")
        
        try:
            # Load processor (handles image and text preprocessing)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = HFCLIPModel.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested and available
            if self.use_fp16 and self.device != 'cpu':
                self.model.half()
            
            logger.info(
                f"CLIP model loaded successfully on {self.device}"
            )
        
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def predict_deepfake(
        self,
        image: Union[str, Path, Image.Image],
        prompts: Optional[List[str]] = None,
        return_scores: bool = False
    ) -> Union[float, Dict]:
        """
        Predict if image is a deepfake using zero-shot classification.
        
        Args:
            image: Image path or PIL Image
            prompts: Custom prompts (uses defaults if None)
            return_scores: Return detailed scores
            
        Returns:
            float: Authenticity score [0, 1] or dict with detailed scores
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Use default prompts if not provided
        if prompts is None:
            prompts = self.DEEPFAKE_PROMPTS
        
        # Compute image-text similarity
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()[0]
        
        # Calculate authenticity score
        # Assume first half are "real" prompts, second half are "fake"
        num_real = len(prompts) // 2
        real_score = probs_np[:num_real].sum()
        fake_score = probs_np[num_real:].sum()
        
        # Normalize to [0, 1] where 1 is authentic
        authenticity_score = float(real_score / (real_score + fake_score))
        
        if return_scores:
            return {
                'authenticity_score': authenticity_score,
                'real_probability': float(real_score),
                'fake_probability': float(fake_score),
                'detailed_scores': {
                    prompt: float(score)
                    for prompt, score in zip(prompts, probs_np)
                },
                'prediction': 'authentic' if authenticity_score > 0.5 else 'fake',
                'confidence': abs(authenticity_score - 0.5) * 2  # [0, 1]
            }
        
        return authenticity_score
    
    def classify_image(
        self,
        image: Union[str, Path, Image.Image],
        labels: List[str]
    ) -> Dict[str, float]:
        """
        Zero-shot image classification.
        
        Args:
            image: Image path or PIL Image
            labels: List of text labels to classify
            
        Returns:
            dict: Label probabilities
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Prepare inputs
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        
        # Convert to dict
        probs_np = probs.cpu().numpy()[0]
        
        return {
            label: float(prob)
            for label, prob in zip(labels, probs_np)
        }
    
    def compute_image_text_similarity(
        self,
        image: Union[str, Path, Image.Image],
        text: str
    ) -> float:
        """
        Compute similarity between image and text.
        
        Args:
            image: Image path or PIL Image
            text: Text description
            
        Returns:
            float: Similarity score [0, 1]
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Normalize to [0, 1]
            similarity = torch.sigmoid(outputs.logits_per_image[0, 0])
        
        return float(similarity.cpu().numpy())
    
    def extract_image_features(
        self,
        image: Union[str, Path, Image.Image]
    ) -> np.ndarray:
        """
        Extract image embeddings.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            np.ndarray: Image features
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def batch_predict_deepfake(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 8
    ) -> List[float]:
        """
        Batch deepfake prediction.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            
        Returns:
            List[float]: Authenticity scores
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Load images if paths
            loaded_images = []
            for img in batch_images:
                if isinstance(img, (str, Path)):
                    from backend.utils.image_utils import load_image
                    loaded_images.append(load_image(img))
                else:
                    loaded_images.append(img)
            
            # Process batch
            for img in loaded_images:
                score = self.predict_deepfake(img)
                scores.append(score)
        
        return scores
    
    def detect_with_augmentation(
        self,
        image: Union[str, Path, Image.Image],
        num_augmentations: int = 5
    ) -> Dict:
        """
        Deepfake detection with test-time augmentation.
        
        Args:
            image: Image path or PIL Image
            num_augmentations: Number of augmentations
            
        Returns:
            dict: Aggregated results
        """
        from backend.utils.image_utils import augment_image
        
        # Load image if path
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        scores = []
        
        # Original image
        original_score = self.predict_deepfake(image)
        scores.append(original_score)
        
        # Augmented versions
        for _ in range(num_augmentations - 1):
            augmented = augment_image(image)
            score = self.predict_deepfake(augmented)
            scores.append(score)
        
        # Aggregate scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return {
            'authenticity_score': float(mean_score),
            'confidence': float(1 - std_score),  # Lower variance = higher confidence
            'individual_scores': scores,
            'std_deviation': float(std_score),
            'num_augmentations': num_augmentations
        }
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model info
        """
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'use_fp16': self.use_fp16,
            'model_size_mb': 350,  # Approximate
            'capabilities': [
                'zero_shot_classification',
                'deepfake_detection',
                'image_text_similarity',
                'feature_extraction'
            ]
        }

""" Usage Examples
Example 1: Basic Deepfake Detection
python
from backend.models.model_cache import get_clip_model

# Get model
model = get_clip_model()

# Predict authenticity
score = model.predict_deepfake('suspect_image.jpg')
print(f"Authenticity score: {score:.2f}")

if score < 0.5:
    print("⚠️ Potential deepfake detected!")
else:
    print("✓ Image appears authentic")
Example 2: Detailed Analysis
python
# Get detailed scores
result = model.predict_deepfake('image.jpg', return_scores=True)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Authenticity: {result['authenticity_score']:.2f}")
print("\nDetailed scores:")
for prompt, score in result['detailed_scores'].items():
    print(f"  {prompt}: {score:.3f}")
Example 3: Test-Time Augmentation
python
# More robust detection with augmentation
result = model.detect_with_augmentation('image.jpg', num_augmentations=10)

print(f"Mean score: {result['authenticity_score']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Std deviation: {result['std_deviation']:.3f}")
Example 4: Custom Prompts
python
# Use domain-specific prompts
custom_prompts = [
    "a real government document",
    "an authentic official photo",
    "a forged document",
    "a manipulated official image"
]

score = model.predict_deepfake('document.jpg', prompts=custom_prompts)
print(f"Document authenticity: {score:.2f}")"""