"""
BLIP Model - Image captioning and visual understanding

Uses Salesforce/blip-image-captioning-base for:
- Image caption generation
- Visual description
- Cross-modal verification
- Image understanding
"""

import logging
from pathlib import Path
from typing import Optional, Union, List

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from backend.models.base_model import BaseModel
from backend.utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class BLIPModel(BaseModel):
    """
    BLIP model for image captioning.
    
    Model: Salesforce/blip-image-captioning-base
    Size: ~500MB
    
    Features:
    - Unconditional image captioning
    - Conditional image captioning (with prompts)
    - Visual understanding
    - Caption generation for verification
    """
    
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    MAX_LENGTH = 50  # Max caption length
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        Initialize BLIP model.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto)
            use_fp16: Use FP16 precision for faster inference
        """
        super().__init__(
            model_name=self.MODEL_NAME,
            device=device,
            use_fp16=use_fp16
        )
        
        self.max_length = self.MAX_LENGTH
    
    def _load_model(self):
        """Load BLIP model and processor."""
        logger.info(f"Loading BLIP model: {self.model_name}")
        
        try:
            # Load processor
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Enable FP16 if requested and available
            if self.use_fp16 and self.device != 'cpu':
                self.model.half()
            
            logger.info(
                f"BLIP model loaded successfully on {self.device}"
            )
            
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise
    
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: int = 5
    ) -> str:
        """
        Generate caption for image.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt for conditional captioning
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Set max length
        if max_length is None:
            max_length = self.max_length
        
        # Prepare inputs
        if prompt:
            # Conditional captioning
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
        else:
            # Unconditional captioning
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode caption
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def generate_multiple_captions(
        self,
        image: Union[str, Path, Image.Image],
        num_captions: int = 3,
        temperature: float = 1.0
    ) -> List[str]:
        """
        Generate multiple diverse captions for image.
        
        Args:
            image: Image path or PIL Image
            num_captions: Number of captions to generate
            temperature: Sampling temperature for diversity
            
        Returns:
            List[str]: List of generated captions
        """
        # Ensure model is loaded
        if not self.is_loaded:
            self.load()
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            from backend.utils.image_utils import load_image
            image = load_image(image)
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        captions = []
        
        # Generate multiple captions
        with torch.no_grad():
            for _ in range(num_captions):
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95
                )
                
                caption = self.processor.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                captions.append(caption)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_captions = []
        for caption in captions:
            if caption not in seen:
                seen.add(caption)
                unique_captions.append(caption)
        
        return unique_captions
    
    def verify_image_text_consistency(
        self,
        image: Union[str, Path, Image.Image],
        text: str
    ) -> dict:
        """
        Verify consistency between image and text.
        
        Args:
            image: Image path or PIL Image
            text: Text description to verify
            
        Returns:
            dict: Verification results
        """
        # Generate caption from image
        generated_caption = self.generate_caption(image)
        
        # Compare with provided text using sentence similarity
        from backend.models.model_cache import get_sentence_transformer
        
        st_model = get_sentence_transformer()
        similarity = st_model.similarity(generated_caption, text)
        
        # Determine consistency
        is_consistent = similarity > 0.6  # Threshold
        
        return {
            'is_consistent': is_consistent,
            'similarity_score': float(similarity),
            'generated_caption': generated_caption,
            'provided_text': text,
            'confidence': 'high' if similarity > 0.75 else 'medium' if similarity > 0.6 else 'low'
        }
    
    def batch_generate_captions(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 4
    ) -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            
        Returns:
            List[str]: List of generated captions
        """
        captions = []
        
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
            inputs = self.processor(
                images=loaded_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate captions
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=5
                )
            
            # Decode all captions in batch
            batch_captions = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            captions.extend(batch_captions)
        
        return captions
    
    def describe_image_details(
        self,
        image: Union[str, Path, Image.Image]
    ) -> dict:
        """
        Generate detailed description of image.
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            dict: Detailed image description
        """
        # Generate multiple captions with different prompts
        prompts = [
            None,  # Unconditional
            "a photography of",
            "this image shows",
            "in this picture"
        ]
        
        descriptions = {}
        
        for i, prompt in enumerate(prompts):
            caption = self.generate_caption(image, prompt=prompt)
            key = f'description_{i+1}' if prompt is None else f'description_with_prompt_{i+1}'
            descriptions[key] = caption
        
        # Generate overall summary
        all_captions = list(descriptions.values())
        
        # Find most common words/themes
        from collections import Counter
        
        words = []
        for caption in all_captions:
            words.extend(caption.lower().split())
        
        common_words = Counter(words).most_common(5)
        
        return {
            'descriptions': descriptions,
            'primary_caption': all_captions[0],
            'common_themes': [word for word, count in common_words],
            'total_descriptions': len(all_captions)
        }
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            dict: Model info
        """
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device': str(self.device),
            'is_loaded': self.is_loaded,
            'use_fp16': self.use_fp16,
            'model_size_mb': 500,  # Approximate
            'capabilities': [
                'unconditional_captioning',
                'conditional_captioning',
                'image_text_verification',
                'batch_processing'
            ]
        }

"""Integration Examples
Using Wav2Vec2 in Layer 2 (Credibility Assessment)
python
# backend/core/layer2_credibility.py

from backend.models.model_cache import get_wav2vec_model

class CredibilityAssessmentLayer:
    def __init__(self):
        self.audio_model = get_wav2vec_model()
    
    def analyze_audio_evidence(self, audio_path: str) -> dict:
        
        # Extract features and analyze
        analysis = self.audio_model.analyze_audio_authenticity(audio_path)
        
        return {
            'authenticity_score': analysis['authenticity_score'],
            'has_artifacts': analysis['artifacts']['total_artifacts'] > 5,
            'analysis': analysis
        }
Using BLIP in Layer 2 (Cross-Modal Verification)
python
# backend/core/layer2_credibility.py

from backend.models.model_cache import get_blip_model

class CredibilityAssessmentLayer:
    def __init__(self):
        self.caption_model = get_blip_model()
    
    def verify_image_narrative_consistency(
        self,
        image_path: str,
        narrative: str
    ) -> dict:
        
        result = self.caption_model.verify_image_text_consistency(
            image_path,
            narrative
        )
        
        return {
            'is_consistent': result['is_consistent'],
            'similarity': result['similarity_score'],
            'generated_caption': result['generated_caption']
        }"""