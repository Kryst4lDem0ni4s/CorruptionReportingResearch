"""
Metadata Service - File metadata removal

Provides:
- EXIF data stripping from images
- GPS location removal
- Camera/device information removal
- Timestamp sanitization
- Metadata anonymization
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PIL import Image
from PIL.ExifTags import TAGS

# Initialize logger
logger = logging.getLogger(__name__)


class MetadataService:
    """
    Metadata Service - Remove sensitive metadata from files.
    
    Features:
    - EXIF data removal from images (JPEG, PNG, TIFF, etc.)
    - GPS location stripping
    - Camera model/serial number removal
    - Timestamp anonymization
    - File metadata sanitization
    
    Protects whistleblower anonymity by removing identifying information.
    """
    
    def __init__(self):
        """Initialize metadata service."""
        logger.info("MetadataService initialized")
    
    def strip_image_metadata(
        self,
        input_path: Path,
        output_path: Path,
        preserve_orientation: bool = True
    ) -> Dict:
        """
        Strip all EXIF/metadata from image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save cleaned image
            preserve_orientation: Keep image orientation (recommended)
            
        Returns:
            dict: Report of stripped metadata
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file is not a valid image
            IOError: If processing fails
        """
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Open image
            with Image.open(input_path) as img:
                # Extract metadata before stripping (for logging)
                original_metadata = self._extract_metadata(img)
                
                # Get image format
                img_format = img.format or 'JPEG'
                
                # Handle orientation if needed
                if preserve_orientation and hasattr(img, '_getexif'):
                    img = self._correct_orientation(img)
                
                # Create new image without metadata
                # Convert to RGB if necessary (removes alpha channel issues)
                if img.mode not in ('RGB', 'L'):
                    if img.mode == 'RGBA':
                        # Create white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])  # Use alpha as mask
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Save without metadata
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save parameters (no EXIF)
                save_kwargs = {
                    'format': img_format,
                    'quality': 95,  # High quality
                    'optimize': True
                }
                
                # For JPEG, explicitly disable EXIF
                if img_format.upper() in ('JPEG', 'JPG'):
                    save_kwargs['exif'] = b''
                
                img.save(output_path, **save_kwargs)
            
            # Verify metadata was removed
            cleaned_metadata = self._get_metadata_summary(output_path)
            
            report = {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'original_metadata_count': len(original_metadata),
                'cleaned_metadata_count': len(cleaned_metadata),
                'metadata_removed': original_metadata,
                'sensitive_data_stripped': self._identify_sensitive_fields(original_metadata),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Metadata stripped: {input_path.name} → {output_path.name} "
                f"({len(original_metadata)} fields removed)"
            )
            
            return report
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Metadata stripping failed for {input_path}: {e}")
            raise IOError(f"Metadata stripping failed: {str(e)}")
    
    def _extract_metadata(self, img: Image.Image) -> Dict:
        """
        Extract all EXIF/metadata from image.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Metadata key-value pairs
        """
        metadata = {}
        
        try:
            # Try to get EXIF data
            exif_data = img._getexif()
            
            if exif_data:
                for tag_id, value in exif_data.items():
                    # Get human-readable tag name
                    tag = TAGS.get(tag_id, tag_id)
                    
                    # Convert bytes to string for logging
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            value = f"<binary data: {len(value)} bytes>"
                    
                    metadata[tag] = str(value)
            
        except AttributeError:
            # No EXIF data
            pass
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _correct_orientation(self, img: Image.Image) -> Image.Image:
        """
        Correct image orientation based on EXIF data.
        
        Args:
            img: PIL Image object
            
        Returns:
            Image: Oriented image
        """
        try:
            exif = img._getexif()
            
            if exif:
                # Orientation tag
                orientation_tag = 0x0112
                
                if orientation_tag in exif:
                    orientation = exif[orientation_tag]
                    
                    # Rotation mappings
                    rotations = {
                        3: 180,
                        6: 270,
                        8: 90
                    }
                    
                    if orientation in rotations:
                        img = img.rotate(rotations[orientation], expand=True)
                        logger.debug(f"Corrected orientation: {orientation}")
        
        except Exception as e:
            logger.warning(f"Orientation correction failed: {e}")
        
        return img
    
    def _get_metadata_summary(self, image_path: Path) -> Dict:
        """
        Get summary of remaining metadata in image.
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: Metadata summary
        """
        try:
            with Image.open(image_path) as img:
                return self._extract_metadata(img)
        except Exception as e:
            logger.warning(f"Failed to get metadata summary: {e}")
            return {}
    
    def _identify_sensitive_fields(self, metadata: Dict) -> list:
        """
        Identify sensitive metadata fields that were removed.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            list: List of sensitive field names
        """
        sensitive_fields = []
        
        # Common sensitive EXIF tags
        sensitive_tags = [
            'GPSInfo',           # GPS location
            'GPSLatitude',
            'GPSLongitude',
            'GPSAltitude',
            'GPSTimeStamp',
            'Make',              # Camera manufacturer
            'Model',             # Camera model
            'Software',          # Editing software
            'DateTime',          # Capture time
            'DateTimeOriginal',
            'DateTimeDigitized',
            'HostComputer',      # Computer name
            'Copyright',         # Copyright info
            'Artist',            # Artist/creator
            'CameraSerialNumber',
            'LensSerialNumber',
            'OwnerName',
            'UserComment'
        ]
        
        for tag in sensitive_tags:
            if tag in metadata:
                sensitive_fields.append(tag)
        
        return sensitive_fields
    
    def has_gps_data(self, image_path: Path) -> bool:
        """
        Check if image contains GPS location data.
        
        Args:
            image_path: Path to image
            
        Returns:
            bool: True if GPS data present, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                metadata = self._extract_metadata(img)
                
                gps_tags = ['GPSInfo', 'GPSLatitude', 'GPSLongitude']
                return any(tag in metadata for tag in gps_tags)
                
        except Exception as e:
            logger.warning(f"GPS check failed: {e}")
            return False
    
    def get_metadata_report(self, image_path: Path) -> Dict:
        """
        Get detailed metadata report for image.
        
        Args:
            image_path: Path to image
            
        Returns:
            dict: Metadata report
        """
        try:
            with Image.open(image_path) as img:
                metadata = self._extract_metadata(img)
                
                report = {
                    'file_path': str(image_path),
                    'format': img.format,
                    'size': img.size,
                    'mode': img.mode,
                    'metadata_count': len(metadata),
                    'has_gps': any(k.startswith('GPS') for k in metadata.keys()),
                    'has_camera_info': 'Make' in metadata or 'Model' in metadata,
                    'has_timestamp': any(
                        k in metadata for k in ['DateTime', 'DateTimeOriginal']
                    ),
                    'sensitive_fields': self._identify_sensitive_fields(metadata),
                    'all_metadata': metadata
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Failed to get metadata report: {e}")
            return {
                'error': str(e),
                'file_path': str(image_path)
            }
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove identifying information.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Get extension
        path = Path(filename)
        extension = path.suffix
        
        # Generate anonymous name
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        sanitized = f"evidence_{timestamp}{extension}"
        
        logger.debug(f"Filename sanitized: {filename} → {sanitized}")
        
        return sanitized
    
    def batch_strip_metadata(
        self,
        input_dir: Path,
        output_dir: Path,
        extensions: Optional[list] = None
    ) -> Dict:
        """
        Strip metadata from all images in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            extensions: Image extensions to process (default: jpg, png, jpeg)
            
        Returns:
            dict: Batch processing report
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        processed = []
        failed = []
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Find all images
            image_files = []
            for ext in extensions:
                image_files.extend(input_dir.glob(f"*{ext}"))
                image_files.extend(input_dir.glob(f"*{ext.upper()}"))
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Process each image
            for input_path in image_files:
                try:
                    output_path = output_dir / input_path.name
                    
                    report = self.strip_image_metadata(input_path, output_path)
                    processed.append(report)
                    
                except Exception as e:
                    logger.error(f"Failed to process {input_path}: {e}")
                    failed.append({
                        'file': str(input_path),
                        'error': str(e)
                    })
            
            batch_report = {
                'input_directory': str(input_dir),
                'output_directory': str(output_dir),
                'total_files': len(image_files),
                'processed': len(processed),
                'failed': len(failed),
                'processed_files': processed,
                'failed_files': failed,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Batch processing complete: {len(processed)} processed, "
                f"{len(failed)} failed"
            )
            
            return batch_report
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {
                'error': str(e),
                'processed': len(processed),
                'failed': len(failed)
            }
