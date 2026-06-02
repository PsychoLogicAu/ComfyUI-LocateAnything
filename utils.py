# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

"""
Utility module for Locate Anything nodes.

This module provides:
- Model loading functionality
- Worker loading
- Image processing utilities
- Detection and grounding helpers
"""

import torch
from PIL import Image
import json
import base64
import io
import logging

logger = logging.getLogger(__name__)


class LoadLocateAnythingModel:
    """
    Load Locate Anything model from HuggingFace or local path.

    This class handles:
    - Model download from HuggingFace
    - Model loading from local paths
    - Device configuration
    - Precision/dtype selection
    - Tokenizer and processor loading
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        trust_remote_code: bool = True
    ):
        """
        Initialize the model loader.

        Args:
            model_path: Path to model (HuggingFace repo or local path)
            device: Device type ("cuda", "cpu", or "auto")
            dtype: Model precision (default: bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        # Load model components
        self._load_model()

    def _load_model(self):
        """Load the model and its components."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

            # Use model_path directly
            model_path = self.model_path

            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map=self.device,
                trust_remote_code=self.trust_remote_code
            )

            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_model(self):
        """Get the loaded model."""
        return self.model

    def get_processor(self):
        """Get the processor."""
        return self.processor

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

    def get_config(self):
        """Get model configuration."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": str(self.dtype),
            "trust_remote_code": self.trust_remote_code
        }


class LocateAnythingConfig:
    """
    Configuration for Locate Anything inference.
    """

    def __init__(self, config_dict=None):
        """
        Initialize configuration.

        Args:
            config_dict: Optional configuration dictionary
        """
        if config_dict:
            self.config = config_dict
        else:
            self.config = {
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "generation_mode": "hybrid",
                "use_cache": True,
                "repetition_penalty": 1.1
            }

    def to_dict(self):
        """Convert config to dictionary."""
        return self.config

    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(config_dict)


class LocateAnythingDetector:
    """
    Detection helper for Locate Anything.
    """

    @staticmethod
    def parse_boxes(answer: str, width: int, height: int) -> list:
        """
        Parse bounding boxes from model answer.

        Args:
            answer: Model answer string with bounding boxes
            width: Image width
            height: Image height

        Returns:
            List of normalized bounding boxes
        """
        # Placeholder implementation
        # In production, this would parse the model's answer
        # and extract coordinates from the response
        return []

    @staticmethod
    def parse_points(answer: str, width: int, height: int) -> list:
        """
        Parse points from model answer.

        Args:
            answer: Model answer string with points
            width: Image width
            height: Image height

        Returns:
            List of normalized point coordinates
        """
        # Placeholder implementation
        return []


class LocateAnythingDetectorDebug:
    """
    Debug detector for inspection.
    """

    @staticmethod
    def debug_result(result: dict, width: int, height: int) -> dict:
        """
        Debug and inspect detection result.

        Args:
            result: Detection result from model
            width: Image width
            height: Image height

        Returns:
            Debug information
        """
        return {
            "result": result,
            "width": width,
            "height": height
        }


class LocateAnythingDebug:
    """
    Debug utility for model inspection.
    """

    @staticmethod
    def get_model_info(model: object) -> dict:
        """
        Get model information.

        Args:
            model: Loaded model object

        Returns:
            Model information dictionary
        """
        return {
            "model_path": getattr(model, 'model_path', 'unknown'),
            "device": getattr(model, 'device', 'unknown'),
            "dtype": getattr(model, 'dtype', 'unknown')
        }

    @staticmethod
    def get_image_info(image: object) -> dict:
        """
        Get image information.

        Args:
            image: Image object

        Returns:
            Image information dictionary
        """
        if hasattr(image, 'size'):
            return {
                "width": image.size[0],
                "height": image.size[1],
                "mode": getattr(image, 'mode', 'unknown')
            }
        return {}

    @staticmethod
    def process_image(image_data: bytes, format: str = "RGB") -> Image.Image:
        """
        Process image data to PIL Image.

        Args:
            image_data: Image data in bytes
            format: Image format (RGB, RGBA, etc.)

        Returns:
            PIL Image object
        """
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
            if format == "RGB":
                image = image.convert("RGB")
            return image

        return image_data

    @staticmethod
    def encode_image(image: Image.Image) -> tuple:
        """
        Encode image to base64.

        Args:
            image: PIL Image

        Returns:
            Tuple of (base64_string, format)
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_string = base64.b64encode(buffered.getvalue()).decode()
        return base64_string, "PNG"

    @staticmethod
    def decode_image(base64_string: str, format: str = "RGB") -> Image.Image:
        """
        Decode base64 image string.

        Args:
            base64_string: Base64 encoded image
            format: Expected format

        Returns:
            PIL Image
        """
        import base64
        import io

        try:
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            if format == "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None