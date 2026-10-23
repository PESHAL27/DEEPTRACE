"""
Deepfake Detector Adapters for DEEPTRACE Multi-Model Cross-Validation.
"""

from .base_detector import BaseDetector
from .xception_adapter import XceptionAdapter
from .meso4_adapter import Meso4Adapter
from .ai_generated_image_detector import detect_ai_generated_image, get_smaug_detector

__all__ = [
    "BaseDetector",
    "XceptionAdapter",
    "Meso4Adapter",
    "detect_ai_generated_image",
    "get_smaug_detector"
]
