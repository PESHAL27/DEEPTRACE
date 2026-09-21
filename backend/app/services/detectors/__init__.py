"""
Deepfake Detector Adapters for DEEPTRACE Multi-Model Cross-Validation.
"""

from .base_detector import BaseDetector
from .xception_adapter import XceptionAdapter
from .meso4_adapter import Meso4Adapter

__all__ = ["BaseDetector", "XceptionAdapter", "Meso4Adapter"]
