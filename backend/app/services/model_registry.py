"""
Model Registry service for DEEPTRACE Multi-Model Cross-Validation.
Maintains all supported detectors, their roles (primary vs comparison),
metadata descriptors, and singleton adapter instances.
"""

from typing import Dict, List, Optional
from .detectors.base_detector import BaseDetector
from .detectors.xception_adapter import XceptionAdapter
from .detectors.meso4_adapter import Meso4Adapter
from ..models.cross_validation_schemas import ModelRegistryEntry
from ..utils.logger import logger


class ModelRegistry:
    """
    Central registry for primary and comparison deepfake detection models.
    Guarantees that detectors are initialized as singletons, handles failure isolation,
    and exposes live operational metadata.
    """

    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._primary_id: str = "xception_deepfakebench"
        self._initialize_default_registry()

    def _initialize_default_registry(self):
        """Registers the primary production detector and comparison detector."""
        # 1. Primary Production Model: DeepfakeBench Xception
        xception = XceptionAdapter()
        self.register(xception)
        self._primary_id = xception.model_id

        # 2. Comparison Model (Research): DeepfakeBench MesoNet (Meso4)
        meso4 = Meso4Adapter()
        self.register(meso4)

    def register(self, detector: BaseDetector):
        """Registers a detector adapter instance in the registry."""
        self._detectors[detector.model_id] = detector
        logger.info(f"[ModelRegistry] Registered detector '{detector.model_id}' ({detector.display_name}) [role={detector.role}].")

    def get_detector(self, model_id: str) -> Optional[BaseDetector]:
        """Retrieves a detector by its unique identifier."""
        return self._detectors.get(model_id)

    def get_primary_detector(self) -> BaseDetector:
        """Retrieves the primary production detector (DeepfakeBench Xception)."""
        detector = self._detectors.get(self._primary_id)
        if detector is None:
            raise RuntimeError(f"Primary detector '{self._primary_id}' not found in registry.")
        return detector

    def get_comparison_detectors(self) -> List[BaseDetector]:
        """Retrieves all registered comparison detectors."""
        return [d for d in self._detectors.values() if d.role == "comparison"]

    def list_models(self) -> List[ModelRegistryEntry]:
        """Returns metadata descriptors for all registered detectors."""
        return [d.get_metadata() for d in self._detectors.values()]

    def list_model_ids(self) -> List[str]:
        """Returns a list of all registered model IDs."""
        return list(self._detectors.keys())


# Singleton instance
_model_registry_instance: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Returns the singleton ModelRegistry instance."""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry()
    return _model_registry_instance
