"""
Base Detector interface for DEEPTRACE Multi-Model Cross Validation.
Every registered deepfake detector implements this contract.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np

from ...models.cross_validation_schemas import (
    NormalizedDetectorPrediction,
    ModelRegistryEntry
)


class BaseDetector(ABC):
    """
    Abstract contract for a deepfake detection model adapter.
    Guarantees consistent normalized predictions, explicit class conversions,
    model-specific preprocessing isolation, and failure containment.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique programmatic identifier for the detector."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable display title."""
        pass

    @property
    @abstractmethod
    def role(self) -> str:
        """Model role: 'primary' or 'comparison'."""
        pass

    @property
    @abstractmethod
    def default_threshold(self) -> float:
        """Default binary decision threshold."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """
        Loads the model weights and initializes the inference pipeline.
        Must be idempotent and safe to call multiple times.
        Returns True if loaded and ready, False if unavailable.
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the underlying model is initialized and ready."""
        pass

    @abstractmethod
    def predict_face_crop(
        self,
        face_rgb: np.ndarray,
        threshold: Optional[float] = None
    ) -> NormalizedDetectorPrediction:
        """
        Runs model-specific preprocessing and forward pass on a single RGB face crop.
        Returns a NormalizedDetectorPrediction.
        """
        pass

    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        threshold: Optional[float] = None
    ) -> List[NormalizedDetectorPrediction]:
        """
        Runs predictions on a list of face crops.
        Default implementation iterates over predict_face_crop.
        """
        return [self.predict_face_crop(crop, threshold=threshold) for crop in face_crops]

    @abstractmethod
    def get_metadata(self) -> ModelRegistryEntry:
        """Returns full metadata entry for the model registry."""
        pass
