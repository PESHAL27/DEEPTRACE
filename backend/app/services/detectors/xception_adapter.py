"""
Adapter for the Primary DeepfakeBench Xception detector.
Wraps existing production Xception inference into the standardized BaseDetector interface.
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np

from .base_detector import BaseDetector
from ...models.cross_validation_schemas import (
    NormalizedDetectorPrediction,
    ModelRegistryEntry
)
from ..detection_service import get_xception_detector, predict_face_crop, WEIGHTS_PATH
from ...config import DEEPFAKE_THRESHOLD
from ...utils.logger import logger


class XceptionAdapter(BaseDetector):
    """
    Adapter for the primary DeepfakeBench Xception detector.
    Role: 'primary'
    """

    @property
    def model_id(self) -> str:
        return "xception_deepfakebench"

    @property
    def display_name(self) -> str:
        return "DeepfakeBench Xception"

    @property
    def role(self) -> str:
        return "primary"

    @property
    def default_threshold(self) -> float:
        return DEEPFAKE_THRESHOLD

    def load(self) -> bool:
        detector, _, _ = get_xception_detector()
        return detector is not None

    def is_loaded(self) -> bool:
        detector, _, _ = get_xception_detector()
        return detector is not None

    def predict_face_crop(
        self,
        face_rgb: np.ndarray,
        threshold: Optional[float] = None
    ) -> NormalizedDetectorPrediction:
        t = threshold if threshold is not None else self.default_threshold
        start_time = time.perf_counter()

        try:
            if not self.is_loaded():
                self.load()

            # Execute production predict_face_crop (face_rgb is already RGB)
            res = predict_face_crop(face_rgb, threshold=t, is_bgr=False)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            manipulated_prob = res.get("fake_probability", 0.0)
            auth_prob = res.get("real_probability", round(1.0 - manipulated_prob, 4))
            is_manipulated = (manipulated_prob >= t)
            pred = "manipulated" if is_manipulated else "authentic"


            return NormalizedDetectorPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                role=self.role,
                prediction=pred,
                manipulated_probability=round(manipulated_prob, 4),
                authentic_probability=auth_prob,
                threshold=t,
                status="success",
                inference_time_ms=elapsed_ms
            )
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"[XceptionAdapter] Inference error: {e}", exc_info=True)
            return NormalizedDetectorPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                role=self.role,
                prediction=None,
                manipulated_probability=None,
                authentic_probability=None,
                threshold=t,
                status="error",
                error_message=str(e),
                inference_time_ms=elapsed_ms
            )

    def get_metadata(self) -> ModelRegistryEntry:
        ckpt_path = Path(WEIGHTS_PATH)
        ckpt_size = round(ckpt_path.stat().st_size / 1024, 1) if ckpt_path.exists() else None
        status = "available" if (ckpt_path.exists() and self.is_loaded()) else "unavailable"

        return ModelRegistryEntry(
            model_id=self.model_id,
            display_name=self.display_name,
            architecture="Xception",
            framework="DeepfakeBench",
            task="binary deepfake classification",
            role=self.role,
            checkpoint=str(ckpt_path.name),
            checkpoint_size_kb=ckpt_size,
            input_size="256x256",
            preprocessing_mean=[0.5, 0.5, 0.5],
            preprocessing_std=[0.5, 0.5, 0.5],
            classes=["authentic", "manipulated"],
            default_threshold=self.default_threshold,
            license="CC BY-NC 4.0",
            availability_status=status,
            source_url="https://github.com/SCLBD/DeepfakeBench",
            paper_citation="Yan et al., 'DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection', NeurIPS 2023 D&B"
        )
