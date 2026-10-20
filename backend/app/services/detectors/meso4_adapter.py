"""
Adapter for the Comparison DeepfakeBench MesoNet (Meso4) detector.
Implements the BaseDetector interface for multi-model cross-validation and comparative evaluation.
"""

import sys
import time
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from .base_detector import BaseDetector
from ...models.cross_validation_schemas import (
    NormalizedDetectorPrediction,
    ModelRegistryEntry
)
from ...config import DEEPFAKE_THRESHOLD
from ...utils.logger import logger

from ...config import MESO4_WEIGHTS_PATH, BENCH_DIR
_DEEPFAKEBENCH_DIR = BENCH_DIR
_MESO4_CHECKPOINT_PATH = MESO4_WEIGHTS_PATH

# Ensure DeepfakeBench is importable
if str(_DEEPFAKEBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_DEEPFAKEBENCH_DIR))
if str(_DEEPFAKEBENCH_DIR / "training") not in sys.path:
    sys.path.insert(0, str(_DEEPFAKEBENCH_DIR / "training"))


class Meso4Adapter(BaseDetector):
    """
    Adapter for the independent DeepfakeBench Meso4 deepfake detector.
    Role: 'comparison'
    """

    def __init__(self):
        self._model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._transform = None
        self._load_error: Optional[str] = None
        self._is_initialized = False

    @property
    def model_id(self) -> str:
        return "meso4_deepfakebench"

    @property
    def display_name(self) -> str:
        return "DeepfakeBench MesoNet (Meso4)"

    @property
    def role(self) -> str:
        return "comparison"

    @property
    def default_threshold(self) -> float:
        return 0.50

    def load(self) -> bool:
        """Loads Meso4 model weights and caches in memory."""
        if self._is_initialized and self._model is not None:
            return True

        if not _MESO4_CHECKPOINT_PATH.exists():
            self._load_error = f"Checkpoint not found at {_MESO4_CHECKPOINT_PATH}"
            logger.warning(f"[Meso4Adapter] {self._load_error}")
            return False

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"[Meso4Adapter] Initializing Meso4 on {self._device}...")

            # Import Meso4 architecture from DeepfakeBench networks
            from networks.mesonet import Meso4
            config = {"num_classes": 2, "inc": 3}
            model = Meso4(config)

            state_dict = torch.load(_MESO4_CHECKPOINT_PATH, map_location=self._device, weights_only=False)

            # Strip 'backbone.' prefix if present in the state dict
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    cleaned_state_dict[k[len("backbone."):]] = v
                else:
                    cleaned_state_dict[k] = v

            model.load_state_dict(cleaned_state_dict, strict=True)
            model.to(self._device)
            model.eval()

            # Preprocessing transform according to meso4.yaml:
            # resolution: 256, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            self._model = model
            self._is_initialized = True
            self._load_error = None
            logger.info("[Meso4Adapter] Meso4 detector initialized successfully.")
            return True
        except Exception as e:
            self._load_error = f"Failed to initialize Meso4 detector: {e}"
            logger.error(f"[Meso4Adapter] {self._load_error}", exc_info=True)
            self._model = None
            self._is_initialized = False
            return False

    def is_loaded(self) -> bool:
        return self._is_initialized and self._model is not None

    def predict_face_crop(
        self,
        face_rgb: np.ndarray,
        threshold: Optional[float] = None
    ) -> NormalizedDetectorPrediction:
        t = threshold if threshold is not None else self.default_threshold
        start_time = time.perf_counter()

        if not self.is_loaded():
            loaded = self.load()
            if not loaded or self._model is None:
                elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return NormalizedDetectorPrediction(
                    model_id=self.model_id,
                    display_name=self.display_name,
                    role=self.role,
                    prediction=None,
                    manipulated_probability=None,
                    authentic_probability=None,
                    threshold=t,
                    status="unavailable",
                    error_message=self._load_error or "Model checkpoint unavailable.",
                    inference_time_ms=elapsed_ms
                )

        try:
            # Model-specific preprocessing
            tensor = self._transform(face_rgb).unsqueeze(0).to(self._device)

            with torch.inference_mode():
                out, _ = self._model(tensor)
                # Softmax over 2 classes: Index 0 = authentic, Index 1 = manipulated
                probs = torch.softmax(out, dim=1)[0]
                manip_prob = round(float(probs[1].item()), 4)
                auth_prob = round(float(probs[0].item()), 4)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            is_manip = manip_prob >= t
            pred = "manipulated" if is_manip else "authentic"

            raw_logits_list = [round(float(v), 4) for v in out.squeeze().tolist()]

            logger.info(
                f"[Meso4 Forensic Diagnostic] Model: {self.model_id} | Checkpoint: {_MESO4_CHECKPOINT_PATH.name} | "
                f"Input: 256x256 RGB | Raw Logits: {raw_logits_list} | "
                f"Class 0 (Authentic): {auth_prob:.4f} | Class 1 (Manipulated): {manip_prob:.4f} | "
                f"Selected: {pred} (manip_prob={manip_prob:.4f}) | Decision Threshold: {t:.2f}"
            )

            return NormalizedDetectorPrediction(
                model_id=self.model_id,
                display_name=self.display_name,
                role=self.role,
                prediction=pred,
                manipulated_probability=manip_prob,
                authentic_probability=auth_prob,
                threshold=t,
                status="success",
                inference_time_ms=elapsed_ms
            )
        except Exception as e:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(f"[Meso4Adapter] Inference error: {e}", exc_info=True)
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
        ckpt_exists = _MESO4_CHECKPOINT_PATH.exists()
        ckpt_size = round(_MESO4_CHECKPOINT_PATH.stat().st_size / 1024, 1) if ckpt_exists else None
        status = "available" if (ckpt_exists and self.is_loaded()) else ("available" if ckpt_exists else "unavailable")

        return ModelRegistryEntry(
            model_id=self.model_id,
            display_name=self.display_name,
            architecture="MesoNet (Meso4)",
            framework="DeepfakeBench",
            task="binary deepfake classification",
            role=self.role,
            checkpoint=str(_MESO4_CHECKPOINT_PATH.name),
            checkpoint_size_kb=ckpt_size,
            input_size="256x256",
            preprocessing_mean=[0.5, 0.5, 0.5],
            preprocessing_std=[0.5, 0.5, 0.5],
            classes=["authentic", "manipulated"],
            default_threshold=self.default_threshold,
            license="CC BY-NC 4.0",
            availability_status=status,
            availability_notes=None if ckpt_exists else "Checkpoint meso4_best.pth not downloaded yet.",
            source_url="https://github.com/SCLBD/DeepfakeBench",
            paper_citation="Afchar et al., 'MesoNet: a Compact Facial Video Forgery Detection Network', IEEE WIFS 2018"
        )
