from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..config import HEATMAPS_DIR
from ..utils.logger import logger
from .detection_service import get_xception_detector, preprocess_face_crop, _inference_lock


class XceptionGradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) engine for the DeepfakeBench Xception detector.
    
    Extracts visual attribution for the manipulated class (class 1) by computing the gradient of
    the raw class logit with respect to the final convolutional feature maps (model.backbone.conv4).
    """

    def __init__(self, target_layer=None):
        self.model, self.config, self.device = get_xception_detector()
        # The final convolutional layer of DeepfakeBench Xception is conv4 (SeparableConv2d, 2048 channels)
        self.target_layer = target_layer or self.model.backbone.conv4
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles = []

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple where the first element is the gradient of the loss/logit wrt module output
        self.gradients = grad_out[0]

    def register_hooks(self):
        self.remove_hooks()
        h1 = self.target_layer.register_forward_hook(self._forward_hook)
        h2 = self.target_layer.register_full_backward_hook(self._backward_hook)
        self.hook_handles = [h1, h2]

    def remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles = []

    def generate_cam(
        self,
        face_crop_bgr: np.ndarray,
        target_class: int = 1,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the normalized Grad-CAM activation map and colored overlay for a given face crop.

        Args:
            face_crop_bgr: Face crop in BGR format (numpy array).
            target_class: Class index for attribution (1 = manipulated/fake).
            colormap: OpenCV colormap (default: COLORMAP_JET).

        Returns:
            Tuple of (overlay_bgr, heatmap_colored_bgr).
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            raise ValueError("Invalid or empty face crop provided for Grad-CAM.")

        h_orig, w_orig = face_crop_bgr.shape[:2]
        resolution = self.config.get("resolution", 256)
        mean = self.config.get("mean", [0.5, 0.5, 0.5])
        std = self.config.get("std", [0.5, 0.5, 0.5])

        # Preprocess input face crop
        tensor = preprocess_face_crop(face_crop_bgr, resolution=resolution, mean=mean, std=std)
        tensor = tensor.to(self.device)
        tensor.requires_grad = True

        with _inference_lock:
            self.register_hooks()
            try:
                # Ensure model parameters require grad for backprop
                self.model.eval()
                self.model.zero_grad()

                # Forward pass through detector
                output = self.model({'image': tensor})
                logits = output['cls']  # Shape: [1, 2]

                # Target score: raw logit of class 1 (manipulated/deepfake)
                target_score = logits[:, target_class]

                # Backward pass to extract feature gradients
                target_score.backward(retain_graph=False)

                if self.activations is None or self.gradients is None:
                    raise RuntimeError("Failed to capture activations or gradients during Grad-CAM backpropagation.")

                # Compute channel importance weights via global-average-pooling of gradients
                # weights shape: [1, C, 1, 1]
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)

                # Weighted combination of forward activation maps
                # cam shape: [1, 1, H, W]
                cam = (weights * self.activations).sum(dim=1, keepdim=True)

                # Apply ReLU: only positive contributions to manipulated class are visualized
                cam = torch.relu(cam)

                # Extract 2D activation array
                cam_np = cam.squeeze().detach().cpu().numpy()

                # Normalize to [0.0, 1.0]
                cam_max = cam_np.max()
                if cam_max > 0:
                    cam_normalized = cam_np / cam_max
                else:
                    cam_normalized = np.zeros_like(cam_np)

                # Resize to match the original face crop dimensions
                cam_resized = cv2.resize(cam_normalized, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
                cam_resized = np.clip(cam_resized, 0.0, 1.0)

                # Generate 8-bit colorized heatmap
                heatmap_uint8 = np.uint8(255 * cam_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

                # Blend with original face crop (60% face, 40% heatmap)
                overlay = cv2.addWeighted(face_crop_bgr, 0.6, heatmap_colored, 0.4, 0)

                return overlay, heatmap_colored

            finally:
                self.remove_hooks()
                self.model.zero_grad()
                self.activations = None
                self.gradients = None


_gradcam_engine: Optional["XceptionGradCAM"] = None


def get_gradcam_engine() -> "XceptionGradCAM":
    """
    Returns the cached singleton instance of XceptionGradCAM.
    Hooks and activations are cleaned up deterministically in generate_cam finally block.
    """
    global _gradcam_engine
    if _gradcam_engine is None:
        _gradcam_engine = XceptionGradCAM()
    return _gradcam_engine


def generate_gradcam_for_face(
    face_crop_bgr: np.ndarray,
    identifier_prefix: str,
    target_class: int = 1
) -> Dict[str, Any]:
    """
    Convenience wrapper to compute and persist Grad-CAM visualization files.
    Safely catches exceptions so visualization issues never crash the main prediction pipeline.

    Args:
        face_crop_bgr: Face crop image as BGR array.
        identifier_prefix: Unique file prefix (e.g. 'DT-2026-XXXX_face0').
        target_class: Target class for visual attribution (1 = manipulated).

    Returns:
        Dict with status, file paths, and accessible static URLs.
    """
    try:
        HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
        gradcam = get_gradcam_engine()

        overlay_bgr, heatmap_bgr = gradcam.generate_cam(
            face_crop_bgr,
            target_class=target_class
        )

        # File naming
        overlay_filename = f"{identifier_prefix}_overlay.jpg"
        heatmap_filename = f"{identifier_prefix}_heatmap.jpg"
        crop_filename = f"{identifier_prefix}_crop.jpg"

        overlay_path = HEATMAPS_DIR / overlay_filename
        heatmap_path = HEATMAPS_DIR / heatmap_filename
        crop_path = HEATMAPS_DIR / crop_filename

        # Save artifacts to disk
        cv2.imwrite(str(overlay_path), overlay_bgr)
        cv2.imwrite(str(heatmap_path), heatmap_bgr)
        cv2.imwrite(str(crop_path), face_crop_bgr)

        logger.info(f"Grad-CAM visual attribution generated successfully for {identifier_prefix}.")

        return {
            "gradcam_available": True,
            "gradcam_overlay_url": f"/uploads/heatmaps/{overlay_filename}",
            "gradcam_heatmap_url": f"/uploads/heatmaps/{heatmap_filename}",
            "original_face_crop_url": f"/uploads/heatmaps/{crop_filename}",
            "target_layer": "model.backbone.conv4",
            "target_class": target_class
        }
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed for {identifier_prefix}: {e}", exc_info=True)
        return {
            "gradcam_available": False,
            "gradcam_overlay_url": None,
            "gradcam_heatmap_url": None,
            "original_face_crop_url": None,
            "error": str(e)
        }
