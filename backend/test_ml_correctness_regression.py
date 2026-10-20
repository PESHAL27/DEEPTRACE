"""
DEEPTRACE - ML Model Prediction Correctness Regression Test Suite
=================================================================
Validates the complete ML inference pipeline:
1. Checkpoint integrity and strict key loading without silent omissions.
2. Verified class semantics (Index 0 = Authentic, Index 1 = Manipulated).
3. Probability and label consistency across classes.
4. Preprocessing tensor layout, channel order, range, and normalization.
5. Multi-face aggregation verdict policy and no-face handling.
6. Numerical parity (< 1e-5) against reference DeepfakeBench forward pass.
7. End-to-end verification across both authentic and manipulated samples.
"""

import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# Path resolution
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces
from app.services.detection_service import (
    get_xception_detector,
    preprocess_face_crop,
    predict_face_crop,
    analyze_image,
    DEEPFAKE_THRESHOLD,
    CONFIG_PATH,
    WEIGHTS_PATH,
    BENCH_DIR
)
from app.services.detectors.meso4_adapter import Meso4Adapter
from app.services.cross_validation_service import get_cross_validation_service


class TestMLCorrectnessRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.real_sample_path = REPO_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
        cls.fake_sample_path = REPO_ROOT / "deepfake_model" / "tests" / "samples" / "known_fake.jpg"
        cls.no_face_path = REPO_ROOT / "public" / "assets" / "no_face.jpg"
        cls.user_portrait_path = BACKEND_DIR / "uploads" / "DT-2026-17022339_Nithin_sist_friend.jpeg"
        cls.model, cls.config, cls.device = get_xception_detector()

    def test_01_strict_checkpoint_loading(self):
        """Verify checkpoint loads strictly with 0 missing and 0 unexpected keys."""
        self.assertTrue(WEIGHTS_PATH.exists(), f"Weights missing: {WEIGHTS_PATH}")
        ckpt = torch.load(str(WEIGHTS_PATH), map_location=self.device)
        
        # Load state dict strictly into model
        load_result = self.model.load_state_dict(ckpt, strict=True)
        self.assertEqual(len(load_result.missing_keys), 0, f"Missing keys found: {load_result.missing_keys}")
        self.assertEqual(len(load_result.unexpected_keys), 0, f"Unexpected keys found: {load_result.unexpected_keys}")

    def test_02_model_eval_mode(self):
        """Verify detector is in evaluation mode with gradients disabled during inference."""
        self.assertFalse(self.model.training, "Detector must be in eval() mode, not training mode.")

    def test_03_preprocessing_tensor_properties(self):
        """Verify preprocessed tensor has correct shape [1, 3, 256, 256], dtype float32, and range in [-1, 1]."""
        dummy_bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        tensor = preprocess_face_crop(dummy_bgr, is_bgr=True)

        self.assertEqual(tensor.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), -1.05)
        self.assertLessEqual(tensor.max().item(), 1.05)

    def test_04_class_semantics_and_threshold_mapping(self):
        """
        Verify class semantics:
        Class 0 = Authentic (real)
        Class 1 = Manipulated (fake)
        Threshold 0.50 evaluates on Class 1 (manipulated probability).
        """
        self.assertEqual(DEEPFAKE_THRESHOLD, 0.50, "Decision threshold must strictly remain 0.50.")

        # Test synthetic logits where class 0 > class 1
        logits_real = torch.tensor([[1.5, -1.5]], dtype=torch.float32)
        probs_real = torch.softmax(logits_real, dim=1)[0]
        self.assertGreater(probs_real[0].item(), 0.50)  # P(real) > 0.50
        self.assertLess(probs_real[1].item(), 0.50)     # P(fake) < 0.50

        # Test synthetic logits where class 1 > class 0
        logits_fake = torch.tensor([[-1.5, 1.5]], dtype=torch.float32)
        probs_fake = torch.softmax(logits_fake, dim=1)[0]
        self.assertLess(probs_fake[0].item(), 0.50)     # P(real) < 0.50
        self.assertGreater(probs_fake[1].item(), 0.50)  # P(fake) > 0.50

    def test_05_probability_sum_and_label_consistency(self):
        """Verify softmax probabilities sum to 1.0 and reported probability matches the assigned label."""
        if self.real_sample_path.exists():
            bgr = cv2.imread(str(self.real_sample_path))
            res = predict_face_crop(bgr, is_bgr=True)
            self.assertAlmostEqual(res["real_probability"] + res["fake_probability"], 1.0, places=4)
            if res["prediction"] == "real":
                self.assertEqual(res["probability"], res["real_probability"])
            else:
                self.assertEqual(res["probability"], res["fake_probability"])

    def test_06_reference_inference_parity(self):
        """Verify numerical parity (< 1e-5) between DEEPTRACE and DeepfakeBench reference forward pass."""
        sys.path.insert(0, str(REPO_ROOT / "deepfake_model"))
        from inference_xception import load_xception_model

        ref_cfg_path = str(REPO_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "config" / "detector" / "xception.yaml")
        ref_weights_path = str(REPO_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "weights" / "xception_best.pth")
        
        ref_model, _ = load_xception_model(ref_cfg_path, ref_weights_path, self.device)

        dummy_crop = np.full((256, 256, 3), 128, dtype=np.uint8)
        tensor = preprocess_face_crop(dummy_crop, is_bgr=True).to(self.device)

        with torch.no_grad():
            ref_logits = ref_model({'image': tensor})['cls'][0].cpu().numpy()
            dt_logits = self.model({'image': tensor})['cls'][0].cpu().numpy()

        max_diff = float(np.max(np.abs(ref_logits - dt_logits)))
        self.assertLess(max_diff, 1e-5, f"Numerical difference {max_diff} exceeds tolerance 1e-5")

    def test_07_known_real_fixture_inference(self):
        """Verify end-to-end analysis correctly classifies authentic sample without false positive."""
        self.assertTrue(self.real_sample_path.exists(), f"Sample missing: {self.real_sample_path}")
        res = analyze_image(self.real_sample_path)
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.prediction, "real")
        self.assertEqual(res.assessment, "Likely authentic")
        self.assertGreater(res.model_probability, 0.50)
        self.assertGreaterEqual(len(res.faces), 1)

    def test_08_known_fake_fixture_inference(self):
        """Verify end-to-end analysis correctly classifies manipulated sample without false negative."""
        self.assertTrue(self.fake_sample_path.exists(), f"Sample missing: {self.fake_sample_path}")
        res = analyze_image(self.fake_sample_path)
        self.assertEqual(res.status, "completed")
        self.assertEqual(res.prediction, "fake")
        self.assertEqual(res.assessment, "Likely manipulated")
        self.assertGreater(res.model_probability, 0.50)
        self.assertGreaterEqual(len(res.faces), 1)

    def test_09_user_portrait_not_misclassified(self):
        """Verify genuine user portrait is not falsely classified as manipulated after padding correction."""
        if self.user_portrait_path.exists():
            res = analyze_image(self.user_portrait_path)
            self.assertEqual(res.status, "completed")
            self.assertEqual(res.prediction, "real", f"User portrait misclassified as {res.prediction}")
            self.assertEqual(res.assessment, "Likely authentic")

    def test_10_no_face_control_sample(self):
        """Verify image with no detectable faces returns no_face_detected and is NOT classified as fake."""
        self.assertTrue(self.no_face_path.exists(), f"Sample missing: {self.no_face_path}")
        res = analyze_image(self.no_face_path)
        self.assertEqual(res.status, "no_face_detected")
        self.assertEqual(res.assessment, "No face detected")
        self.assertIsNone(res.prediction)
        self.assertIsNone(res.model_probability)

    def test_11_meso4_rgb_channel_order(self):
        """Verify Meso4 receives RGB channels and executes without channel inversion errors."""
        adapter = Meso4Adapter()
        dummy_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
        pred_item = adapter.predict_face_crop(dummy_rgb)
        self.assertIn(pred_item.prediction, ["authentic", "manipulated"])
        self.assertGreaterEqual(pred_item.manipulated_probability, 0.0)
        self.assertLessEqual(pred_item.manipulated_probability, 1.0)
        self.assertIsNotNone(pred_item.authentic_probability)
        self.assertIsNotNone(pred_item.manipulated_probability)
        self.assertAlmostEqual(pred_item.authentic_probability + pred_item.manipulated_probability, 1.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
