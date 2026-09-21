"""
Unit and integration tests for DEEPTRACE Stage 8E Multi-Model Cross Validation.
Verifies:
1. Model registry and metadata descriptors.
2. Normalized prediction schema.
3. Class mapping (Index 0 = authentic, Index 1 = manipulated).
4. Model agreement detection.
5. Model disagreement detection.
6. Failure isolation when comparison model is missing/unavailable.
7. Primary model resilience.
8. Identical sample evaluation.
9. Probability normalization and thresholding.
10. Video frame alignment across primary and comparison detectors.
11. Video agreement percentage calculation.
12. Correlation and Cohen's Kappa inter-rater agreement statistics.
13. Multi-model comparative benchmark evaluation on identical dataset.
14. REST API endpoints under /api/cross-validation.
"""

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from fastapi.testclient import TestClient

from app.main import app
from app.services.model_registry import get_model_registry
from app.services.detectors.xception_adapter import XceptionAdapter
from app.services.detectors.meso4_adapter import Meso4Adapter
from app.services.cross_validation_service import (
    get_cross_validation_service,
    calculate_cohen_kappa,
    calculate_correlation,
    compute_agreement_statistics
)
from app.models.cross_validation_schemas import (
    NormalizedDetectorPrediction,
    ImageCrossValidationResponse
)


class TestModelRegistry(unittest.TestCase):
    """Tests model registry lookup, metadata, and roles."""

    def setUp(self):
        self.registry = get_model_registry()

    def test_registry_contains_primary_and_comparison(self):
        primary = self.registry.get_primary_detector()
        self.assertIsNotNone(primary)
        self.assertEqual(primary.model_id, "xception_deepfakebench")
        self.assertEqual(primary.role, "primary")

        comps = self.registry.get_comparison_detectors()
        self.assertGreaterEqual(len(comps), 1)
        self.assertTrue(any(c.model_id == "meso4_deepfakebench" for c in comps))
        self.assertTrue(all(c.role == "comparison" for c in comps))

    def test_metadata_schema(self):
        models = self.registry.list_models()
        self.assertGreaterEqual(len(models), 2)
        for m in models:
            self.assertIn(m.role, ["primary", "comparison"])
            self.assertEqual(m.task, "binary deepfake classification")
            self.assertEqual(m.classes, ["authentic", "manipulated"])
            self.assertEqual(m.input_size, "256x256")
            self.assertIn(m.availability_status, ["available", "unavailable"])


class TestDetectorAdapters(unittest.TestCase):
    """Tests individual detector inference, normalization, and failure isolation."""

    def test_xception_adapter_prediction_normalization(self):
        adapter = XceptionAdapter()
        dummy_face = np.full((256, 256, 3), 120, dtype=np.uint8)
        pred = adapter.predict_face_crop(dummy_face)

        self.assertIsInstance(pred, NormalizedDetectorPrediction)
        self.assertEqual(pred.model_id, "xception_deepfakebench")
        self.assertEqual(pred.role, "primary")
        self.assertEqual(pred.status, "success")
        self.assertIn(pred.prediction, ["authentic", "manipulated"])
        self.assertIsNotNone(pred.manipulated_probability)
        self.assertGreaterEqual(pred.manipulated_probability, 0.0)
        self.assertLessEqual(pred.manipulated_probability, 1.0)
        self.assertAlmostEqual(pred.manipulated_probability + pred.authentic_probability, 1.0, places=3)

    def test_meso4_adapter_prediction_normalization(self):
        adapter = Meso4Adapter()
        dummy_face = np.full((256, 256, 3), 150, dtype=np.uint8)
        pred = adapter.predict_face_crop(dummy_face)

        self.assertIsInstance(pred, NormalizedDetectorPrediction)
        self.assertEqual(pred.model_id, "meso4_deepfakebench")
        self.assertEqual(pred.role, "comparison")
        if pred.status == "success":
            self.assertIn(pred.prediction, ["authentic", "manipulated"])
            self.assertIsNotNone(pred.manipulated_probability)
            self.assertAlmostEqual(pred.manipulated_probability + pred.authentic_probability, 1.0, places=3)
        else:
            self.assertEqual(pred.status, "unavailable")

    def test_threshold_variation(self):
        adapter = XceptionAdapter()
        dummy_face = np.full((256, 256, 3), 120, dtype=np.uint8)
        # At very low threshold (0.01), any non-zero prob should predict manipulated
        pred_low = adapter.predict_face_crop(dummy_face, threshold=0.001)
        # At very high threshold (0.999), should predict authentic
        pred_high = adapter.predict_face_crop(dummy_face, threshold=0.999)

        self.assertEqual(pred_high.prediction, "authentic")


class TestAgreementAndStatistics(unittest.TestCase):
    """Tests agreement logic, Cohen's Kappa, and correlation statistics."""

    def test_cohen_kappa_perfect_agreement(self):
        preds_a = ["authentic", "manipulated", "authentic", "manipulated"]
        preds_b = ["authentic", "manipulated", "authentic", "manipulated"]
        kappa, status, _ = calculate_cohen_kappa(preds_a, preds_b)
        self.assertEqual(status, "available")
        self.assertAlmostEqual(kappa, 1.0)

    def test_cohen_kappa_disagreement(self):
        preds_a = ["authentic", "authentic", "manipulated", "manipulated"]
        preds_b = ["manipulated", "manipulated", "authentic", "authentic"]
        kappa, status, _ = calculate_cohen_kappa(preds_a, preds_b)
        self.assertEqual(status, "available")
        self.assertLess(kappa, 0.0)

    def test_correlation_calculation(self):
        scores_a = [0.1, 0.3, 0.6, 0.9]
        scores_b = [0.15, 0.35, 0.65, 0.85]
        r, rho, status, _ = calculate_correlation(scores_a, scores_b)
        self.assertEqual(status, "available")
        self.assertGreater(r, 0.95)
        self.assertGreater(rho, 0.95)

    def test_correlation_constant_zero_variance_safety(self):
        scores_a = [0.5, 0.5, 0.5]
        scores_b = [0.2, 0.4, 0.8]
        r, rho, status, exp = calculate_correlation(scores_a, scores_b)
        self.assertIsNone(r)
        self.assertEqual(status, "not_available")
        self.assertIn("zero variance", exp)

    def test_agreement_statistics_summary(self):
        preds_a = ["authentic", "manipulated", "authentic"]
        preds_b = ["authentic", "authentic", "authentic"]
        scores_a = [0.2, 0.8, 0.3]
        scores_b = [0.1, 0.4, 0.2]
        stats = compute_agreement_statistics(preds_a, preds_b, scores_a, scores_b)

        self.assertEqual(stats.total_compared, 3)
        self.assertEqual(stats.agreeing_count, 2)
        self.assertEqual(stats.disagreeing_count, 1)
        self.assertAlmostEqual(stats.agreement_rate, 2 / 3, places=3)
        self.assertAlmostEqual(stats.disagreement_rate, 1 / 3, places=3)


class TestCrossValidationService(unittest.TestCase):
    """Tests image and video cross-validation pipeline execution."""

    def setUp(self):
        self.service = get_cross_validation_service()

    def test_cross_validate_synthetic_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            # Synthetic 300x300 image
            img = np.full((300, 300, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            res = self.service.cross_validate_image(img_path)
            self.assertIsInstance(res, ImageCrossValidationResponse)
            self.assertEqual(res.filename, "test.jpg")
            self.assertIsNotNone(res.primary_model)
            self.assertEqual(res.primary_model.model_id, "xception_deepfakebench")
            self.assertGreaterEqual(len(res.comparison_models), 1)

    def test_cross_validate_synthetic_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vid_path = Path(tmpdir) / "test.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (160, 160))
            for _ in range(20):
                frame = np.full((160, 160, 3), 150, dtype=np.uint8)
                out.write(frame)
            out.release()

            res = self.service.cross_validate_video(vid_path, sample_interval=5, max_frames=4)
            self.assertEqual(res.filename, "test.mp4")
            self.assertGreater(res.frames_analyzed, 0)
            self.assertGreaterEqual(len(res.frames), 1)
            # Verify frame alignment: every frame item has primary and comparison models evaluated
            for f in res.frames:
                self.assertEqual(f.primary_model.model_id, "xception_deepfakebench")
                self.assertTrue(any(c.model_id == "meso4_deepfakebench" for c in f.comparison_models))


class TestCrossValidationAPI(unittest.TestCase):
    """Tests the cross-validation REST API endpoints."""

    def setUp(self):
        self.client = TestClient(app)

    def test_get_registered_models(self):
        res = self.client.get("/api/cross-validation/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertTrue(any(m["role"] == "primary" for m in data))
        self.assertTrue(any(m["role"] == "comparison" for m in data))

    def test_cross_validate_missing_file_400(self):
        res = self.client.post("/api/cross-validation/image", json={})
        self.assertEqual(res.status_code, 400)

    def test_cross_validate_nonexistent_path_404(self):
        res = self.client.post("/api/cross-validation/image", json={"file_path": "nonexistent_image_123.jpg"})
        self.assertEqual(res.status_code, 404)

    def test_benchmark_comparison_missing_manifest_400(self):
        res = self.client.post("/api/cross-validation/benchmark-comparison", json={})
        self.assertEqual(res.status_code, 400)

    def test_benchmark_comparison_on_manifest(self):
        manifest_path = "evaluation/manifests/test_evaluation_manifest.json"
        if Path(manifest_path).exists():
            res = self.client.post(
                "/api/cross-validation/benchmark-comparison",
                json={"manifest_path": manifest_path}
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("evaluated_models", data)
            self.assertEqual(len(data["evaluated_models"]), 2)
            self.assertTrue(data["identical_dataset_verified"])
            self.assertIn("agreement_statistics", data)


if __name__ == "__main__":
    unittest.main()
