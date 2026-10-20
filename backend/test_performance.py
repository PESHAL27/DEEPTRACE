"""
DEEPTRACE Stage 9B — Performance & Efficiency Regression Tests
Verifies that optimizations maintain schema validity, memory stability,
concurrency safety, Grad-CAM hook cleanup, and forensic output integrity
without relying on brittle timing thresholds.
"""

import unittest
import threading
import numpy as np
import torch
from pathlib import Path
import sys
import psutil
import os

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.detection_service import (
    analyze_image,
    analyze_video,
    predict_face_crop,
    get_xception_detector
)
from app.services.face_service import detect_and_crop_faces
from app.services.gradcam_service import get_gradcam_engine, generate_gradcam_for_face
from app.services.frequency_service import analyze_frequency_domain
from app.services.report_service import ForensicReportService
from app.models.schemas import AnalysisResponse, DetectionResult, FaceDetectionResult
from app.config import MAX_ANALYZED_FRAMES, DEEPFAKE_THRESHOLD

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
FIXTURE_KNOWN_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
FIXTURE_FAKE_VIDEO = BACKEND_DIR / "test_fixtures" / "test_fake_video.mp4"


class TestPerformanceAndEfficiency(unittest.TestCase):
    """Test suite ensuring performance invariants, resource safety, and output validity."""

    @classmethod
    def setUpClass(cls):
        # Warm up singleton models once
        get_xception_detector()
        cls.process = psutil.Process(os.getpid())

    def test_01_image_pipeline_memory_stability_and_schema(self):
        """Image analysis pipeline completes with valid schema and bounded memory."""
        rss_start = self.process.memory_info().rss
        self.assertTrue(FIXTURE_KNOWN_REAL.exists(), f"Missing fixture: {FIXTURE_KNOWN_REAL}")

        result: DetectionResult = analyze_image(FIXTURE_KNOWN_REAL)

        # Output schema validation
        self.assertEqual(result.status, "completed")
        self.assertIn(result.prediction, ["real", "fake"])
        self.assertIsInstance(result.probability, float)
        self.assertGreaterEqual(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)
        self.assertGreater(len(result.faces), 0)
        self.assertTrue(result.gradcam_available)
        self.assertIsNotNone(result.gradcam_heatmap_url)
        self.assertIsNotNone(result.frequency_analysis)
        self.assertTrue(result.frequency_analysis.available)

        # Memory delta should be reasonably bounded (no memory explosion)
        rss_end = self.process.memory_info().rss
        delta_mb = (rss_end - rss_start) / (1024 * 1024)
        self.assertLess(delta_mb, 400.0, f"Unexpected memory explosion: {delta_mb:.2f} MB")

    def test_02_inference_mode_returns_detached_scalars(self):
        """predict_face_crop returns pure Python scalars with no retained autograd graphs."""
        dummy_crop = np.zeros((256, 256, 3), dtype=np.uint8)
        pred = predict_face_crop(dummy_crop)

        self.assertIn("prediction", pred)
        self.assertIn("probability", pred)
        self.assertIn("real_probability", pred)
        self.assertIn("fake_probability", pred)
        self.assertIsInstance(pred["probability"], float)
        self.assertIsInstance(pred["real_probability"], float)
        self.assertIsInstance(pred["fake_probability"], float)

    def test_03_detection_info_reuse_avoids_duplicate_processing(self):
        """Passing detection_info to analyze_image safely bypasses redundant face detection."""
        detection_info = detect_and_crop_faces(FIXTURE_KNOWN_REAL)
        self.assertGreater(detection_info["count"], 0)
        self.assertIn("source_image_bgr", detection_info)

        # Reuse pre-computed detection_info
        result = analyze_image(FIXTURE_KNOWN_REAL, detection_info=detection_info)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.faces), detection_info["count"])

    def test_04_gradcam_hook_lifecycle_and_cleanup(self):
        """Grad-CAM hooks, activations, and gradient tensors are reliably purged after execution."""
        engine = get_gradcam_engine()
        dummy_face = np.full((128, 128, 3), 120, dtype=np.uint8)

        # Before execution
        self.assertEqual(len(engine.hook_handles), 0)
        self.assertIsNone(engine.activations)
        self.assertIsNone(engine.gradients)

        # Execution
        res = generate_gradcam_for_face(dummy_face, identifier_prefix="test_hook_cleanup", target_class=1)
        self.assertTrue(res.get("gradcam_available"))

        # After execution: hooks and tensors MUST be cleaned up
        self.assertEqual(len(engine.hook_handles), 0)
        self.assertIsNone(engine.activations)
        self.assertIsNone(engine.gradients)

    def test_05_concurrent_inference_isolation(self):
        """Simultaneous multi-threaded inference requests execute with strict isolation."""
        dummy_face_a = np.zeros((128, 128, 3), dtype=np.uint8)
        dummy_face_b = np.full((128, 128, 3), 200, dtype=np.uint8)

        results = {}
        exceptions = []

        def worker(key, crop):
            try:
                out = predict_face_crop(crop)
                results[key] = out
            except Exception as e:
                exceptions.append(e)

        t1 = threading.Thread(target=worker, args=("A", dummy_face_a))
        t2 = threading.Thread(target=worker, args=("B", dummy_face_b))

        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        self.assertEqual(len(exceptions), 0, f"Concurrent thread exceptions: {exceptions}")
        self.assertIn("A", results)
        self.assertIn("B", results)
        self.assertIn("probability", results["A"])
        self.assertIn("probability", results["B"])

    def test_06_video_pipeline_bounded_frames_and_schema(self):
        """Video analysis respects MAX_ANALYZED_FRAMES and produces complete temporal schema."""
        if not FIXTURE_FAKE_VIDEO.exists():
            self.skipTest(f"Video fixture not found: {FIXTURE_FAKE_VIDEO}")

        result: DetectionResult = analyze_video(FIXTURE_FAKE_VIDEO, sample_interval=15, max_frames=MAX_ANALYZED_FRAMES)

        self.assertEqual(result.status, "completed")
        self.assertIn(result.prediction, ["real", "fake"])
        self.assertLessEqual(result.frames_analyzed, MAX_ANALYZED_FRAMES)
        self.assertIsNotNone(result.temporal_analysis)
        self.assertEqual(result.temporal_analysis.status, "available")
        self.assertIsInstance(result.temporal_analysis.frame_series, list)

    def test_07_forensic_report_generation_integrity(self):
        """Report generator creates valid report and HTML representation with SHA-256."""
        from app.models.schemas import MediaMetadata

        dummy_resp = AnalysisResponse(
            analysis_id="DT-TEST-PERF-01",
            filename="known_real.jpg",
            file_size=721954,
            file_size_human="705.0 KB",
            media_type="image",
            media_url="/uploads/test.jpg",
            created_at="Sep 26, 2026",
            status="completed",
            prediction="real",
            confidence_score=62.5,
            model="Xception",
            metadata=MediaMetadata(
                file_name="known_real.jpg",
                file_type="image/jpeg",
                file_size=721954,
                file_size_human="705.0 KB",
                width=1200,
                height=896
            ),
            face_detection=FaceDetectionResult(
                status="completed",
                count=2,
                faces=[]
            ),
            detection=DetectionResult(
                status="completed",
                prediction="real",
                probability=0.6246,
                model_probability=0.6246
            )
        )

        service = ForensicReportService()
        report = service.generate_professional_report(dummy_resp)

        self.assertEqual(report.caseId, "DT-TEST-PERF-01")
        self.assertEqual(len(report.hashSha256), 64)
        html_content = service.render_printable_report_html(report)
        self.assertIn("DT-TEST-PERF-01", html_content)
        self.assertIn(report.hashSha256, html_content)
        self.assertIn("DEEPTRACE", html_content)


if __name__ == "__main__":
    unittest.main()
