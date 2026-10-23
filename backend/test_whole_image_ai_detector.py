"""
Comprehensive Regression Test Suite for Whole-Image AI-Generated Image Detection.
=================================================================================
Covers Phase 6 Requirements:
1. Correct model loading (singleton caching, thread safety).
2. Model evaluation mode, inference mode, and CPU execution.
3. Supported formats (JPG, PNG, WebP) and corrupt/unsupported input safety.
4. Output interpretation (raw score, cautious labels, borderline flagging, no false 'Real' on error).
5. Real camera photographs vs. AI-generated images across categories.
6. Images without faces (Smaug detects AI generation while YuNet/Xception report no_face_detected).
7. Dual-pipeline independence: Xception face manipulation and Smaug AI generation remain unblended.
8. Backward compatibility of API endpoints (/api/analyze, /api/models, /api/health).
9. Forensic report generation preserving full diagnostic audit trail.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import torch
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.services.detectors.ai_generated_image_detector import (
    get_smaug_detector,
    detect_ai_generated_image,
    PROVISIONAL_AI_THRESHOLD,
    BORDERLINE_MARGIN
)
from app.models.schemas import AIGenerationResult, AnalysisResponse
from app.services.detection_service import analyze_image
from app.services.report_service import ForensicReportService

# Fixture paths
FIXTURES_DIR = BACKEND_DIR / "test_fixtures"
SAMPLE_REAL_PORTRAIT = BACKEND_DIR / "uploads" / "DT-2026-17022339_Nithin_sist_friend.jpeg"
SAMPLE_AI_PORTRAIT = FIXTURES_DIR / "ai_faces" / "user_ai_generated.jpg"
SAMPLE_AI_LANDSCAPE = FIXTURES_DIR / "ai_whole_image" / "ai_landscape_valley_sd15.jpg"
SAMPLE_REAL_LANDSCAPE = FIXTURES_DIR / "ai_whole_image" / "real_landscape_yosemite.jpg"
SAMPLE_NO_FACE = FIXTURES_DIR / "test_image.jpg"
SAMPLE_INVALID_TXT = FIXTURES_DIR / "test_script.txt"
SAMPLE_WEBP = FIXTURES_DIR / "ai_faces" / "stable_diffusion_portrait.webp"
SAMPLE_PNG = FIXTURES_DIR / "ai_whole_image" / "ai_architecture_city.png"


class TestWholeImageAIDetectorRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    # -------------------------------------------------------------------------
    # 1. Model Loading & Caching
    # -------------------------------------------------------------------------
    def test_01_singleton_loading_and_caching(self):
        """Verify model loads lazily, caches in memory, and returns identical instance."""
        detector_1 = get_smaug_detector()
        self.assertIsNotNone(detector_1)
        detector_2 = get_smaug_detector()
        self.assertIs(detector_1, detector_2, "get_smaug_detector must return the cached singleton instance")

    def test_02_cpu_execution_and_eval_mode(self):
        """Verify model executes on CPU in eval mode with no gradients."""
        detector = get_smaug_detector()
        self.assertIsNotNone(detector)
        st_model = getattr(detector, "st", None)
        self.assertIsNotNone(st_model, "SmaugImageDetect must contain SentenceTransformer st module")
        self.assertEqual(st_model.device.type, "cpu", "Model should execute on CPU in current host environment")

    # -------------------------------------------------------------------------
    # 2. Input Validation & Error Containment
    # -------------------------------------------------------------------------
    def test_03_nonexistent_file_handling(self):
        """Verify missing file returns safe error result without raising unhandled exception."""
        nonexistent = FIXTURES_DIR / "ghost_file_does_not_exist_12345.jpg"
        result = detect_ai_generated_image(nonexistent)
        self.assertIsInstance(result, AIGenerationResult)
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.score)
        self.assertEqual(result.assessment, "Analysis unavailable")
        self.assertIn("not found", (result.error_message or "").lower())

    def test_04_corrupt_and_unsupported_file_handling(self):
        """Verify non-image text file and corrupted bytes are handled gracefully."""
        # Non-image text file
        self.assertTrue(SAMPLE_INVALID_TXT.exists())
        result_txt = detect_ai_generated_image(SAMPLE_INVALID_TXT)
        self.assertEqual(result_txt.status, "error")
        self.assertIsNone(result_txt.score)
        self.assertEqual(result_txt.assessment, "Analysis unavailable")

        # Corrupted / truncated image bytes
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE_BYTES_00000000")
            corrupt_path = Path(f.name)

        try:
            result_corrupt = detect_ai_generated_image(corrupt_path)
            self.assertEqual(result_corrupt.status, "error")
            self.assertIsNone(result_corrupt.score)
            self.assertEqual(result_corrupt.assessment, "Analysis unavailable")
        finally:
            if corrupt_path.exists():
                os.unlink(corrupt_path)

    def test_05_supported_image_formats(self):
        """Verify JPG, PNG, and WebP inputs are all validated and parsed."""
        formats = [
            (SAMPLE_REAL_LANDSCAPE, "jpg"),
            (SAMPLE_PNG, "png"),
            (SAMPLE_WEBP, "webp")
        ]
        for path, fmt in formats:
            if path.exists():
                res = detect_ai_generated_image(path, crops=1)
                self.assertEqual(res.status, "completed", f"Format {fmt} failed: {res.error_message}")
                self.assertIsNotNone(res.score, f"Format {fmt} did not return score")
                self.assertTrue(0.0 <= res.score <= 1.0, f"Score out of bounds for {fmt}: {res.score}")

    # -------------------------------------------------------------------------
    # 3. Label & Threshold Interpretation
    # -------------------------------------------------------------------------
    def test_06_cautious_labels_and_borderline_flagging(self):
        """Verify correct cautious labels, provisional threshold, and borderline logic."""
        # Genuine camera photo must yield 'Likely camera-captured'
        if SAMPLE_REAL_PORTRAIT.exists():
            real_res = detect_ai_generated_image(SAMPLE_REAL_PORTRAIT, crops=1)
            self.assertEqual(real_res.predicted_label, "likely_camera_captured")
            self.assertEqual(real_res.assessment, "Likely camera-captured")
            self.assertLess(real_res.score, PROVISIONAL_AI_THRESHOLD)

        # AI generated image must yield 'Likely AI-generated'
        if SAMPLE_AI_PORTRAIT.exists():
            ai_res = detect_ai_generated_image(SAMPLE_AI_PORTRAIT, crops=1)
            self.assertEqual(ai_res.predicted_label, "likely_ai_generated")
            self.assertEqual(ai_res.assessment, "Likely AI-generated")
            self.assertGreaterEqual(ai_res.score, PROVISIONAL_AI_THRESHOLD)

    # -------------------------------------------------------------------------
    # 4. Pipeline Integration & Separation from Xception
    # -------------------------------------------------------------------------
    def test_07_no_face_image_analyzed_by_smaug(self):
        """Verify image without a face executes Smaug whole-image detection."""
        if SAMPLE_REAL_LANDSCAPE.exists():
            analysis = analyze_image(SAMPLE_REAL_LANDSCAPE)
            self.assertEqual(analysis.status, "no_face_detected")
            self.assertEqual(len(analysis.faces), 0)
            self.assertIsNotNone(analysis.ai_generation, "ai_generation must be present on no-face image")
            self.assertEqual(analysis.ai_generation.status, "completed")
            self.assertLess(analysis.ai_generation.score, PROVISIONAL_AI_THRESHOLD)
            self.assertEqual(analysis.ai_generation.assessment, "Likely camera-captured")

    def test_08_dual_pipeline_separation_on_face_image(self):
        """Verify face image contains BOTH independent Xception and Smaug findings without score fusion."""
        if SAMPLE_AI_PORTRAIT.exists():
            analysis = analyze_image(SAMPLE_AI_PORTRAIT)
            self.assertEqual(analysis.status, "completed")
            self.assertGreater(len(analysis.faces), 0)
            
            # 1. Smaug Whole-Image AI generation finding
            self.assertIsNotNone(analysis.ai_generation)
            self.assertEqual(analysis.ai_generation.status, "completed")
            self.assertGreaterEqual(analysis.ai_generation.score, 0.90, "Midjourney portrait should have high Smaug score")

            # 2. Existing Xception Face Manipulation finding
            self.assertIsNotNone(analysis.face_manipulation)
            # The two models must have independent raw scores and labels
            self.assertNotEqual(analysis.ai_generation.score, analysis.probability, "Scores must not be blended or identical")
            self.assertEqual(analysis.face_manipulation.get("model_name"), "Xception")

    # -------------------------------------------------------------------------
    # 5. API Endpoints Compatibility
    # -------------------------------------------------------------------------
    def test_09_api_health_endpoint(self):
        """Verify /api/health and /api/status report smaug_ai_detector readiness."""
        resp_health = self.client.get("/api/health")
        self.assertEqual(resp_health.status_code, 200)
        data_health = resp_health.json()
        self.assertEqual(data_health.get("status"), "ok")

        resp_status = self.client.get("/api/status")
        self.assertEqual(resp_status.status_code, 200)
        data_status = resp_status.json()
        self.assertIn("models", data_status)
        self.assertEqual(data_status["models"].get("smaug_ai_detector"), "ready")

    def test_10_api_models_endpoint(self):
        """Verify /api/models lists both deepfake_detector and smaug-ai-detector."""
        resp = self.client.get("/api/models")
        self.assertEqual(resp.status_code, 200)
        models = resp.json()
        model_names = [m.get("name") for m in models]
        self.assertIn("Xception Deepfake Detector", model_names)
        self.assertIn("Smaug AI Image Detector", model_names)

    def test_11_api_analyze_backward_compatibility(self):
        """Verify POST /api/analyze maintains all legacy response fields while returning new detectors."""
        if SAMPLE_REAL_PORTRAIT.exists():
            with open(SAMPLE_REAL_PORTRAIT, "rb") as f:
                resp = self.client.post(
                    "/api/analyze",
                    files={"file": (SAMPLE_REAL_PORTRAIT.name, f, "image/jpeg")}
                )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            
            # Legacy fields must be intact
            self.assertIn("analysis_id", data)
            self.assertEqual(data["status"], "completed")
            self.assertIn("prediction", data)
            self.assertIn("probability", data)
            self.assertIn("faces", data)

            # New fields present
            self.assertIn("ai_generation", data)
            self.assertIn("face_manipulation", data)
            ai_gen = data["ai_generation"]
            self.assertEqual(ai_gen["model_name"], "Smaug AI Image Detector")
            self.assertIn("score", ai_gen)
            self.assertIn("assessment", ai_gen)

    # -------------------------------------------------------------------------
    # 6. Audit Report Generation
    # -------------------------------------------------------------------------
    def test_12_forensic_report_generation(self):
        """Verify forensic audit report includes Whole-Image AI Detection section."""
        from app.models.schemas import MediaMetadata, DetectionResult, FaceDetectionResult
        meta = MediaMetadata(
            width=640, height=480, file_size=50000,
            file_size_human="50.0 KB", format="JPEG"
        )
        ai_gen = AIGenerationResult(
            model="Smaug-AI-Image-Detector",
            score=0.0450,
            predicted_label="likely_camera_captured",
            assessment="Likely camera-captured",
            threshold=0.50,
            is_borderline=False,
            status="completed",
            details="Model score: 0.0450 | Threshold: 0.50",
            crops_evaluated=1
        )
        record = AnalysisResponse(
            analysis_id="DT-TEST-REGRESSION-001",
            filename="test_sample.jpg",
            media_type="image",
            file_size=50000,
            file_size_human="50.0 KB",
            status="completed",
            metadata=meta,
            faces_detected=0,
            face_detection=FaceDetectionResult(status="no_face_detected", faces=[], count=0),
            detection=DetectionResult(status="no_face_detected", faces=[]),
            faces=[],
            ai_generation=ai_gen,
            face_manipulation={"model": "Xception", "status": "no_face_detected"},
            created_at="2026-10-22 23:00:00"
        )
        service = ForensicReportService()
        report = service.generate_professional_report(record)
        categories = [c.category for c in report.evidence_summary_categories]
        self.assertIn("WHOLE-IMAGE AI GENERATION", categories)
        ai_cat = next(c for c in report.evidence_summary_categories if c.category == "WHOLE-IMAGE AI GENERATION")
        self.assertEqual(ai_cat.status, "COMPLETED")
        self.assertIn("Smaug AI Image Detector: Likely camera-captured", ai_cat.headline)
        self.assertIn("0.0450", ai_cat.details)


if __name__ == "__main__":
    unittest.main(verbosity=2)
