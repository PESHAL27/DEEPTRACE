"""
Security, Reliability & Failure-Handling Test Suite for DEEPTRACE Forensics.
Verifies all 25 core security requirements of Stage 9A.
"""

import os
import io
import sys
import time
import json
import re
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
from PIL import Image

# Setup path
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.config import (
    UPLOADS_DIR,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    EVALUATION_DIR,
    MANIFESTS_DIR
)
from app.utils.file_utils import (
    sanitize_filename,
    validate_file_content,
    sanitize_error_message,
    is_safe_path
)
from app.models.schemas import (
    AnalysisResponse,
    MediaMetadata,
    FaceBoundingBox,
    FaceAnalysisItem,
    FaceDetectionResult,
    DetectionResult,
    ModelDetails
)
from app.models.report_schemas import DetailedForensicReport
from app.services.report_service import get_report_service

SAMPLE_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
SAMPLE_NO_FACE = BACKEND_DIR / "test_fixtures" / "test_image.jpg"


class TestSecurityReliabilitySuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()
        cls.test_fixtures_dir = BACKEND_DIR / "test_fixtures"
        cls.test_fixtures_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    # 1. Path traversal filename
    def test_01_path_traversal_filename(self):
        """User-controlled filename with ../ or ..\\ is strictly sanitized to a basename."""
        traversal_names = [
            "../../evil.jpg",
            "..\\..\\evil.jpg",
            "../../../backend/app/main.py.jpg",
            "....//....//evil.jpg",
            "safe/../../evil.png"
        ]
        for name in traversal_names:
            safe = sanitize_filename(name)
            self.assertNotIn("/", safe)
            self.assertNotIn("\\", safe)
            self.assertNotIn("..", safe)
            self.assertTrue(safe.endswith((".jpg", ".png")))

    # 2. Absolute path filename
    def test_02_absolute_path_filename(self):
        """Absolute Windows and POSIX filenames are stripped down to safe basenames."""
        abs_names = [
            r"C:\Windows\System32\cmd.exe.jpg",
            r"D:\Secret\Project\photo.png",
            "/etc/passwd.jpg",
            "/var/log/syslog.png"
        ]
        for name in abs_names:
            safe = sanitize_filename(name)
            self.assertNotIn(":", safe)
            self.assertNotIn("/", safe)
            self.assertNotIn("\\", safe)
            self.assertTrue(safe.endswith((".jpg", ".png")))

    # 3. Unsupported file extension
    def test_03_unsupported_extension(self):
        """Uploading unsupported extension (.exe, .txt, .sh) is rejected with HTTP 415."""
        res = self.client.post(
            "/api/analyze",
            files={"file": ("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/x-msdownload")}
        )
        self.assertEqual(res.status_code, 415)
        self.assertIn("Unsupported file format", res.json()["detail"])

    # 4. Wrong MIME/content
    def test_04_wrong_content_spoofing(self):
        """Uploading a shell script or Windows PE disguised as .jpg is rejected with HTTP 400."""
        # Spoofed with shell script
        res_sh = self.client.post(
            "/api/analyze",
            files={"file": ("exploit.jpg", b"#!/bin/bash\nrm -rf /", "image/jpeg")}
        )
        self.assertEqual(res_sh.status_code, 400)
        self.assertIn("Dangerous or invalid content", res_sh.json()["detail"])

        # Spoofed with Windows PE
        res_pe = self.client.post(
            "/api/analyze",
            files={"file": ("exploit.jpg", b"MZ\x00\x00\x00\x00\x00\x00", "image/jpeg")}
        )
        self.assertEqual(res_pe.status_code, 400)

    # 5. Corrupt image
    def test_05_corrupt_image(self):
        """Corrupted/truncated image bytes are rejected with HTTP 422 without crash."""
        corrupt_jpg = b"\xFF\xD8\xFF" + b"\x00" * 30  # Invalid truncated JPEG stream
        res = self.client.post(
            "/api/analyze",
            files={"file": ("corrupt.jpg", corrupt_jpg, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("detail", res.json())
        self.assertNotIn("Traceback", res.text)

    # 6. Corrupt video
    def test_06_corrupt_video(self):
        """Corrupted video bytes are cleanly rejected with HTTP 422."""
        corrupt_mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\xFF" * 100
        res = self.client.post(
            "/api/analyze",
            files={"file": ("corrupt.mp4", corrupt_mp4, "video/mp4")}
        )
        self.assertEqual(res.status_code, 422)
        self.assertNotIn("Traceback", res.text)

    # 7. Oversized image
    def test_07_oversized_image(self):
        """Images exceeding MAX_IMAGE_SIZE_BYTES (20 MB) are rejected with HTTP 413."""
        # Create a header-valid JPEG stream larger than 20MB
        header = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        oversized_data = header + b"\x00" * (20 * 1024 * 1024 + 1024)
        res = self.client.post(
            "/api/analyze",
            files={"file": ("large_image.jpg", oversized_data, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds maximum allowed size", res.json()["detail"])

    # 8. Oversized video
    def test_08_oversized_video(self):
        """Videos exceeding MAX_VIDEO_SIZE_BYTES (100 MB) are rejected with HTTP 413."""
        header = b"\x00\x00\x00\x18ftypmp42"
        # Mock streaming size check with an explicit Content-Length or streaming simulation
        # Using a 101MB mock stream or directly testing endpoint rejection
        from app.config import MAX_VIDEO_SIZE_BYTES
        res = self.client.post(
            "/api/analyze",
            files={"file": ("large_video.mp4", header + b"\x00" * (MAX_VIDEO_SIZE_BYTES + 5000), "video/mp4")}
        )
        self.assertEqual(res.status_code, 413)
        self.assertIn("exceeds maximum allowed size", res.json()["detail"])

    # 9. Excessive image dimensions / decompression bomb
    def test_09_excessive_image_dimensions(self):
        """Image dimensions exceeding MAX_IMAGE_DIMENSION (8192px) are rejected with HTTP 422."""
        from app.services.media_service import extract_image_metadata
        # Create a temporary image with dimensions 8200 x 100
        big_img_path = self.test_fixtures_dir / "excessive_dim.png"
        img = Image.new("RGB", (8200, 100), color=(255, 0, 0))
        img.save(big_img_path)

        with self.assertRaises(ValueError) as ctx:
            extract_image_metadata(big_img_path)
        self.assertIn("exceed maximum allowed dimension", str(ctx.exception))

        if big_img_path.exists():
            big_img_path.unlink()

    # 10. Excessive video resolution
    def test_10_excessive_video_resolution(self):
        """Video resolution exceeding MAX_VIDEO_DIMENSION (4096px) raises clean ValueError."""
        from app.services.media_service import extract_video_metadata
        # Mock cv2.VideoCapture returning 5000x5000
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 5000,
                cv2.CAP_PROP_FRAME_HEIGHT: 5000,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 30
            }.get(prop, 0)
            mock_cap.read.return_value = (True, np.zeros((10, 10, 3)))
            mock_cap_cls.return_value = mock_cap

            fake_video_path = self.test_fixtures_dir / "dummy.mp4"
            fake_video_path.write_bytes(b"dummy")
            try:
                with self.assertRaises(ValueError) as ctx:
                    extract_video_metadata(fake_video_path)
                self.assertIn("exceeds maximum allowed dimension", str(ctx.exception))
            finally:
                if fake_video_path.exists():
                    fake_video_path.unlink()

    # 11. Excessive video duration
    def test_11_excessive_video_duration(self):
        """Video duration exceeding MAX_VIDEO_DURATION_SECONDS (600s) raises clean ValueError."""
        from app.services.media_service import extract_video_metadata
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720,
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_COUNT: 30000  # 1000 seconds > 600s
            }.get(prop, 0)
            mock_cap.read.return_value = (True, np.zeros((10, 10, 3)))
            mock_cap_cls.return_value = mock_cap

            fake_video_path = self.test_fixtures_dir / "long_video.mp4"
            fake_video_path.write_bytes(b"dummy")
            try:
                with self.assertRaises(ValueError) as ctx:
                    extract_video_metadata(fake_video_path)
                self.assertIn("exceeds maximum allowed duration", str(ctx.exception))
            finally:
                if fake_video_path.exists():
                    fake_video_path.unlink()

    # 12. Invalid report ID
    def test_12_invalid_report_id(self):
        """Nonexistent report ID returns HTTP 404; malformed ID returns HTTP 400."""
        # Non-existent ID
        res_404 = self.client.get("/api/reports/REP-NONEXISTENT-99999")
        self.assertEqual(res_404.status_code, 404)

        # Malformed ID with invalid characters
        res_400 = self.client.get("/api/reports/REP-bad!@#$ID")
        self.assertEqual(res_400.status_code, 400)

    # 13. Invalid analysis ID
    def test_13_invalid_analysis_id(self):
        """Nonexistent analysis ID returns HTTP 404; malformed ID returns HTTP 400."""
        res_404 = self.client.get("/api/analysis/DT-2026-999999")
        self.assertEqual(res_404.status_code, 404)

        res_400 = self.client.get("/api/analysis/DT-bad%20id!@#")
        self.assertEqual(res_400.status_code, 400)

    # 14. Report path traversal
    def test_14_report_path_traversal(self):
        """Path traversal patterns in report ID are rejected immediately."""
        traversal_ids = [
            "REP-../../etc/passwd",
            "REP-..\\..\\config",
            "REP-....//....//something"
        ]
        for tid in traversal_ids:
            res = self.client.get(f"/api/reports/{tid}")
            self.assertIn(res.status_code, (400, 404))

    # 15. Static artifact traversal
    def test_15_static_artifact_traversal(self):
        """Static uploads endpoint rejects directory traversal outside uploads directory."""
        res = self.client.get("/uploads/../main.py")
        self.assertEqual(res.status_code, 404)

        res_enc = self.client.get("/uploads/..%2F..%2Fapp%2Fmain.py")
        self.assertEqual(res_enc.status_code, 404)

    # 16. HTML escaping in printable report
    def test_16_html_escaping_in_printable_report(self):
        """HTML and JavaScript injection tags in filename and metadata are escaped in printable HTML."""
        service = get_report_service()
        # Build mock analysis response with XSS vectors
        xss_name = "evidence_<script>alert(1)<script>.jpg"
        xss_summary = "<img src=x onerror=alert(1)>"
        meta = MediaMetadata(width=640, height=480, format="JPEG", file_size=1024, file_size_human="1.0 KB")
        face_det = FaceDetectionResult(status="completed", count=1, faces=[FaceBoundingBox(x=10, y=10, width=50, height=50)])
        det = DetectionResult(status="completed", model=ModelDetails(name="Xception", source="DeepfakeBench"), prediction="fake", probability=0.95)
        mock_record = AnalysisResponse(
            analysis_id="DT-2026-XSS001",
            filename=xss_name,
            media_type="image",
            file_size=1024,
            file_size_human="1.0 KB",
            status="completed",
            metadata=meta,
            face_detection=face_det,
            detection=det,
            faces_detected=1,
            prediction="fake",
            probability=0.95,
            detection_probability=0.95,
            model_probability=0.95,
            assessment="Likely manipulated",
            created_at="Sep 21, 2026 • 12:00"
        )
        report = service.generate_professional_report(mock_record)
        report.executive_summary = xss_summary
        html_out = service.render_printable_report_html(report)

        # Verify raw script tags are NOT present
        self.assertNotIn("<script>alert(1)<script>", html_out)
        self.assertNotIn("<img src=x onerror=alert(1)>", html_out)
        # Verify escaped entities ARE present
        self.assertIn("&lt;script&gt;alert(1)&lt;script&gt;", html_out)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html_out)

    # 17. Missing artifact graceful handling
    def test_17_missing_artifact_graceful_handling(self):
        """Forensic report loads and renders successfully even if physical media file is deleted."""
        service = get_report_service()
        meta = MediaMetadata(width=640, height=480, format="JPEG", file_size=5000, file_size_human="5.0 KB")
        face_det = FaceDetectionResult(status="no_face_detected", count=0, faces=[])
        det = DetectionResult(status="no_face_detected", model=ModelDetails(name="Xception", source="DeepfakeBench"), prediction=None, probability=None)
        mock_record = AnalysisResponse(
            analysis_id="DT-2026-DELETED001",
            filename="deleted_media.jpg",
            media_type="image",
            file_size=5000,
            file_size_human="5.0 KB",
            status="no_face_detected",
            metadata=meta,
            face_detection=face_det,
            detection=det,
            faces_detected=0,
            prediction="no_face_detected",
            probability=None,
            assessment="No face detected",
            created_at="Sep 21, 2026 • 12:00"
        )
        # Ensure file does NOT exist on disk
        target = UPLOADS_DIR / f"{mock_record.analysis_id}_{mock_record.filename}"
        if target.exists():
            target.unlink()

        report = service.generate_professional_report(mock_record)
        self.assertIsNotNone(report.hashSha256)
        self.assertEqual(len(report.hashSha256), 64)
        # Verify printable report renders without error
        html_out = service.render_printable_report_html(report)
        self.assertIn(mock_record.analysis_id, html_out)

    # 18. Model failure handling
    def test_18_model_failure_handling(self):
        """Model inference exceptions result in structured 500 error without exposing stack traces."""
        with patch("app.services.detection_service.predict_face_crop") as mock_predict:
            mock_predict.side_effect = RuntimeError("Internal CUDA matrix computation fault")
            with open(SAMPLE_REAL, "rb") as f:
                res = self.client.post(
                    "/api/analyze",
                    files={"file": ("test_model_fail.jpg", f, "image/jpeg")}
                )
            self.assertEqual(res.status_code, 500)
            self.assertIn("Forensic analysis failed", res.json()["detail"])
            self.assertNotIn("Traceback", res.text)
            self.assertNotIn('File "', res.text)

    # 19. Face detector failure handling
    def test_19_face_detector_failure_handling(self):
        """Face detector failure raises handled 500 error without corrupting application state."""
        with patch("app.api.analysis.detect_faces") as mock_detect:
            mock_detect.side_effect = RuntimeError("YuNet OpenCV ONNX buffer parse failure")
            with open(SAMPLE_REAL, "rb") as f:
                res = self.client.post(
                    "/api/analyze",
                    files={"file": ("test_face_fail.jpg", f, "image/jpeg")}
                )
            self.assertEqual(res.status_code, 500)
            self.assertIn("Forensic analysis failed", res.json()["detail"])

    # 20. Concurrent analysis isolation
    def test_20_concurrent_analysis_isolation(self):
        """Concurrent requests are processed safely without cross-contaminating analysis IDs."""
        def run_upload(file_id: int):
            with open(SAMPLE_NO_FACE, "rb") as f:
                return self.client.post(
                    "/api/analyze",
                    files={"file": (f"test_conc_{file_id}.jpg", f, "image/jpeg")}
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(run_upload, 1)
            fut2 = executor.submit(run_upload, 2)
            res1 = fut1.result()
            res2 = fut2.result()

        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res2.status_code, 200)
        data1 = res1.json()
        data2 = res2.json()

        # Unique analysis IDs
        self.assertNotEqual(data1["analysis_id"], data2["analysis_id"])
        self.assertIn("test_conc_1", data1["filename"])
        self.assertIn("test_conc_2", data2["filename"])

    # 21. Temporary file cleanup
    def test_21_temporary_file_cleanup(self):
        """Failed analyses and uploads immediately delete temporary staging files."""
        corrupt_jpg = b"\xFF\xD8\xFF\x00\x00\x00corrupt"
        unique_name = f"cleanup_test_{int(time.time())}.jpg"
        res = self.client.post(
            "/api/analyze",
            files={"file": (unique_name, corrupt_jpg, "image/jpeg")}
        )
        self.assertEqual(res.status_code, 422)

        # Verify no file with unique_name exists in UPLOADS_DIR
        matches = list(UPLOADS_DIR.glob(f"*{unique_name}*"))
        self.assertEqual(len(matches), 0, f"Temporary file was not cleaned up: {matches}")

    # 22. API error response does not contain traceback
    def test_22_api_error_response_no_traceback(self):
        """API error responses must NEVER contain Python traceback keywords."""
        res = self.client.post(
            "/api/analyze",
            files={"file": ("corrupt.jpg", b"garbage_data", "image/jpeg")}
        )
        body = res.text
        self.assertNotIn("Traceback (most recent call last)", body)
        self.assertNotIn("app/api/analysis.py", body)
        self.assertNotIn(".py\", line", body)

    # 23. API response does not expose absolute paths
    def test_23_api_response_no_absolute_paths(self):
        """API responses must NEVER expose Windows C:\\ or POSIX /home/ internal paths."""
        res = self.client.post(
            "/api/analyze",
            files={"file": ("corrupt.jpg", b"\xFF\xD8\xFF" + b"\x00" * 20, "image/jpeg")}
        )
        body = res.text
        # Check Windows absolute path regex
        self.assertFalse(bool(re.search(r"[a-zA-Z]:\\[Uu]sers\\", body)), f"Absolute Windows path found in: {body}")
        # Check Unix path regex
        self.assertFalse(bool(re.search(r"/(?:home|root|tmp)/", body)), f"Absolute Unix path found in: {body}")

    # 24. Benchmark arbitrary-path protection
    def test_24_benchmark_arbitrary_path_protection(self):
        """Benchmark manifest resolution prevents arbitrary system file reads outside approved dirs."""
        # Attempting to access an arbitrary path outside evaluation/datasets/manifests
        res = self.client.post(
            "/api/benchmark/run",
            json={"manifest_path": "C:\\Windows\\System32\\drivers\\etc\\hosts"}
        )
        self.assertIn(res.status_code, (400, 403, 404))
        if res.status_code == 403:
            self.assertIn("forbidden", res.json()["detail"].lower())

    # 25. Cross-validation failure isolation
    def test_25_cross_validation_failure_isolation(self):
        """Failure in comparison detector (MesoNet-4) does not break primary Xception detection."""
        from app.services.cross_validation_service import get_cross_validation_service
        from app.services.model_registry import get_model_registry

        cv_service = get_cross_validation_service()
        registry = get_model_registry()
        meso_det = registry.get_detector("meso4_deepfakebench")

        # Temporarily mock meso4 to throw an error
        with patch.object(meso_det, "predict_face_crop", side_effect=RuntimeError("Meso4 tensor failure")):
            res = cv_service.cross_validate_image(SAMPLE_REAL)
            # Primary detector should still succeed
            self.assertEqual(res.primary_model.model_id, "xception_deepfakebench")
            self.assertIn(res.primary_model.prediction, ("authentic", "manipulated"))
            # Comparison detector logs error gracefully
            meso_eval = next(c for c in res.comparison_models if c.model_id == "meso4_deepfakebench")
            self.assertEqual(meso_eval.status, "error")
            self.assertIn("Meso4 tensor failure", meso_eval.error_message)


if __name__ == "__main__":
    unittest.main()
