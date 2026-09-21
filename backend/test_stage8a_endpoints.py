import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import sys

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.api.analysis import analysis_store

SAMPLE_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
SAMPLE_FAKE = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_fake.jpg"
SAMPLE_FAKE_VIDEO = BACKEND_DIR / "test_fixtures" / "test_fake_video.mp4"
SAMPLE_ZERO_FACE_VIDEO = BACKEND_DIR / "test_fixtures" / "test_video.mp4"


class TestStage8AEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_get_models_live_telemetry(self):
        """Verify GET /api/models returns live runtime information for the loaded Xception detector."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)

        # Primary live Xception detector
        xception = next((m for m in models if m["id"] == "xception-detector"), None)
        self.assertIsNotNone(xception, "Missing 'xception-detector' in /api/models response")
        self.assertEqual(xception["name"], "Xception Deepfake Detector")
        self.assertEqual(xception["architecture"], "Xception (DeepfakeBench)")
        self.assertEqual(xception["checkpoint"], "xception_best.pth")
        self.assertEqual(xception["status"], "Ready")
        self.assertEqual(xception["inputResolution"], "256x256")
        self.assertEqual(xception["mean"], [0.5, 0.5, 0.5])
        self.assertEqual(xception["std"], [0.5, 0.5, 0.5])
        self.assertEqual(xception["numClasses"], 2)
        self.assertTrue("M" in xception["parameters"], f"Unexpected parameter string: {xception['parameters']}")
        self.assertTrue("CPU" in xception["device"] or "CUDA" in xception["device"], f"Unexpected device: {xception['device']}")

    def test_02_get_reports_initial_and_populated(self):
        """Verify GET /api/reports and GET /api/reports/{id} using real analyses."""
        # 1. Ingest an actual image analysis
        with open(SAMPLE_REAL, "rb") as f:
            upload_res = self.client.post("/api/analyze", files={"file": ("test_audit_real.jpg", f, "image/jpeg")})
        self.assertEqual(upload_res.status_code, 200)
        analysis_data = upload_res.json()
        case_id = analysis_data["analysis_id"]

        # 2. Query /api/reports
        rep_res = self.client.get("/api/reports")
        self.assertEqual(rep_res.status_code, 200)
        reports = rep_res.json()
        self.assertIsInstance(reports, list)
        self.assertGreaterEqual(len(reports), 1)

        target_rep = next((r for r in reports if r["caseId"] == case_id), None)
        self.assertIsNotNone(target_rep, f"Case {case_id} not found in reports")
        self.assertEqual(target_rep["id"], f"REP-{case_id}")
        self.assertEqual(target_rep["filename"], "test_audit_real.jpg")
        self.assertEqual(target_rep["mediaType"], "image")
        self.assertEqual(target_rep["result"], "REAL")
        self.assertGreater(len(target_rep["hashSha256"]), 32)
        self.assertTrue(len(target_rep["summary"]) > 0)
        self.assertIn("Xception", target_rep["methodology"])

        # 3. Query /api/reports/{report_id}
        single_res = self.client.get(f"/api/reports/REP-{case_id}")
        self.assertEqual(single_res.status_code, 200)
        single_data = single_res.json()
        self.assertEqual(single_data["caseId"], case_id)

        # 4. Query 404 for invalid report
        not_found_res = self.client.get("/api/reports/REP-NONEXISTENT-999")
        self.assertEqual(not_found_res.status_code, 404)

    def test_03_video_face_detection_status(self):
        """Verify video analysis reports actual face detector status, never 'face_detector_not_loaded'."""
        with open(SAMPLE_FAKE_VIDEO, "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("audit_fake_video.mp4", f, "video/mp4")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        face_detection = data.get("face_detection", {})
        self.assertNotEqual(face_detection.get("status"), "face_detector_not_loaded")
        self.assertEqual(face_detection.get("status"), "completed")
        self.assertGreaterEqual(face_detection.get("count", 0), 1)

    def test_04_zero_face_video_status(self):
        """Verify video analysis on video without faces returns 'no_face_detected'."""
        with open(SAMPLE_ZERO_FACE_VIDEO, "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("audit_zero_video.mp4", f, "video/mp4")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        face_detection = data.get("face_detection", {})
        self.assertNotEqual(face_detection.get("status"), "face_detector_not_loaded")
        self.assertEqual(face_detection.get("status"), "no_face_detected")
        self.assertEqual(data.get("status"), "no_face_detected")


if __name__ == "__main__":
    unittest.main()
