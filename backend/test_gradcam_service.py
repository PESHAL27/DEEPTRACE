import unittest
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient
import sys

BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.config import HEATMAPS_DIR
from app.services.face_service import detect_and_crop_faces
from app.services.gradcam_service import XceptionGradCAM, generate_gradcam_for_face

SAMPLE_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
SAMPLE_FAKE = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_fake.jpg"
SAMPLE_NO_FACE = BACKEND_DIR / "test_fixtures" / "test_image.jpg"
SAMPLE_FAKE_VIDEO = BACKEND_DIR / "test_fixtures" / "test_fake_video.mp4"


class TestGradCAMService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        assert SAMPLE_REAL.exists(), f"Missing {SAMPLE_REAL}"
        assert SAMPLE_FAKE.exists(), f"Missing {SAMPLE_FAKE}"
        assert SAMPLE_NO_FACE.exists(), f"Missing {SAMPLE_NO_FACE}"
        assert SAMPLE_FAKE_VIDEO.exists(), f"Missing {SAMPLE_FAKE_VIDEO}"

    def test_01_gradcam_engine_initialization(self):
        """Verify XceptionGradCAM connects to the active singleton and targets conv4."""
        cam_engine = XceptionGradCAM()
        self.assertIsNotNone(cam_engine.model)
        self.assertIsNotNone(cam_engine.target_layer)
        self.assertEqual(cam_engine.target_layer, cam_engine.model.backbone.conv4)

    def test_02_generate_cam_for_real_and_fake_face_crops(self):
        """Verify Grad-CAM forward and backward passes capture activations and gradients."""
        cam_engine = XceptionGradCAM()

        for sample_path in [SAMPLE_REAL, SAMPLE_FAKE]:
            det_res = detect_and_crop_faces(sample_path)
            self.assertGreater(det_res["count"], 0)
            crop = det_res["crops"][0]

            overlay, heatmap = cam_engine.generate_cam(crop, target_class=1)
            self.assertEqual(overlay.shape, crop.shape)
            self.assertEqual(heatmap.shape, crop.shape)
            self.assertEqual(overlay.dtype, np.uint8)
            self.assertEqual(heatmap.dtype, np.uint8)

            # Ensure hooks were cleanly detached
            self.assertEqual(len(cam_engine.hook_handles), 0)

    def test_03_generate_gradcam_file_persistence(self):
        """Verify generate_gradcam_for_face saves overlay, heatmap, and crop artifacts to disk."""
        det_res = detect_and_crop_faces(SAMPLE_FAKE)
        crop = det_res["crops"][0]

        prefix = "unit_test_gradcam"
        result = generate_gradcam_for_face(crop, identifier_prefix=prefix, target_class=1)

        self.assertTrue(result["gradcam_available"])
        self.assertIsNotNone(result["gradcam_overlay_url"])
        self.assertIsNotNone(result["gradcam_heatmap_url"])
        self.assertIsNotNone(result["original_face_crop_url"])

        # Check physical files exist
        overlay_file = HEATMAPS_DIR / f"{prefix}_overlay.jpg"
        heatmap_file = HEATMAPS_DIR / f"{prefix}_heatmap.jpg"
        crop_file = HEATMAPS_DIR / f"{prefix}_crop.jpg"

        self.assertTrue(overlay_file.exists())
        self.assertTrue(heatmap_file.exists())
        self.assertTrue(crop_file.exists())
        self.assertGreater(overlay_file.stat().st_size, 1000)

    def test_04_api_image_analysis_returns_gradcam(self):
        """Verify POST /api/analyze returns valid Grad-CAM URLs for face images."""
        with open(SAMPLE_FAKE, "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("test_gradcam_fake.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data.get("gradcam_available"))
        self.assertTrue(data.get("gradcam_overlay_url", "").startswith("/uploads/heatmaps/"))
        self.assertTrue(data.get("gradcam_heatmap_url", "").startswith("/uploads/heatmaps/"))
        self.assertTrue(data.get("original_face_crop_url", "").startswith("/uploads/heatmaps/"))

        # Verify the saved overlay can be fetched via static mount
        overlay_url = data["gradcam_overlay_url"]
        fetch_res = self.client.get(overlay_url)
        self.assertEqual(fetch_res.status_code, 200)

    def test_05_api_no_face_image_safely_handles_gradcam(self):
        """Verify no-face images gracefully return gradcam_available = False without crashing."""
        with open(SAMPLE_NO_FACE, "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("test_gradcam_noface.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data.get("status"), "no_face_detected")
        self.assertFalse(data.get("gradcam_available", False))
        self.assertIsNone(data.get("gradcam_overlay_url"))

    def test_06_api_video_analysis_returns_suspicious_frame_gradcam(self):
        """Verify video analysis generates Grad-CAM for suspicious frames."""
        with open(SAMPLE_FAKE_VIDEO, "rb") as f:
            res = self.client.post("/api/analyze", files={"file": ("test_gradcam_video.mp4", f, "video/mp4")})
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertTrue(data.get("gradcam_available"))
        self.assertIsNotNone(data.get("gradcam_overlay_url"))

        frame_results = data.get("frame_results", [])
        self.assertGreater(len(frame_results), 0)
        suspicious_cams = [f for f in frame_results if f.get("gradcam_available")]
        self.assertGreater(len(suspicious_cams), 0)
        self.assertTrue(suspicious_cams[0]["gradcam_overlay_url"].startswith("/uploads/heatmaps/"))


if __name__ == "__main__":
    unittest.main()
