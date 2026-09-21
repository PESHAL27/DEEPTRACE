import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.services.detection_service import analyze_video
from app.config import VIDEO_SAMPLE_INTERVAL, MAX_ANALYZED_FRAMES

FIXTURES_DIR = BACKEND_DIR / "test_fixtures"
FAKE_VIDEO_PATH = FIXTURES_DIR / "test_fake_video.mp4"
REAL_VIDEO_PATH = FIXTURES_DIR / "test_real_video.mp4"
ZERO_FACE_VIDEO_PATH = FIXTURES_DIR / "test_video.mp4"

class TestVideoAnalysisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        assert FAKE_VIDEO_PATH.exists(), f"Missing {FAKE_VIDEO_PATH}"
        assert REAL_VIDEO_PATH.exists(), f"Missing {REAL_VIDEO_PATH}"
        assert ZERO_FACE_VIDEO_PATH.exists(), f"Missing {ZERO_FACE_VIDEO_PATH}"

    def test_01_analyze_video_fake(self):
        """Test video analysis on synthetic deepfake video."""
        result = analyze_video(FAKE_VIDEO_PATH, sample_interval=15, max_frames=5)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.prediction, "fake")
        self.assertGreaterEqual(result.video_probability, 0.70)
        self.assertGreater(result.frames_analyzed, 0)
        self.assertGreaterEqual(result.faces_detected, 1)
        self.assertIsNotNone(result.frame_results)
        self.assertGreater(len(result.frame_results), 0)

        # Verify frame item structure
        first_frame = result.frame_results[0]
        self.assertIsInstance(first_frame.frame_number, int)
        self.assertIsInstance(first_frame.timestamp, float)
        self.assertTrue(first_frame.face_detected)
        self.assertIsNotNone(first_frame.bounding_box)
        self.assertIn(first_frame.prediction, ["fake", "real"])
        self.assertIsInstance(first_frame.probability, float)

    def test_02_analyze_video_real(self):
        """Test video analysis on authentic video."""
        result = analyze_video(REAL_VIDEO_PATH, sample_interval=15, max_frames=5)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.prediction, "real")
        self.assertGreaterEqual(result.video_probability, 0.50)
        self.assertGreater(result.frames_analyzed, 0)
        self.assertGreaterEqual(result.faces_detected, 1)
        self.assertIsNotNone(result.frame_results)

    def test_03_analyze_video_no_face(self):
        """Test video analysis on video without any faces."""
        result = analyze_video(ZERO_FACE_VIDEO_PATH, sample_interval=15, max_frames=5)
        self.assertEqual(result.status, "no_face_detected")
        self.assertIsNone(result.prediction)
        self.assertIsNone(result.video_probability)
        self.assertEqual(result.faces_detected, 0)
        self.assertGreater(result.frames_analyzed, 0)
        for frame in result.frame_results:
            self.assertFalse(frame.face_detected)
            self.assertIsNone(frame.bounding_box)
            self.assertIsNone(frame.prediction)

    def test_04_api_endpoint_video_upload(self):
        """Test POST /api/analyze with video file upload."""
        with open(FAKE_VIDEO_PATH, "rb") as f:
            files = {"file": ("test_fake_video.mp4", f, "video/mp4")}
            response = self.client.post("/api/analyze", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["media_type"], "video")
        self.assertEqual(data["prediction"], "fake")
        self.assertIn("video_probability", data)
        self.assertGreaterEqual(data["video_probability"], 0.70)
        self.assertIn("frames_analyzed", data)
        self.assertGreater(data["frames_analyzed"], 0)
        self.assertIn("faces_detected", data)
        self.assertGreaterEqual(data["faces_detected"], 1)
        self.assertIn("frame_results", data)
        self.assertIsInstance(data["frame_results"], list)
        self.assertGreater(len(data["frame_results"]), 0)

        # Check required frame result schema
        frame0 = data["frame_results"][0]
        self.assertIn("frame_number", frame0)
        self.assertIn("timestamp", frame0)
        self.assertIn("face_detected", frame0)
        self.assertIn("bounding_box", frame0)
        self.assertIn("prediction", frame0)
        self.assertIn("probability", frame0)

    def test_05_api_endpoint_zero_face_video(self):
        """Test POST /api/analyze with zero-face video file."""
        with open(ZERO_FACE_VIDEO_PATH, "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            response = self.client.post("/api/analyze", files=files)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["media_type"], "video")
        self.assertEqual(data["status"], "no_face_detected")
        self.assertIsNone(data["prediction"])
        self.assertIsNone(data["video_probability"])
        self.assertEqual(data["faces_detected"], 0)


if __name__ == "__main__":
    unittest.main()
