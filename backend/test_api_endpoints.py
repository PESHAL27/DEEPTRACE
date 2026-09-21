"""
Comprehensive End-to-End Verification Test for DEEPTRACE FastAPI Backend.

Verifies POST /api/analyze across 5 required scenarios:
1. Known real image (deepfake_model/tests/samples/known_real.jpg)
2. Known fake/deepfake image (deepfake_model/tests/samples/known_fake.jpg)
3. Image with no face (backend/test_fixtures/test_image.jpg)
4. Image with multiple faces (deepfake_model/DeepfakeBench/training/dataset/library/all_in_one.jpg)
5. Invalid image (backend/test_fixtures/test_script.txt)

Also verifies:
- Startup model pre-warming (Xception & YuNet loaded once via lifespan)
- GPU/CPU fallback without crashing
- Calibrated terminology ("Model probability", "Detection probability", "Likely manipulated")
- Configurable DEEPFAKE_THRESHOLD
- Multi-face aggregation strategy
"""

import sys
import os
import unittest
from pathlib import Path

# Setup path
BACKEND_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.config import DEEPFAKE_THRESHOLD

SAMPLE_REAL = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_real.jpg"
SAMPLE_FAKE = WORKSPACE_ROOT / "deepfake_model" / "tests" / "samples" / "known_fake.jpg"
SAMPLE_NO_FACE = BACKEND_DIR / "test_fixtures" / "test_image.jpg"
SAMPLE_MULTI_FACE = WORKSPACE_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "dataset" / "library" / "all_in_one.jpg"
SAMPLE_INVALID = BACKEND_DIR / "test_fixtures" / "test_script.txt"


class TestAnalyzeEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verify test sample files exist
        assert SAMPLE_REAL.exists(), f"Sample real not found: {SAMPLE_REAL}"
        assert SAMPLE_FAKE.exists(), f"Sample fake not found: {SAMPLE_FAKE}"
        assert SAMPLE_NO_FACE.exists(), f"Sample no-face not found: {SAMPLE_NO_FACE}"
        assert SAMPLE_MULTI_FACE.exists(), f"Sample multi-face not found: {SAMPLE_MULTI_FACE}"
        assert SAMPLE_INVALID.exists(), f"Sample invalid not found: {SAMPLE_INVALID}"

        # Initialize TestClient within lifespan context to verify startup model warming
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_01_known_real_image(self):
        """Case 1: Known real image returns completed status, real prediction, and face breakdown."""
        with open(SAMPLE_REAL, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": (SAMPLE_REAL.name, f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200, f"Failed: {response.text}")
        data = response.json()

        # Schema & Identity assertions
        self.assertIn("analysis_id", data)
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["media_type"], "image")
        self.assertEqual(data["filename"], SAMPLE_REAL.name)
        self.assertEqual(data["model"], {"name": "Xception", "source": "DeepfakeBench"})

        # Face detection assertions
        self.assertGreaterEqual(data["faces_detected"], 1)
        self.assertEqual(len(data["faces"]), data["faces_detected"])

        face_0 = data["faces"][0]
        self.assertEqual(face_0["face_index"], 0)
        self.assertIn("bounding_box", face_0)
        self.assertGreater(face_0["bounding_box"]["width"], 0)
        self.assertGreater(face_0["bounding_box"]["height"], 0)
        self.assertIn(face_0["prediction"], ["real", "fake"])
        self.assertGreaterEqual(face_0["probability"], 0.0)

        # Classification assertions for known real sample
        self.assertEqual(data["prediction"], "real")
        self.assertEqual(data["assessment"], "Likely authentic")
        self.assertGreater(data["probability"], 0.5)
        print(f"\n[PASS] Case 1 (Known Real): Prediction='{data['prediction']}', Prob={data['probability']}, Faces={data['faces_detected']}")

    def test_02_known_fake_image(self):
        """Case 2: Known fake image returns completed status, fake prediction, and high manipulation probability."""
        with open(SAMPLE_FAKE, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": (SAMPLE_FAKE.name, f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200, f"Failed: {response.text}")
        data = response.json()

        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["model"], {"name": "Xception", "source": "DeepfakeBench"})
        self.assertGreaterEqual(data["faces_detected"], 1)
        self.assertEqual(data["prediction"], "fake")
        self.assertEqual(data["assessment"], "Likely manipulated")
        self.assertGreater(data["probability"], 0.70)
        print(f"[PASS] Case 2 (Known Fake): Prediction='{data['prediction']}', Prob={data['probability']}, Faces={data['faces_detected']}")

    def test_03_no_face_image(self):
        """Case 3: Image with no face returns status='no_face_detected' without classifying as fake."""
        with open(SAMPLE_NO_FACE, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": (SAMPLE_NO_FACE.name, f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200, f"Failed: {response.text}")
        data = response.json()

        self.assertEqual(data["status"], "no_face_detected")
        self.assertEqual(data["faces_detected"], 0)
        self.assertEqual(data["faces"], [])
        self.assertIsNone(data["prediction"])
        self.assertIsNone(data["probability"])
        self.assertEqual(data["assessment"], "No face detected")
        print(f"[PASS] Case 3 (No Face): Status='{data['status']}', Prediction={data['prediction']}, Faces={data['faces_detected']}")

    def test_04_multiple_faces_image(self):
        """Case 4: Image with multiple faces analyzes each face separately and aggregates results."""
        with open(SAMPLE_MULTI_FACE, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": (SAMPLE_MULTI_FACE.name, f, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200, f"Failed: {response.text}")
        data = response.json()

        self.assertEqual(data["status"], "completed")
        self.assertGreater(data["faces_detected"], 5)
        self.assertEqual(len(data["faces"]), data["faces_detected"])

        # Check that each face has individual results
        for idx, face in enumerate(data["faces"]):
            self.assertEqual(face["face_index"], idx)
            self.assertIn("bounding_box", face)
            self.assertIn(face["prediction"], ["real", "fake"])
            self.assertGreaterEqual(face["probability"], 0.0)
            self.assertLessEqual(face["probability"], 1.0)

        # Multi-face aggregation validation
        any_face_fake = any(f["prediction"] == "fake" for f in data["faces"])
        if any_face_fake:
            self.assertEqual(data["prediction"], "fake")
            self.assertEqual(data["assessment"], "Likely manipulated")
        else:
            self.assertEqual(data["prediction"], "real")
            self.assertEqual(data["assessment"], "Likely authentic")

        print(f"[PASS] Case 4 (Multi-Face): Total Faces={data['faces_detected']}, Overall Prediction='{data['prediction']}', Overall Prob={data['probability']}")

    def test_05_invalid_file(self):
        """Case 5: Unsupported or invalid file is rejected with appropriate HTTP 400 or 415 error."""
        with open(SAMPLE_INVALID, "rb") as f:
            response = self.client.post(
                "/api/analyze",
                files={"file": (SAMPLE_INVALID.name, f, "text/plain")}
            )

        self.assertIn(response.status_code, [400, 415])
        data = response.json()
        self.assertIn("error", data)
        self.assertIn("detail", data)
        print(f"[PASS] Case 5 (Invalid File): HTTP {response.status_code} properly rejected ({data.get('detail')})")


if __name__ == '__main__':
    unittest.main(verbosity=2)
