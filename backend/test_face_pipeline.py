"""
Independent Verification Test for Face Preprocessing and Detection Pipeline.

Verifies:
1. Face detection & cropping (face_service.py) on:
   - Zero-face image
   - Single-face image
   - Multi-face image
2. DeepfakeBench exact preprocessing (detection_service.py):
   - BGR -> RGB
   - 256x256 interpolation
   - ToTensor()
   - Normalize([-1, 1])
3. Xception inference & probability extraction (predict_face_crop)
4. End-to-end analyze_image on:
   - Zero-face image -> {"status": "no_face_detected"}
   - Single-face image -> {"status": "completed", "faces": [...]}
   - Multi-face image -> Analyzes each face separately
"""

import sys
import os
from pathlib import Path
import unittest
import torch
import numpy as np

# Setup paths
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces, detect_faces
from app.services.detection_service import (
    preprocess_face_crop,
    predict_face_crop,
    analyze_image,
    analyze_image_dict,
    get_xception_detector
)

WORKSPACE_ROOT = BACKEND_DIR.parent
SAMPLE_REAL = WORKSPACE_ROOT / "public" / "assets" / "sample_frame.jpg"
SAMPLE_NO_FACE = BACKEND_DIR / "test_fixtures" / "test_image.jpg"
SAMPLE_MULTI_FACE = WORKSPACE_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "dataset" / "library" / "all_in_one.jpg"


class TestFacePipeline(unittest.TestCase):
    def test_01_no_face_detection(self):
        """Verify image with no face returns count=0 and status='no_face_detected'."""
        self.assertTrue(SAMPLE_NO_FACE.exists(), f"Sample not found: {SAMPLE_NO_FACE}")
        result = detect_and_crop_faces(SAMPLE_NO_FACE)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["bounding_boxes"], [])
        self.assertEqual(result["crops"], [])
        self.assertEqual(result["status"], "no_face_detected")

    def test_02_single_face_detection_and_crop(self):
        """Verify single face image returns count=1, valid bbox, and crop."""
        self.assertTrue(SAMPLE_REAL.exists(), f"Sample not found: {SAMPLE_REAL}")
        result = detect_and_crop_faces(SAMPLE_REAL)
        self.assertGreaterEqual(result["count"], 1)
        self.assertEqual(result["status"], "faces_detected")

        bbox = result["bounding_boxes"][0]
        self.assertIn("x", bbox)
        self.assertIn("y", bbox)
        self.assertIn("width", bbox)
        self.assertIn("height", bbox)
        self.assertGreater(bbox["width"], 50)
        self.assertGreater(bbox["height"], 50)

        crop = result["crops"][0]
        self.assertIsInstance(crop, np.ndarray)
        self.assertEqual(len(crop.shape), 3)
        self.assertEqual(crop.shape[2], 3)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_03_multi_face_detection(self):
        """Verify multi-face image detects multiple faces and generates separate crops."""
        self.assertTrue(SAMPLE_MULTI_FACE.exists(), f"Sample not found: {SAMPLE_MULTI_FACE}")
        result = detect_and_crop_faces(SAMPLE_MULTI_FACE)
        self.assertGreater(result["count"], 5)
        self.assertEqual(len(result["bounding_boxes"]), result["count"])
        self.assertEqual(len(result["crops"]), result["count"])

    def test_04_face_crop_preprocessing(self):
        """Verify preprocessing matches DeepfakeBench: (1, 3, 256, 256) in [-1.0, 1.0]."""
        dummy_crop = np.random.randint(0, 256, (180, 180, 3), dtype=np.uint8)
        tensor = preprocess_face_crop(dummy_crop, resolution=256, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        self.assertEqual(tensor.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), -1.01)
        self.assertLessEqual(tensor.max().item(), 1.01)

    def test_05_predict_face_crop_inference(self):
        """Verify Xception inference on a face crop returns valid probabilities."""
        result = detect_and_crop_faces(SAMPLE_REAL)
        crop = result["crops"][0]

        pred_result = predict_face_crop(crop)
        self.assertIn("prediction", pred_result)
        self.assertIn(pred_result["prediction"], ["real", "fake"])
        self.assertIn("probability", pred_result)
        self.assertGreaterEqual(pred_result["probability"], 0.0)
        self.assertLessEqual(pred_result["probability"], 1.0)
        self.assertAlmostEqual(pred_result["real_probability"] + pred_result["fake_probability"], 1.0, places=3)

    def test_06_analyze_image_no_face(self):
        """Verify analyze_image returns status='no_face_detected' when no faces exist."""
        result = analyze_image(SAMPLE_NO_FACE)
        self.assertEqual(result.status, "no_face_detected")
        self.assertIsNone(result.prediction)
        self.assertIsNone(result.probability)
        self.assertEqual(len(result.faces), 0)

    def test_07_analyze_image_single_face(self):
        """Verify analyze_image on a single face returns completed status and face item."""
        result = analyze_image(SAMPLE_REAL)
        self.assertEqual(result.status, "completed")
        model_name = result.model.name if hasattr(result.model, 'name') else str(result.model)
        self.assertEqual(model_name, "Xception")
        self.assertIn(result.prediction, ["real", "fake"])
        self.assertGreater(result.probability, 0.0)
        self.assertGreaterEqual(len(result.faces), 1)

        face_0 = result.faces[0]
        self.assertEqual(face_0.face_index, 0)
        self.assertGreater(face_0.bounding_box.width, 0)
        self.assertGreater(face_0.bounding_box.height, 0)
        self.assertIn(face_0.prediction, ["real", "fake"])
        self.assertGreater(face_0.probability, 0.0)

    def test_08_analyze_image_multiple_faces(self):
        """Verify analyze_image analyzes each face separately on multi-face images."""
        result = analyze_image(SAMPLE_MULTI_FACE)
        self.assertEqual(result.status, "completed")
        self.assertGreater(len(result.faces), 5)

        for i, face_item in enumerate(result.faces):
            self.assertEqual(face_item.face_index, i)
            self.assertIn(face_item.prediction, ["real", "fake"])
            self.assertGreaterEqual(face_item.probability, 0.0)
            self.assertLessEqual(face_item.probability, 1.0)

    def test_09_analyze_image_dict_no_face(self):
        """Verify analyze_image_dict returns {'status': 'no_face_detected'} when no face is found."""
        result = analyze_image_dict(SAMPLE_NO_FACE)
        self.assertEqual(result, {"status": "no_face_detected"})

    def test_10_analyze_image_dict_structure(self):
        """Verify analyze_image_dict returns the exact required schema with faces list."""
        result = analyze_image_dict(SAMPLE_REAL)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model"], {"name": "Xception", "source": "DeepfakeBench"})
        self.assertIn(result["prediction"], ["real", "fake"])
        self.assertIn("faces", result)
        self.assertGreaterEqual(len(result["faces"]), 1)

        face_0 = result["faces"][0]
        self.assertEqual(face_0["face_index"], 0)
        self.assertIn("bounding_box", face_0)
        self.assertIn("x", face_0["bounding_box"])
        self.assertIn("y", face_0["bounding_box"])
        self.assertIn("width", face_0["bounding_box"])
        self.assertIn("height", face_0["bounding_box"])
        self.assertIn("prediction", face_0)
        self.assertIn("probability", face_0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

