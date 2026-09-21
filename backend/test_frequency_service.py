import unittest
from pathlib import Path
import numpy as np
import cv2

from app.services.frequency_service import (
    analyze_frequency_domain,
    compute_fft_magnitude_spectrum,
    extract_spectral_features
)
from app.models.schemas import FrequencyAnalysisResult
from app.config import FREQUENCY_DIR


class TestFrequencyService(unittest.TestCase):
    def setUp(self):
        # Create a synthetic image with varying frequency textures
        h, w = 128, 128
        y, x = np.ogrid[:h, :w]
        # Low frequency sinusoid + high frequency noise pattern
        pattern = 128 + 60 * np.sin(2 * np.pi * x / 32) + 20 * np.sin(2 * np.pi * y / 4)
        self.synthetic_img = np.clip(pattern, 0, 255).astype(np.uint8)
        self.synthetic_bgr = cv2.cvtColor(self.synthetic_img, cv2.COLOR_GRAY2BGR)

    def test_valid_image_fft_and_artifact_generation(self):
        prefix = "test_unit_freq"
        res = analyze_frequency_domain(self.synthetic_bgr, identifier_prefix=prefix, analysis_region="face_region")

        self.assertIsInstance(res, FrequencyAnalysisResult)
        self.assertTrue(res.available)
        self.assertEqual(res.analysis_region, "face_region")
        self.assertIsNotNone(res.frequency_spectrum_url)
        self.assertTrue(res.frequency_spectrum_url.startswith("/uploads/forensics/frequency/"))

        # Verify artifact file exists on disk
        artifact_name = f"{prefix}_frequency_spectrum.jpg"
        artifact_path = FREQUENCY_DIR / artifact_name
        self.assertTrue(artifact_path.exists())

        # Verify artifact image is readable and valid
        saved_img = cv2.imread(str(artifact_path))
        self.assertIsNotNone(saved_img)
        self.assertEqual(saved_img.shape[:2], (128, 128))

    def test_numerical_spectral_features(self):
        res = analyze_frequency_domain(self.synthetic_bgr, identifier_prefix="test_num_feat")

        self.assertGreater(res.total_energy, 0.0)
        self.assertGreaterEqual(res.low_frequency_energy, 0.0)
        self.assertGreaterEqual(res.mid_frequency_energy, 0.0)
        self.assertGreaterEqual(res.high_frequency_energy, 0.0)

        # Check high-frequency ratio is between 0 and 1
        self.assertGreaterEqual(res.high_frequency_ratio, 0.0)
        self.assertLessEqual(res.high_frequency_ratio, 1.0)

        # Check spectral centroid is within [0, 1]
        self.assertGreaterEqual(res.spectral_centroid, 0.0)
        self.assertLessEqual(res.spectral_centroid, 1.0)

        # Mean and median magnitudes must be positive
        self.assertGreater(res.mean_magnitude, 0.0)
        self.assertGreater(res.median_magnitude, 0.0)

        # Check findings exist and are factual
        self.assertGreaterEqual(len(res.findings), 2)

    def test_constant_and_zero_image_stability(self):
        # 1. All-zeros image
        zero_img = np.zeros((64, 64, 3), dtype=np.uint8)
        res_zero = analyze_frequency_domain(zero_img, identifier_prefix="test_zero_img")
        self.assertTrue(res_zero.available)
        self.assertEqual(res_zero.total_energy, 0.0)
        self.assertEqual(res_zero.high_frequency_ratio, 0.0)
        self.assertEqual(res_zero.spectral_centroid, 0.0)

        # 2. Constant non-zero image (pure DC component only)
        const_img = np.full((64, 64, 3), 180, dtype=np.uint8)
        res_const = analyze_frequency_domain(const_img, identifier_prefix="test_const_img")
        self.assertTrue(res_const.available)
        self.assertGreater(res_const.total_energy, 0.0)
        # DC component falls strictly in low-frequency band (rho = 0)
        self.assertGreater(res_const.low_frequency_energy, 0.0)
        self.assertEqual(res_const.high_frequency_energy, 0.0)
        self.assertEqual(res_const.high_frequency_ratio, 0.0)

    def test_invalid_input_handling(self):
        empty_arr = np.array([])
        res = analyze_frequency_domain(empty_arr, identifier_prefix="test_empty")
        self.assertFalse(res.available)
        self.assertGreaterEqual(len(res.findings), 1)

        none_res = analyze_frequency_domain(None, identifier_prefix="test_none")
        self.assertFalse(none_res.available)

    def test_deterministic_output(self):
        res1 = analyze_frequency_domain(self.synthetic_bgr, identifier_prefix="test_det_1")
        res2 = analyze_frequency_domain(self.synthetic_bgr, identifier_prefix="test_det_2")

        self.assertEqual(res1.total_energy, res2.total_energy)
        self.assertEqual(res1.high_frequency_ratio, res2.high_frequency_ratio)
        self.assertEqual(res1.spectral_centroid, res2.spectral_centroid)
        self.assertEqual(res1.mean_magnitude, res2.mean_magnitude)
        self.assertEqual(res1.median_magnitude, res2.median_magnitude)


if __name__ == "__main__":
    unittest.main()
