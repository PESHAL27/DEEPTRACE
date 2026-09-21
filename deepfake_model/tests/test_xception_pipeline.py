"""
Unit and Integration Tests for DeepfakeBench Pretrained Xception Detector.

Tests:
1. Model initialization from training/config/detector/xception.yaml
2. Checkpoint loading from training/weights/xception_best.pth (strict=True)
3. Image preprocessing pipeline (Resize -> Tensor -> Normalize)
4. Forward pass through Xception architecture
5. Output tensor validation (cls, feat, prob)
6. Probability conversion (softmax, real/fake thresholding)
7. End-to-end inference on known-real sample
8. End-to-end inference on known-fake sample
"""

import os
import sys
import unittest
import yaml
import torch
import cv2
import torchvision.transforms as T

# Locate project directories
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(TESTS_DIR)
BENCH_DIR = os.path.join(MODEL_DIR, 'DeepfakeBench')
TRAINING_DIR = os.path.join(BENCH_DIR, 'training')

# Setup path
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

# Configure detectors module mapping
import types
from metrics.registry import DETECTOR
if 'detectors' not in sys.modules:
    det_mod = types.ModuleType('detectors')
    det_mod.__path__ = [os.path.join(TRAINING_DIR, 'detectors')]
    det_mod.DETECTOR = DETECTOR
    sys.modules['detectors'] = det_mod

from detectors.xception_detector import XceptionDetector
from inference_xception import load_xception_model, preprocess_image, run_inference


class TestXceptionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = os.path.join(TRAINING_DIR, 'config', 'detector', 'xception.yaml')
        cls.weights_path = os.path.join(TRAINING_DIR, 'weights', 'xception_best.pth')
        cls.samples_dir = os.path.join(TESTS_DIR, 'samples')
        cls.real_sample = os.path.join(cls.samples_dir, 'known_real.jpg')
        cls.fake_sample = os.path.join(cls.samples_dir, 'known_fake.jpg')
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)

        if not os.path.isabs(cls.config['pretrained']):
            cls.config['pretrained'] = os.path.normpath(os.path.join(BENCH_DIR, cls.config['pretrained']))

    def test_01_model_initialization(self):
        """Verify model initialization matches xception.yaml config."""
        self.assertEqual(self.config['model_name'], 'xception')
        self.assertEqual(self.config['backbone_name'], 'xception')
        self.assertEqual(self.config['backbone_config']['num_classes'], 2)
        self.assertEqual(self.config['resolution'], 256)

        model = XceptionDetector(self.config)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'backbone'))
        self.assertTrue(hasattr(model, 'loss_func'))

    def test_02_checkpoint_loading(self):
        """Verify checkpoint loading with strict=True against all 283 parameters."""
        self.assertTrue(os.path.isfile(self.weights_path), f"Checkpoint not found at {self.weights_path}")
        model = XceptionDetector(self.config).to(self.device)
        ckpt = torch.load(self.weights_path, map_location=self.device)
        
        load_result = model.load_state_dict(ckpt, strict=True)
        # strict=True returns IncompatibleKeys with empty missing_keys and unexpected_keys
        self.assertEqual(len(load_result.missing_keys), 0, f"Missing keys: {load_result.missing_keys}")
        self.assertEqual(len(load_result.unexpected_keys), 0, f"Unexpected keys: {load_result.unexpected_keys}")

    def test_03_preprocessing_pipeline(self):
        """Verify preprocessing transforms image correctly to [1, 3, 256, 256] in [-1, 1]."""
        tensor, in_dims, proc_dims = preprocess_image(self.real_sample, 256)
        self.assertEqual(tensor.shape, torch.Size([1, 3, 256, 256]))
        self.assertEqual(proc_dims, "256x256x3")
        # Check normalized range approximately in [-1.0, 1.0]
        self.assertGreaterEqual(tensor.min().item(), -1.01)
        self.assertLessEqual(tensor.max().item(), 1.01)

    def test_04_forward_pass_and_output_tensors(self):
        """Verify forward pass returns cls, feat, and prob with proper shapes."""
        detector, _ = load_xception_model(self.config_path, self.weights_path, self.device)
        dummy_input = {'image': torch.randn(1, 3, 256, 256, device=self.device)}
        with torch.no_grad():
            output = detector(dummy_input)

        self.assertIn('cls', output)
        self.assertIn('prob', output)
        self.assertIn('feat', output)

        self.assertEqual(output['cls'].shape, torch.Size([1, 2]))
        self.assertEqual(output['prob'].shape, torch.Size([1]))
        self.assertEqual(output['feat'].shape, torch.Size([1, 2048, 8, 8]))

    def test_05_probability_conversion(self):
        """Verify softmax output produces valid probabilities summing to 1.0."""
        detector, _ = load_xception_model(self.config_path, self.weights_path, self.device)
        dummy_input = {'image': torch.randn(1, 3, 256, 256, device=self.device)}
        with torch.no_grad():
            output = detector(dummy_input)

        probs = torch.softmax(output['cls'], dim=1)[0]
        real_p = probs[0].item()
        fake_p = probs[1].item()

        self.assertGreaterEqual(real_p, 0.0)
        self.assertLessEqual(real_p, 1.0)
        self.assertGreaterEqual(fake_p, 0.0)
        self.assertLessEqual(fake_p, 1.0)
        self.assertAlmostEqual(real_p + fake_p, 1.0, places=4)

    def test_06_real_sample_inference(self):
        """Verify end-to-end inference correctly identifies known real sample."""
        result = run_inference(self.real_sample, self.config_path, self.weights_path, str(self.device))
        self.assertEqual(result['model'], 'Xception')
        self.assertEqual(result['prediction'], 'real')
        self.assertGreater(result['probability'], 0.5)

    def test_07_fake_sample_inference(self):
        """Verify end-to-end inference correctly identifies known fake sample."""
        result = run_inference(self.fake_sample, self.config_path, self.weights_path, str(self.device))
        self.assertEqual(result['model'], 'Xception')
        self.assertEqual(result['prediction'], 'fake')
        self.assertGreater(result['probability'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
