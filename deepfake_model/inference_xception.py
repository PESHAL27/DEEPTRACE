#!/usr/bin/env python3
"""
Single-image inference wrapper for DeepfakeBench Pretrained Xception Detector.

Usage:
    python inference_xception.py --image path/to/image.jpg

Outputs:
- Input image dimensions
- Processed dimensions
- Model device
- Model loaded successfully
- Inference time
- JSON structured prediction:
  {
      "model": "Xception",
      "prediction": "real" | "fake",
      "probability": 0.XX
  }
"""

import os
import sys
import time
import json
import argparse
import yaml
import cv2
import torch
import torchvision.transforms as T

# Locate DeepfakeBench directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(CURRENT_DIR, 'DeepfakeBench')
TRAINING_DIR = os.path.join(BENCH_DIR, 'training')

# Setup import paths
if TRAINING_DIR not in sys.path:
    sys.path.insert(0, TRAINING_DIR)
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

# Configure detectors module mapping to avoid unnecessary dependencies of other detectors
import types
from metrics.registry import DETECTOR
if 'detectors' not in sys.modules:
    det_mod = types.ModuleType('detectors')
    det_mod.__path__ = [os.path.join(TRAINING_DIR, 'detectors')]
    det_mod.DETECTOR = DETECTOR
    sys.modules['detectors'] = det_mod

from detectors.xception_detector import XceptionDetector


def load_xception_model(config_path: str, weights_path: str, device: torch.device) -> XceptionDetector:
    """Initializes XceptionDetector and loads the deepfake checkpoint."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure pretrained ImageNet backbone path is absolute
    if not os.path.isabs(config['pretrained']):
        config['pretrained'] = os.path.normpath(os.path.join(BENCH_DIR, config['pretrained']))

    # Instantiate detector
    detector = XceptionDetector(config)
    detector = detector.to(device)

    # Load pretrained deepfake detector checkpoint with strict matching
    checkpoint = torch.load(weights_path, map_location=device)
    detector.load_state_dict(checkpoint, strict=True)
    detector.eval()
    return detector, config


def preprocess_image(image_path: str, resolution: int):
    """
    Applies the official DeepfakeBench test preprocessing pipeline:
    1. Read BGR image
    2. Convert BGR to RGB
    3. Resize to target resolution (256x256) with cubic interpolation
    4. Convert to float tensor scaled to [0.0, 1.0]
    5. Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] to [-1.0, 1.0]
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    orig_h, orig_w, orig_c = img_bgr.shape
    input_dims = f"{orig_w}x{orig_h}x{orig_c}"

    # BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
    proc_dims = f"{resolution}x{resolution}x3"

    # Tensor conversion and normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor = transform(img_resized).unsqueeze(0)  # Shape: [1, 3, resolution, resolution]

    return tensor, input_dims, proc_dims


def run_inference(image_path: str, config_path: str = None, weights_path: str = None, device_str: str = 'auto'):
    if config_path is None:
        config_path = os.path.join(TRAINING_DIR, 'config', 'detector', 'xception.yaml')
    if weights_path is None:
        weights_path = os.path.join(TRAINING_DIR, 'weights', 'xception_best.pth')

    # Determine device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    # 1. Load Model
    detector, config = load_xception_model(config_path, weights_path, device)
    model_loaded_successfully = True

    # 2. Preprocess Image
    resolution = config.get('resolution', 256)
    tensor, input_dims, proc_dims = preprocess_image(image_path, resolution)
    tensor = tensor.to(device)

    # 3. Model Forward Pass
    start_time = time.perf_counter()
    with torch.no_grad():
        output = detector({'image': tensor})
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Probability Conversion
    logits = output['cls']  # Shape: [1, 2]
    probs = torch.softmax(logits, dim=1)[0]
    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    # In DeepfakeBench: class 0 = real, class 1 = fake
    if fake_prob >= 0.5:
        prediction = "fake"
        confidence = fake_prob
    else:
        prediction = "real"
        confidence = real_prob

    # Print diagnostics
    print(f"input image dimensions: {input_dims}")
    print(f"processed dimensions: {proc_dims}")
    print(f"model device: {device}")
    print(f"model loaded successfully: {model_loaded_successfully}")
    print(f"inference time: {inference_time_ms:.2f} ms")

    # Structured JSON output
    result = {
        "model": "Xception",
        "prediction": prediction,
        "probability": round(confidence, 4),
        "raw_scores": {
            "real": round(real_prob, 4),
            "fake": round(fake_prob, 4)
        }
    }
    print(json.dumps(result, indent=4))
    return result


def main():
    parser = argparse.ArgumentParser(description="DeepfakeBench Pretrained Xception Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to input image file")
    parser.add_argument('--config', type=str, default=None, help="Path to xception.yaml config file")
    parser.add_argument('--weights', type=str, default=None, help="Path to xception_best.pth checkpoint")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help="Execution device")
    args = parser.parse_args()

    run_inference(args.image, args.config, args.weights, args.device)


if __name__ == '__main__':
    main()
