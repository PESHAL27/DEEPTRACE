"""
Verification script for DeepfakeBench Pretrained Xception Detector.
Loads the official deepfake detector checkpoint xception_best.pth into the exact XceptionDetector implementation.
"""
import os
import sys
import yaml
import torch

# Ensure DeepfakeBench training package is in path
repo_root = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(repo_root, 'training')
sys.path.insert(0, training_dir)
sys.path.insert(0, repo_root)

# Isolate detectors registry import to avoid unrelated detectors' dependency conflicts on Windows
import types
from metrics.registry import DETECTOR
det_mod = types.ModuleType('detectors')
det_mod.__path__ = [os.path.join(training_dir, 'detectors')]
det_mod.DETECTOR = DETECTOR
sys.modules['detectors'] = det_mod

from detectors.xception_detector import XceptionDetector

def verify():
    config_path = os.path.join(training_dir, 'config', 'detector', 'xception.yaml')
    weights_path = os.path.join(training_dir, 'weights', 'xception_best.pth')

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Confirm key configuration parameters
    model_name = config.get('model_name')
    backbone_name = config.get('backbone_name')
    num_classes = config.get('backbone_config', {}).get('num_classes')
    resolution = config.get('resolution')

    print(f"Config - model_name: {model_name}")
    print(f"Config - backbone_name: {backbone_name}")
    print(f"Config - num_classes: {num_classes}")
    print(f"Config - resolution: {resolution}")

    assert model_name == 'xception', f"Expected model_name 'xception', got '{model_name}'"
    assert backbone_name == 'xception', f"Expected backbone_name 'xception', got '{backbone_name}'"
    assert num_classes == 2, f"Expected num_classes 2, got {num_classes}"

    # Ensure pretrained ImageNet backbone path is absolute if relative
    if not os.path.isabs(config['pretrained']):
        config['pretrained'] = os.path.normpath(os.path.join(repo_root, config['pretrained']))

    # 1. Initialize exact XceptionDetector
    print("\nInitializing XceptionDetector...")
    detector = XceptionDetector(config)
    detector.eval()
    print("XceptionDetector initialized successfully.")

    # 2. Load deepfake detector checkpoint
    print(f"\nLoading deepfake checkpoint from: {weights_path}")
    assert os.path.isfile(weights_path), f"Checkpoint not found at {weights_path}"
    checkpoint = torch.load(weights_path, map_location='cpu')

    # Load with strict=True
    load_result = detector.load_state_dict(checkpoint, strict=True)
    print(f"Load state_dict result (strict=True): {load_result}")

    # 3. Perform test forward pass
    dummy_input = {'image': torch.randn(1, 3, resolution, resolution)}
    with torch.no_grad():
        output = detector(dummy_input)

    print(f"\nForward Pass Verification:")
    print(f"- Output keys: {list(output.keys())}")
    print(f"- Logits shape (cls): {output['cls'].shape}")
    print(f"- Probability shape (prob): {output['prob'].shape}")
    print(f"- Feature representation shape (feat): {output['feat'].shape}")
    print(f"- Sample fake probability: {output['prob'].item():.4f}")
    print("\nPretrained Xception deepfake detector is fully configured and operational!")

if __name__ == '__main__':
    verify()
