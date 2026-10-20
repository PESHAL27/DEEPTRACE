"""
Objective evaluation of padding ratios: 0.10, 0.15, 0.20, 0.25, 0.30
Evaluates each sample with full metrics and records exact bounding boxes and probabilities.
"""

import sys
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces
from app.services.detection_service import get_xception_detector, analyze_image

samples = [
    {
        'id': 'known_real.jpg',
        'desc': 'FaceForensics++ Authentic Baseline (actor scene, 2 faces)',
        'path': (REPO_ROOT / 'deepfake_model' / 'tests' / 'samples' / 'known_real.jpg').resolve(),
        'gt': 'authentic'
    },
    {
        'id': 'Nithin_sist_friend.jpeg',
        'desc': 'User portrait photograph (in-the-wild authentic camera sample)',
        'path': (BACKEND_DIR / 'uploads' / 'DT-2026-17022339_Nithin_sist_friend.jpeg').resolve(),
        'gt': 'authentic'
    },
    {
        'id': 'MY_PHOTO.jpeg',
        'desc': 'User portrait photograph (in-the-wild authentic camera sample)',
        'path': (BACKEND_DIR / 'uploads' / 'DT-2026-16692338_MY_PHOTO.jpeg').resolve(),
        'gt': 'authentic'
    },
    {
        'id': 'known_fake.jpg',
        'desc': 'FaceForensics++ Deepfake c23 Manipulated Sample',
        'path': (REPO_ROOT / 'deepfake_model' / 'tests' / 'samples' / 'known_fake.jpg').resolve(),
        'gt': 'manipulated'
    },
    {
        'id': 'all_in_one.jpg',
        'desc': 'Face X-Ray Blended Fake Faces (50-face mosaic)',
        'path': (REPO_ROOT / 'deepfake_model' / 'DeepfakeBench' / 'training' / 'dataset' / 'library' / 'all_in_one.jpg').resolve(),
        'gt': 'manipulated'
    }
]

padding_ratios = [0.10, 0.15, 0.20, 0.25, 0.30]
model, cfg, dev = get_xception_detector()

results = {}

for pad in padding_ratios:
    pad_key = f"pad_{pad:.2f}"
    results[pad_key] = {"padding": pad, "samples": [], "metrics": {}}
    tp = fp = tn = fn = 0
    print(f"\n==================== PADDING RATIO: {pad:.2f} ====================")
    
    for s in samples:
        d_info = detect_and_crop_faces(s['path'], padding_ratio=pad)
        res = analyze_image(s['path'], detection_info=d_info)
        
        face0 = res.faces[0] if res.faces else None
        p_real = face0.real_probability if face0 else None
        p_fake = face0.fake_probability if face0 else None
        logits = face0.raw_logits if face0 else None
        bbox = d_info['bounding_boxes'][0] if d_info.get('bounding_boxes') else None
        crop_shape = list(d_info['crops'][0].shape) if d_info.get('crops') else None
        
        gt = s['gt']
        pred = res.prediction
        
        if gt == 'manipulated':
            if pred == 'fake':
                tp += 1
                outcome = 'TP'
            else:
                fn += 1
                outcome = 'FN'
        elif gt == 'authentic':
            if pred == 'real':
                tn += 1
                outcome = 'TN'
            else:
                fp += 1
                outcome = 'FP'
        else:
            outcome = 'CONTROL'

        sample_res = {
            "id": s['id'],
            "ground_truth": gt,
            "prediction": pred,
            "assessment": res.assessment,
            "model_probability": res.model_probability,
            "face_count": len(res.faces),
            "primary_face_crop_shape": crop_shape,
            "primary_face_bbox": bbox,
            "primary_face_real_prob": p_real,
            "primary_face_fake_prob": p_fake,
            "primary_face_logits": logits,
            "outcome": outcome
        }
        results[pad_key]["samples"].append(sample_res)
        
        print(f"Sample: {s['id']}")
        print(f"  Bbox: {bbox}, Crop Shape: {crop_shape}")
        print(f"  Logits: {logits}")
        print(f"  P(authentic)={p_real:.4f}, P(manipulated)={p_fake:.4f} -> {outcome} ({res.assessment})")

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr
    }
    results[pad_key]["metrics"] = metrics
    print(f"Metrics (N={total}): Acc={acc*100:.1f}%, Prec={prec*100:.1f}%, Rec={rec*100:.1f}%, Spec={spec*100:.1f}%, F1={f1:.3f}, FPR={fpr*100:.1f}%, FNR={fnr*100:.1f}%")

output_file = BACKEND_DIR / "evaluation" / "padding_comparison_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll padding comparison results written to: {output_file}")
