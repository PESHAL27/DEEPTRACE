"""
DEEPTRACE - External Benchmark Setup & Manifest Generator
=========================================================
Downloads a strictly pre-defined, reproducible, paired external facial-manipulation
benchmark subset (N=100 samples: 50 authentic, 50 manipulated) from the open
DeepGuard InsightFace benchmark (Sowaiba01/Deepfake).

Selection Protocol:
- Deterministic sequential index partition: i in [0, 49]
- Exactly 50 authentic original identity portraits (real_00000.jpg to real_00049.jpg)
- Exactly 50 paired face-swapped target images (fake_00000.jpg to fake_00049.jpg)
- Balanced 50:50 class distribution
- Pre-selected strictly prior to model inference
- No sample exclusion based on detector predictions
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BACKEND_DIR / "evaluation" / "datasets" / "external_benchmark"
MANIFEST_PATH = BACKEND_DIR / "evaluation" / "manifests" / "external_benchmark_manifest.json"

DATASET_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_benchmark(num_pairs: int = 50):
    print(f"Setting up external facial-manipulation benchmark with {num_pairs} identity pairs ({num_pairs * 2} samples)...")
    samples = []

    for idx in range(num_pairs):
        r_rel = f"real/real_{idx:05d}.jpg"
        f_rel = f"fake/fake_{idx:05d}.jpg"

        src_r = Path(hf_hub_download("Sowaiba01/Deepfake", r_rel, repo_type="dataset"))
        src_f = Path(hf_hub_download("Sowaiba01/Deepfake", f_rel, repo_type="dataset"))

        dst_r = DATASET_DIR / f"real_{idx:05d}.jpg"
        dst_f = DATASET_DIR / f"fake_{idx:05d}.jpg"

        if not dst_r.exists():
            shutil.copy2(src_r, dst_r)
        if not dst_f.exists():
            shutil.copy2(src_f, dst_f)

        sha_r = sha256_file(dst_r)
        sha_f = sha256_file(dst_f)

        # Authentic sample
        samples.append({
            "sample_id": f"EXT-REAL-{idx:04d}",
            "filename": dst_r.name,
            "file_path": f"evaluation/datasets/external_benchmark/{dst_r.name}",
            "sha256": sha_r,
            "media_type": "image",
            "ground_truth": "authentic",
            "category": "genuine_real_photo",
            "source_dataset": "DeepGuard-InsightFace",
            "identity_id": f"identity_{idx:04d}",
            "manipulation_type": "none",
            "generator": "none",
            "is_eligible_for_detection_metrics": True,
            "notes": f"Authentic target face for identity #{idx:04d}."
        })

        # Manipulated sample (paired with authentic target)
        samples.append({
            "sample_id": f"EXT-FAKE-{idx:04d}",
            "filename": dst_f.name,
            "file_path": f"evaluation/datasets/external_benchmark/{dst_f.name}",
            "sha256": sha_f,
            "media_type": "image",
            "ground_truth": "manipulated",
            "category": "known_facial_manipulation",
            "source_dataset": "DeepGuard-InsightFace",
            "identity_id": f"identity_{idx:04d}",
            "manipulation_type": "face_swap",
            "generator": "InsightFace_inswapper_128",
            "is_eligible_for_detection_metrics": True,
            "notes": f"Face-swap deepfake targeting identity #{idx:04d} generated via inswapper_128."
        })

    manifest = {
        "manifest_version": "1.0",
        "dataset_name": "DEEPTRACE-External-Facial-Manipulation-Benchmark",
        "benchmark_source": "DeepGuard InsightFace Paired Benchmark (Sowaiba01/Deepfake)",
        "license": "MIT",
        "selection_protocol": "Deterministic sequential pairing i in [0, 49], 50 authentic, 50 manipulated",
        "class_balance": {
            "authentic": num_pairs,
            "manipulated": num_pairs,
            "total_samples": num_pairs * 2
        },
        "created_date": "2026-10-20",
        "samples": samples
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Benchmark setup complete.")
    print(f"Total downloaded image files in {DATASET_DIR}: {len(list(DATASET_DIR.glob('*.jpg')))}")
    print(f"Manifest generated at: {MANIFEST_PATH}")


if __name__ == "__main__":
    setup_benchmark(num_pairs=50)
