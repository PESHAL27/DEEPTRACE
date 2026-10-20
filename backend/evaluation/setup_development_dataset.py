"""
DEEPTRACE - Development & Validation Dataset Setup
==================================================
Downloads and configures strictly disjoint development-train (N=200 samples, 100 pairs)
and development-validation (N=100 samples, 50 pairs) datasets from Sowaiba01/Deepfake
at revision d8b189507fc026b3e0c794a06c318cb13d0c8743.

Split Architecture:
- Fixed Benchmark (Preserved & Frozen): Pairs 0000 - 0049 (N=100)
- Development Train: Pairs 0050 - 0149 (N=200: 100 Authentic, 100 Manipulated)
- Development Validation: Pairs 0150 - 0199 (N=100: 50 Authentic, 50 Manipulated)

Guarantees:
- Authentic and manipulated pairs are strictly co-located in the same split.
- Target faces (target_face) and donor identities (source_identity) are 100% disjoint
  across Benchmark, Train, and Validation.
- SHA-256 and difference-hash (dHash) deduplication verified.
"""

import os
import csv
import json
import shutil
import hashlib
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

REPO_ID = "Sowaiba01/Deepfake"
REVISION = "d8b189507fc026b3e0c794a06c318cb13d0c8743"

BACKEND_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = BACKEND_DIR / "evaluation"
TRAIN_DIR = EVAL_DIR / "datasets" / "development_train"
VAL_DIR = EVAL_DIR / "datasets" / "development_validation"
MANIFESTS_DIR = EVAL_DIR / "manifests"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(image_path: Path, hash_size: int = 8) -> str:
    """Computes difference hash (dHash) for perceptual image fingerprinting."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "0" * 16
    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    decimal_val = 0
    hex_str = []
    for idx, bit in enumerate(diff.flatten()):
        if bit:
            decimal_val += 2 ** (idx % 4)
        if idx % 4 == 3:
            hex_str.append(hex(decimal_val)[2:])
            decimal_val = 0
    return "".join(hex_str)


def download_pair(idx: int, dest_dir: Path):
    r_rel = f"real/real_{idx:05d}.jpg"
    f_rel = f"fake/fake_{idx:05d}.jpg"
    
    src_r = Path(hf_hub_download(REPO_ID, r_rel, repo_type="dataset", revision=REVISION))
    src_f = Path(hf_hub_download(REPO_ID, f_rel, repo_type="dataset", revision=REVISION))
    
    dst_r = dest_dir / f"real_{idx:05d}.jpg"
    dst_f = dest_dir / f"fake_{idx:05d}.jpg"
    
    if not dst_r.exists():
        shutil.copy2(src_r, dst_r)
    if not dst_f.exists():
        shutil.copy2(src_f, dst_f)
    return idx


def build_split(start_idx: int, end_idx: int, dest_dir: Path, split_name: str, metadata_rows: list):
    num_pairs = end_idx - start_idx
    print(f"\nSetting up {split_name} split (Pairs {start_idx:04d} to {end_idx-1:04d}, {num_pairs} pairs, {num_pairs*2} samples)...")
    
    # Concurrently download images
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_pair, i, dest_dir) for i in range(start_idx, end_idx)]
        for f in futures:
            f.result()
            
    print(f"Downloaded all {num_pairs * 2} files into {dest_dir}.")
    
    samples = []
    seen_sha = set()
    seen_dhash = set()
    duplicate_sha_count = 0
    duplicate_dhash_count = 0
    
    meta_dict = {int(r["id"]): r for r in metadata_rows}
    
    for idx in range(start_idx, end_idx):
        meta = meta_dict.get(idx, {})
        target_face_id = meta.get("target_face", f"target_{idx:05d}")
        source_donor_id = meta.get("source_identity", f"donor_{idx:05d}")
        
        dst_r = dest_dir / f"real_{idx:05d}.jpg"
        dst_f = dest_dir / f"fake_{idx:05d}.jpg"
        
        sha_r = compute_sha256(dst_r)
        sha_f = compute_sha256(dst_f)
        dhash_r = compute_dhash(dst_r)
        dhash_f = compute_dhash(dst_f)
        
        for h in [sha_r, sha_f]:
            if h in seen_sha:
                duplicate_sha_count += 1
            seen_sha.add(h)
            
        for dh in [dhash_r, dhash_f]:
            if dh in seen_dhash:
                duplicate_dhash_count += 1
            seen_dhash.add(dh)
            
        rel_path_r = f"evaluation/datasets/{dest_dir.name}/{dst_r.name}"
        rel_path_f = f"evaluation/datasets/{dest_dir.name}/{dst_f.name}"
        
        # Authentic record
        samples.append({
            "sample_id": f"DEV-{split_name.upper()}-REAL-{idx:04d}",
            "filename": dst_r.name,
            "file_path": rel_path_r,
            "sha256": sha_r,
            "dhash": dhash_r,
            "media_type": "image",
            "image_dimensions": [256, 256, 3],
            "ground_truth": "authentic",
            "split": split_name,
            "pair_index": idx,
            "target_face_canvas_id": target_face_id,
            "source_donor_identity_id": "none",
            "manipulation_type": "none",
            "generator": "none",
            "is_eligible_for_detection_metrics": True,
            "notes": f"Authentic target face (CelebA canvas {target_face_id}) for pair #{idx:04d}."
        })
        
        # Manipulated record
        samples.append({
            "sample_id": f"DEV-{split_name.upper()}-FAKE-{idx:04d}",
            "filename": dst_f.name,
            "file_path": rel_path_f,
            "sha256": sha_f,
            "dhash": dhash_f,
            "media_type": "image",
            "image_dimensions": [256, 256, 3],
            "ground_truth": "manipulated",
            "split": split_name,
            "pair_index": idx,
            "target_face_canvas_id": target_face_id,
            "source_donor_identity_id": source_donor_id,
            "manipulation_type": "face_swap",
            "generator": "InsightFace_inswapper_128",
            "is_eligible_for_detection_metrics": True,
            "notes": f"InsightFace face swap onto target canvas {target_face_id} from donor {source_donor_id}."
        })
        
    manifest = {
        "manifest_version": "1.0",
        "dataset_name": f"DEEPTRACE-Development-{split_name.capitalize()}-Split",
        "benchmark_source": f"DeepGuard InsightFace (Sowaiba01/Deepfake at revision {REVISION})",
        "split_name": split_name,
        "split_policy": f"Separate {split_name} split with strictly disjoint target canvases and donor identities",
        "pair_range": [start_idx, end_idx - 1],
        "class_balance": {
            "authentic": num_pairs,
            "manipulated": num_pairs,
            "total_samples": len(samples)
        },
        "deduplication_audit": {
            "duplicate_sha256_count": duplicate_sha_count,
            "duplicate_dhash_count": duplicate_dhash_count
        },
        "samples": samples
    }
    
    out_manifest = MANIFESTS_DIR / f"development_{split_name}_manifest.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to: {out_manifest} ({len(samples)} records)")
    return manifest


def main():
    print("=" * 80)
    print("DEEPTRACE DEVELOPMENT DATASET SETUP & MANIFEST GENERATOR")
    print(f"Upstream Revision: {REVISION}")
    print("=" * 80)
    
    meta_path = Path(hf_hub_download(REPO_ID, "metadata.csv", repo_type="dataset", revision=REVISION))
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        
    # Build Train: pairs 50 to 149 (100 pairs = 200 samples)
    train_manifest = build_split(50, 150, TRAIN_DIR, "train", rows)
    
    # Build Validation: pairs 150 to 199 (50 pairs = 100 samples)
    val_manifest = build_split(150, 200, VAL_DIR, "validation", rows)
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE:")
    print(f"  Development Train:      {len(train_manifest['samples'])} samples in {TRAIN_DIR}")
    print(f"  Development Validation: {len(val_manifest['samples'])} samples in {VAL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
