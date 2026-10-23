"""
DEEPTRACE - Whole-Image AI-Generated Image Detection Evaluation
===============================================================
Executes an empirical benchmark across balanced, diverse categories:
1. Genuine camera photographs (portraits, landscapes, products, traditional art)
2. AI-generated portraits (Midjourney, StyleGAN, Stable Diffusion)
3. AI-generated landscapes and nature (Stable Diffusion 1.5, AI Mountains)
4. AI-generated commercial products & architecture (Pod vehicle, futuristic city)
5. AI-generated digital artwork & sci-fi scenes (Sci-fi exoplanet)
6. Localized face-swaps / spliced manipulations (FaceForensics++, DeepfakeBench)

Calculates:
- Confusion Matrix (TP, FP, TN, FN)
- Accuracy, Precision, Recall / Sensitivity
- False Positive Rate (FPR) on genuine images
- False Negative Rate (FNR) on AI-generated images
- Breakdown per visual category and per generator family
- Verification of parallel Xception facial manipulation integrity
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.detectors.ai_generated_image_detector import (
    detect_ai_generated_image,
    get_smaug_detector,
    PROVISIONAL_AI_THRESHOLD
)
from app.services.detection_service import analyze_image, predict_face_crop

MANIFEST_DIR = BACKEND_DIR / "evaluation" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = MANIFEST_DIR / "whole_image_eval_manifest.json"
RESULTS_DIR = BACKEND_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "whole_image_evaluation_results.json"


EVALUATION_SAMPLES = [
    # 1. Genuine Camera Photographs
    {
        "id": "real_cam_01",
        "name": "Nithin_portrait.jpeg",
        "path": str(BACKEND_DIR / "uploads" / "DT-2026-17022339_Nithin_sist_friend.jpeg"),
        "category": "genuine_portrait",
        "generator": "camera_smartphone",
        "ground_truth": "real"
    },
    {
        "id": "real_cam_02",
        "name": "MY_PHOTO.jpeg",
        "path": str(BACKEND_DIR / "uploads" / "DT-2026-16692338_MY_PHOTO.jpeg"),
        "category": "genuine_portrait",
        "generator": "camera_smartphone",
        "ground_truth": "real"
    },
    {
        "id": "real_cam_03",
        "name": "real_landscape_yosemite.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "real_landscape_yosemite.jpg"),
        "category": "genuine_landscape",
        "generator": "camera_dslr_canon5d",
        "ground_truth": "real"
    },
    {
        "id": "real_cam_04",
        "name": "real_product_coffee.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "real_product_coffee.jpg"),
        "category": "genuine_product",
        "generator": "camera_dslr",
        "ground_truth": "real"
    },
    {
        "id": "real_cam_05",
        "name": "real_artwork_monalisa.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "real_artwork_monalisa.jpg"),
        "category": "genuine_artwork",
        "generator": "traditional_painting",
        "ground_truth": "real"
    },
    {
        "id": "real_cam_06",
        "name": "no_face_field.jpg",
        "path": str(REPO_ROOT / "public" / "assets" / "no_face.jpg"),
        "category": "genuine_landscape",
        "generator": "camera_nature",
        "ground_truth": "real"
    },

    # 2. AI-Generated Portraits
    {
        "id": "ai_port_01",
        "name": "user_ai_generated.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_faces" / "user_ai_generated.jpg"),
        "category": "ai_portrait",
        "generator": "midjourney",
        "ground_truth": "ai_generated"
    },
    {
        "id": "ai_port_02",
        "name": "stylegan_woman.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_faces" / "stylegan_woman.jpg"),
        "category": "ai_portrait",
        "generator": "stylegan",
        "ground_truth": "ai_generated"
    },
    {
        "id": "ai_port_03",
        "name": "stylegan_boy.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_faces" / "stylegan_boy.jpg"),
        "category": "ai_portrait",
        "generator": "stylegan",
        "ground_truth": "ai_generated"
    },
    {
        "id": "ai_port_04",
        "name": "stable_diffusion_portrait.webp",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_faces" / "stable_diffusion_portrait.webp"),
        "category": "ai_portrait",
        "generator": "stable_diffusion",
        "ground_truth": "ai_generated"
    },

    # 3. AI-Generated Landscapes & Nature
    {
        "id": "ai_land_01",
        "name": "ai_landscape_valley_sd15.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "ai_landscape_valley_sd15.jpg"),
        "category": "ai_landscape",
        "generator": "stable_diffusion_1.5",
        "ground_truth": "ai_generated"
    },
    {
        "id": "ai_land_02",
        "name": "ai_landscape_mountains.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "ai_landscape_mountains.jpg"),
        "category": "ai_landscape",
        "generator": "stable_diffusion",
        "ground_truth": "ai_generated"
    },

    # 4. AI-Generated Products & Architecture
    {
        "id": "ai_prod_01",
        "name": "ai_product_vehicle.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "ai_product_vehicle.jpg"),
        "category": "ai_product",
        "generator": "stable_diffusion",
        "ground_truth": "ai_generated"
    },
    {
        "id": "ai_arch_01",
        "name": "ai_architecture_city.png",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "ai_architecture_city.png"),
        "category": "ai_architecture",
        "generator": "stable_diffusion",
        "ground_truth": "ai_generated"
    },

    # 5. AI-Generated Artwork & Sci-Fi Illustrations
    {
        "id": "ai_art_01",
        "name": "ai_artwork_scifi.jpg",
        "path": str(BACKEND_DIR / "test_fixtures" / "ai_whole_image" / "ai_artwork_scifi.jpg"),
        "category": "ai_artwork",
        "generator": "stable_diffusion",
        "ground_truth": "ai_generated"
    },

    # 6. Face-Swaps (Preserving Existing Xception Pipeline Verification)
    {
        "id": "swap_01",
        "name": "known_fake.jpg",
        "path": str(REPO_ROOT / "deepfake_model" / "tests" / "samples" / "known_fake.jpg"),
        "category": "face_swap_manipulated",
        "generator": "faceforensics_splicing",
        "ground_truth": "ai_generated"
    },
    {
        "id": "swap_02",
        "name": "all_in_one.jpg",
        "path": str(REPO_ROOT / "deepfake_model" / "DeepfakeBench" / "training" / "dataset" / "library" / "all_in_one.jpg"),
        "category": "face_swap_manipulated",
        "generator": "deepfakebench_faceswap",
        "ground_truth": "ai_generated"
    }
]


def run_evaluation() -> Dict[str, Any]:
    print("=" * 80)
    print("DEEPTRACE: Whole-Image AI Detection Benchmark (Smaug-AI-Image-Detector)")
    print("=" * 80)

    # Save manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(EVALUATION_SAMPLES, f, indent=2)
    print(f"Manifest written to: {MANIFEST_PATH}")

    # Initialize model
    print("Initializing Smaug AI Image Detector...")
    t_load_start = time.perf_counter()
    detector = get_smaug_detector()
    t_load_end = time.perf_counter()
    print(f"Model ready (loaded in {t_load_end - t_load_start:.2f}s). Running evaluations...\n")

    results: List[Dict[str, Any]] = []

    print(f"{'Category':<22} | {'Sample Name':<28} | {'Generator':<18} | {'Exp':<6} | {'Smaug P':<8} | {'Smaug Verdict':<24} | {'Status'}")
    print("-" * 125)

    for item in EVALUATION_SAMPLES:
        p = Path(item["path"])
        if not p.exists():
            print(f"Skipping missing: {p}")
            continue

        # Whole-image AI detection (crops=1 for standard whole-image evaluation)
        ai_res = detect_ai_generated_image(p, crops=1, threshold=PROVISIONAL_AI_THRESHOLD)

        # Xception facial manipulation detection (runs in parallel if image contains faces)
        xc_pred = None
        xc_prob = None
        face_count = 0
        try:
            full_analysis = analyze_image(p)
            face_count = len(full_analysis.faces or [])
            xc_pred = full_analysis.prediction
            xc_prob = full_analysis.probability
        except Exception:
            pass

        score = ai_res.score if ai_res.score is not None else 0.0
        exp = item["ground_truth"]
        pred_label = "ai_generated" if score >= PROVISIONAL_AI_THRESHOLD else "real"
        correct = (pred_label == exp)
        status_str = "CORRECT" if correct else ("FALSE_POSITIVE" if exp == "real" else "FALSE_NEGATIVE")

        score_str = f"{score:.4f}" if ai_res.score is not None else "ERR"

        record = {
            "id": item["id"],
            "name": item["name"],
            "category": item["category"],
            "generator": item["generator"],
            "ground_truth": exp,
            "smaug_score": score,
            "smaug_predicted_label": pred_label,
            "smaug_assessment": ai_res.assessment,
            "correct": correct,
            "status": status_str,
            "execution_time_ms": ai_res.execution_time_ms,
            "face_count": face_count,
            "xception_prediction": xc_pred,
            "xception_probability": xc_prob
        }
        results.append(record)

        print(f"{item['category']:<22} | {item['name']:<28} | {item['generator']:<18} | {exp:<6} | {score_str:<8} | {ai_res.assessment:<24} | {status_str}")

    # Compute Aggregate Metrics
    # Exclude face_swap cohort from whole-image pure AI generation calculation (or report both)
    whole_image_samples = [r for r in results if r["category"] != "face_swap_manipulated"]
    
    total = len(whole_image_samples)
    reals = [r for r in whole_image_samples if r["ground_truth"] == "real"]
    ais = [r for r in whole_image_samples if r["ground_truth"] == "ai_generated"]

    TN = sum(1 for r in reals if r["smaug_predicted_label"] == "real")
    FP = sum(1 for r in reals if r["smaug_predicted_label"] == "ai_generated")
    TP = sum(1 for r in ais if r["smaug_predicted_label"] == "ai_generated")
    FN = sum(1 for r in ais if r["smaug_predicted_label"] == "real")

    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 80)
    print("WHOLE-IMAGE AI-GENERATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Evaluated Samples (Non-Spliced): {total}")
    print(f"Genuine Real Samples: {len(reals)} | AI-Generated Samples: {len(ais)}")
    print(f"Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")
    print(f"Accuracy:                {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision:               {precision:.4f} ({precision*100:.1f}%)")
    print(f"Recall (Sensitivity):    {recall:.4f} ({recall*100:.1f}%)")
    print(f"False Negative Rate:     {fnr:.4f} ({fnr*100:.1f}%)")
    print(f"False Positive Rate:     {fpr:.4f} ({fpr*100:.1f}%)")
    print(f"F1-Score:                {f1:.4f}")

    # Per-Category Metrics
    print("\n" + "-" * 80)
    print("PER-CATEGORY BREAKDOWN:")
    categories = sorted(set(r["category"] for r in results))
    cat_summary = {}
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        cat_total = len(cat_rows)
        cat_correct = sum(1 for r in cat_rows if r["correct"])
        cat_acc = (cat_correct / cat_total * 100) if cat_total > 0 else 0.0
        mean_score = np.mean([r["smaug_score"] for r in cat_rows])
        cat_summary[cat] = {
            "total": cat_total,
            "correct": cat_correct,
            "accuracy": cat_acc,
            "mean_score": float(mean_score)
        }
        print(f"  - {cat:<24}: Total={cat_total:2d} | Correct={cat_correct:2d} ({cat_acc:5.1f}%) | Mean Score={mean_score:.4f}")

    # Per-Generator Breakdown
    print("\n" + "-" * 80)
    print("PER-GENERATOR BREAKDOWN:")
    generators = sorted(set(r["generator"] for r in results))
    gen_summary = {}
    for gen in generators:
        gen_rows = [r for r in results if r["generator"] == gen]
        gen_total = len(gen_rows)
        gen_correct = sum(1 for r in gen_rows if r["correct"])
        gen_acc = (gen_correct / gen_total * 100) if gen_total > 0 else 0.0
        mean_score = np.mean([r["smaug_score"] for r in gen_rows])
        gen_summary[gen] = {
            "total": gen_total,
            "correct": gen_correct,
            "accuracy": gen_acc,
            "mean_score": float(mean_score)
        }
        print(f"  - {gen:<24}: Total={gen_total:2d} | Correct={gen_correct:2d} ({gen_acc:5.1f}%) | Mean Score={mean_score:.4f}")

    # Face-Swap Verification Breakdown
    swap_rows = [r for r in results if r["category"] == "face_swap_manipulated"]
    print("\n" + "-" * 80)
    print("PARALLEL XCEPTION PIPELINE INTEGRITY (Face-Swaps):")
    for sr in swap_rows:
        print(f"  - {sr['name']:<20}: Xception Verdict='{sr['xception_prediction']}' (Prob={sr['xception_probability']}) | Faces={sr['face_count']}")

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": "abacusai/Smaug-AI-Image-Detector",
        "threshold": PROVISIONAL_AI_THRESHOLD,
        "metrics": {
            "total": total,
            "tn": TN,
            "fp": FP,
            "fn": FN,
            "tp": TP,
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_negative_rate": round(fnr, 4),
            "false_positive_rate": round(fpr, 4),
            "f1_score": round(f1, 4)
        },
        "category_breakdown": cat_summary,
        "generator_breakdown": gen_summary,
        "sample_records": results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nFull evaluation results saved to: {RESULTS_PATH}")

    return output_data


if __name__ == "__main__":
    run_evaluation()
