"""
DEEPTRACE - Development Baseline Runner
=======================================
Runs the existing frozen Xception detection pipeline on the development-train (N=200)
and development-validation (N=100) splits without modifying production weights,
preprocessing, crop padding (0.20), or the 0.50 decision threshold.

Outputs detailed forensic evaluation metrics, confusion matrices, AP, PR-AUC,
stratified performance by face size and image blurriness, and 95% confidence intervals.
"""

import sys
import os
import json
import time
import math
import hashlib
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces
from app.services.detection_service import analyze_image, DEEPFAKE_THRESHOLD

EVAL_DIR = BACKEND_DIR / "evaluation"
TRAIN_MANIFEST = EVAL_DIR / "manifests" / "development_train_manifest.json"
VAL_MANIFEST = EVAL_DIR / "manifests" / "development_validation_manifest.json"
RESULTS_PATH = EVAL_DIR / "results" / "development_baseline_metrics.json"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def wilson_score_ci(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculates Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (p_hat + z**2 / (2 * n)) / denominator
    adjusted_std = math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    lower = max(0.0, centre_adjusted_probability - z * adjusted_std)
    upper = min(1.0, centre_adjusted_probability + z * adjusted_std)
    return round(lower, 4), round(upper, 4)


def paired_cluster_bootstrap_ci(records: List[Dict[str, Any]], n_bootstraps: int = 1000) -> Dict[str, Any]:
    """
    Cluster bootstrap at the pair level (pairs are clusters of 2 images: 1 real, 1 fake)
    to compute 95% empirical confidence intervals acknowledging paired non-independence.
    """
    pair_dict = {}
    for r in records:
        pid = r["pair_index"]
        if pid not in pair_dict:
            pair_dict[pid] = []
        pair_dict[pid].append(r)
        
    pair_keys = list(pair_dict.keys())
    n_pairs = len(pair_keys)
    if n_pairs == 0:
        return {}
        
    accs = []
    recalls = []
    specificities = []
    rocs = []
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        sampled_keys = np.random.choice(pair_keys, size=n_pairs, replace=True)
        boot_records = []
        for k in sampled_keys:
            boot_records.extend(pair_dict[k])
            
        tp = sum(1 for r in boot_records if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "fake")
        fp = sum(1 for r in boot_records if r["ground_truth"] == "authentic" and r["pf_prediction"] == "fake")
        tn = sum(1 for r in boot_records if r["ground_truth"] == "authentic" and r["pf_prediction"] == "real")
        fn = sum(1 for r in boot_records if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "real")
        
        acc = (tp + tn) / len(boot_records) if boot_records else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        accs.append(acc)
        recalls.append(rec)
        specificities.append(spec)
        
        y_true = [1 if r["ground_truth"] == "manipulated" else 0 for r in boot_records]
        y_scores = [r["pf_fake_probability"] for r in boot_records]
        if len(set(y_true)) > 1:
            try:
                rocs.append(roc_auc_score(y_true, y_scores))
            except Exception:
                pass
                
    return {
        "accuracy_95ci": [round(float(np.percentile(accs, 2.5)), 4), round(float(np.percentile(accs, 97.5)), 4)],
        "recall_95ci": [round(float(np.percentile(recalls, 2.5)), 4), round(float(np.percentile(recalls, 97.5)), 4)],
        "specificity_95ci": [round(float(np.percentile(specificities, 2.5)), 4), round(float(np.percentile(specificities, 97.5)), 4)],
        "roc_auc_95ci": [round(float(np.percentile(rocs, 2.5)), 4), round(float(np.percentile(rocs, 97.5)), 4)] if rocs else [0.0, 0.0]
    }


def evaluate_split(manifest_path: Path, split_name: str) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    samples = manifest["samples"]
    print(f"\nEvaluating {split_name} split ({len(samples)} samples)...")
    
    evaluated_records = []
    t_start = time.perf_counter()
    
    for idx, s in enumerate(samples, 1):
        target_path = BACKEND_DIR / s["file_path"]
        gt = s["ground_truth"]
        
        # Read image to compute blurriness
        img = cv2.imread(str(target_path))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap_var = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
        else:
            lap_var = 0.0
            
        d_info = detect_and_crop_faces(str(target_path))
        if d_info["count"] == 0:
            print(f"Warning: No face detected in {s['filename']}")
            continue
            
        res = analyze_image(target_path, detection_info=d_info)
        pf = res.primary_face_result
        af = res.all_faces_result
        
        pf_pred = pf.prediction if pf else res.prediction
        pf_prob = pf.probability if pf else res.model_probability
        pf_fake_prob = pf.fake_probability if pf else (pf_prob if pf_pred == "fake" else 1.0 - pf_prob)
        pf_is_low_margin = pf.is_low_margin if pf else res.is_low_margin
        
        af_pred = af.prediction if af else res.prediction
        af_prob = af.probability if af else res.model_probability
        af_fake_prob = af.mean_fake_probability if af else pf_fake_prob
        af_is_low_margin = af.is_low_margin if af else res.is_low_margin
        
        pf_area = pf.area if pf else 0
        pf_rel_area = pf.relative_area if pf else 0.0
        
        evaluated_records.append({
            "sample_id": s["sample_id"],
            "filename": s["filename"],
            "pair_index": s["pair_index"],
            "target_canvas_id": s.get("target_face_canvas_id"),
            "donor_identity_id": s.get("source_donor_identity_id"),
            "ground_truth": gt,
            "manipulation_type": s.get("manipulation_type"),
            "generator": s.get("generator"),
            "laplacian_blur_var": lap_var,
            "face_count": d_info["count"],
            "face_area": pf_area,
            "face_relative_area": pf_rel_area,
            "pf_prediction": pf_pred,
            "pf_probability": round(pf_prob, 4),
            "pf_fake_probability": round(pf_fake_prob, 4),
            "pf_is_low_margin": pf_is_low_margin,
            "pf_correct": (pf_pred == "fake" and gt == "manipulated") or (pf_pred == "real" and gt == "authentic"),
            "af_prediction": af_pred,
            "af_fake_probability": round(af_fake_prob, 4),
            "af_is_low_margin": af_is_low_margin,
            "af_correct": (af_pred == "fake" and gt == "manipulated") or (af_pred == "real" and gt == "authentic")
        })
        
        if idx % 50 == 0 or idx == len(samples):
            print(f"  Processed {idx}/{len(samples)} samples...")
            
    elapsed = time.perf_counter() - t_start
    print(f"Evaluation finished in {elapsed:.2f}s ({len(evaluated_records)/elapsed:.2f} samples/s).")
    
    # Calculate Primary Face metrics
    y_true = [1 if r["ground_truth"] == "manipulated" else 0 for r in evaluated_records]
    y_scores = [r["pf_fake_probability"] for r in evaluated_records]
    
    tp = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "fake")
    fp = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["pf_prediction"] == "fake")
    tn = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["pf_prediction"] == "real")
    fn = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "real")
    total = len(evaluated_records)
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    roc_auc = round(float(roc_auc_score(y_true, y_scores)), 4)
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc_trapezoidal = round(float(auc(r_curve, p_curve)), 4)
    ap_score = round(float(average_precision_score(y_true, y_scores)), 4)
    
    # Statistical Confidence Intervals
    wilson_acc = wilson_score_ci(accuracy, total)
    wilson_rec = wilson_score_ci(recall, tp + fn)
    wilson_spec = wilson_score_ci(specificity, tn + fp)
    bootstrap_ci = paired_cluster_bootstrap_ci(evaluated_records)
    
    # Stratified analysis by Face Size (Area in pixels: small < 20k, medium 20k-30k, large > 30k)
    size_bins = {"small (<20k px)": [], "medium (20k-30k px)": [], "large (>30k px)": []}
    for r in evaluated_records:
        area = r["face_area"]
        if area < 20000:
            size_bins["small (<20k px)"].append(r)
        elif area <= 30000:
            size_bins["medium (20k-30k px)"].append(r)
        else:
            size_bins["large (>30k px)"].append(r)
            
    size_performance = {}
    for b_name, recs in size_bins.items():
        if recs:
            c = sum(1 for r in recs if r["pf_correct"])
            size_performance[b_name] = {
                "count": len(recs),
                "accuracy": round(c / len(recs), 4),
                "tp": sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "fake"),
                "fn": sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "real")
            }
            
    # Stratified analysis by Image Blurriness (Laplacian variance: low < 80, med 80-150, high > 150)
    blur_bins = {"low sharpness (<80)": [], "medium sharpness (80-150)": [], "high sharpness (>150)": []}
    for r in evaluated_records:
        lv = r["laplacian_blur_var"]
        if lv < 80:
            blur_bins["low sharpness (<80)"].append(r)
        elif lv <= 150:
            blur_bins["medium sharpness (80-150)"].append(r)
        else:
            blur_bins["high sharpness (>150)"].append(r)
            
    blur_performance = {}
    for b_name, recs in blur_bins.items():
        if recs:
            c = sum(1 for r in recs if r["pf_correct"])
            blur_performance[b_name] = {
                "count": len(recs),
                "accuracy": round(c / len(recs), 4),
                "tp": sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "fake"),
                "fn": sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_prediction"] == "real")
            }
            
    low_margin_count = sum(1 for r in evaluated_records if r["pf_is_low_margin"])
    
    return {
        "split_name": split_name,
        "sample_count": total,
        "elapsed_seconds": round(elapsed, 2),
        "contingency_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "roc_auc": roc_auc,
            "pr_auc_trapezoidal": pr_auc_trapezoidal,
            "average_precision": ap_score
        },
        "confidence_intervals": {
            "wilson_accuracy_95ci": wilson_acc,
            "wilson_recall_95ci": wilson_rec,
            "wilson_specificity_95ci": wilson_spec,
            "paired_cluster_bootstrap_95ci": bootstrap_ci
        },
        "stratification_by_face_size": size_performance,
        "stratification_by_image_sharpness": blur_performance,
        "low_margin_samples_count": low_margin_count,
        "low_margin_ratio": round(low_margin_count / total, 4) if total > 0 else 0.0,
        "evaluated_records": evaluated_records
    }


def main():
    print("=" * 80)
    print("DEEPTRACE DEVELOPMENT BASELINE INFERENCE GATE (FROZEN XCEPTION)")
    print("=" * 80)
    
    train_res = evaluate_split(TRAIN_MANIFEST, "development_train")
    val_res = evaluate_split(VAL_MANIFEST, "development_validation")
    
    full_results = {
        "evaluation_timestamp": "2026-10-20T19:08:00Z",
        "model_architecture": "Xception (DeepfakeBench pretrained)",
        "model_weights_checkpoint": "backend/app/models/weights/xception_best.pth",
        "decision_threshold": DEEPFAKE_THRESHOLD,
        "face_crop_padding": 0.20,
        "splits_evaluated": ["development_train", "development_validation"],
        "development_train": train_res,
        "development_validation": val_res
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
        
    print("\n" + "=" * 80)
    print("DEVELOPMENT BASELINE SUMMARY:")
    print("=" * 80)
    print(f"Train (N={train_res['sample_count']}):      Accuracy: {train_res['metrics']['accuracy']:.2%}, Recall: {train_res['metrics']['recall']:.2%}, Specificity: {train_res['metrics']['specificity']:.2%}, ROC-AUC: {train_res['metrics']['roc_auc']:.4f}")
    print(f"Validation (N={val_res['sample_count']}): Accuracy: {val_res['metrics']['accuracy']:.2%}, Recall: {val_res['metrics']['recall']:.2%}, Specificity: {val_res['metrics']['specificity']:.2%}, ROC-AUC: {val_res['metrics']['roc_auc']:.4f}")
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
