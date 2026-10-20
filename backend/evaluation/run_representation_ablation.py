"""
DEEPTRACE - Phase 2C: Controlled Input-Representation Ablation
=============================================================
Ablates and compares 5 explicitly labeled model variants on the development dataset:
Variant A: Frozen Spatial Xception Baseline (Unchanged)
Variant B: Frequency-Only (Full-Canvas Representation)
Variant C: Frequency-Only (Primary-Face-Crop Representation)
Variant D: Fused Model (Xception + Full-Canvas Frequency Features)
Variant E: Fused Model (Xception + Primary-Face-Crop Frequency Features)

Training Partition:   Development Train (Pairs 0050-0149, N=200)
Validation Partition: Development Validation (Pairs 0150-0199, N=100) - EXPLORATORY
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, auc
)

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces
from app.services.detection_service import DEEPFAKE_THRESHOLD

EVAL_DIR = BACKEND_DIR / "evaluation"
BASELINE_PATH = EVAL_DIR / "results" / "development_baseline_metrics.json"
OUTPUT_JSON_PATH = EVAL_DIR / "results" / "phase2c_representation_ablation.json"


def extract_8d_frequency_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Computes 8-dimensional deterministic frequency feature vector:
    1. FFT high-frequency ratio (rho >= 0.50)
    2. FFT mid-frequency ratio (0.15 <= rho < 0.50)
    3. FFT low-frequency ratio (rho < 0.15)
    4. FFT spectral centroid
    5. FFT log magnitude mean
    6. DCT high-frequency AC energy ratio
    7. DCT mid-frequency AC energy ratio
    8. Log-Laplacian edge variance
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros(8, dtype=np.float32)
        
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_256 = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    h, w = gray_256.shape
    
    # 2D Fast Fourier Transform
    f = np.fft.fft2(np.float32(gray_256))
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    
    cy, cx = h / 2.0, w / 2.0
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_radius = np.sqrt(cy ** 2 + cx ** 2)
    rho = distances / max_radius
    
    squared_mag = mag ** 2
    total_energy = float(np.sum(squared_mag)) + 1e-10
    low_ratio = float(np.sum(squared_mag[rho < 0.15])) / total_energy
    mid_ratio = float(np.sum(squared_mag[(rho >= 0.15) & (rho < 0.50)])) / total_energy
    high_ratio = float(np.sum(squared_mag[rho >= 0.50])) / total_energy
    
    mag_sum = float(np.sum(mag)) + 1e-10
    spectral_centroid = float(np.sum(rho * mag) / mag_sum)
    log_mag_mean = float(np.mean(np.log1p(mag)))
    
    # 2D Discrete Cosine Transform (DCT)
    float_img = np.float32(gray_256) / 255.0
    dct_coeffs = cv2.dct(float_img)
    dct_ac = np.abs(dct_coeffs)
    dct_ac[0, 0] = 0.0
    total_ac = float(np.sum(dct_ac ** 2)) + 1e-10
    
    dct_dist = np.sqrt(y_coords ** 2 + x_coords ** 2) / np.sqrt(h ** 2 + w ** 2)
    dct_mid_ratio = float(np.sum((dct_ac ** 2)[(dct_dist >= 0.15) & (dct_dist < 0.50)])) / total_ac
    dct_high_ratio = float(np.sum((dct_ac ** 2)[dct_dist >= 0.50])) / total_ac
    
    lap_var = float(cv2.Laplacian(gray_256, cv2.CV_64F).var())
    
    return np.array([
        high_ratio, mid_ratio, low_ratio,
        spectral_centroid, log_mag_mean,
        dct_high_ratio, dct_mid_ratio,
        np.log1p(lap_var)
    ], dtype=np.float32)


def compute_metrics(y_true: List[int], y_pred: List[int], y_scores: List[float]) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = len(y_true)
    
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    roc_auc = round(float(roc_auc_score(y_true, y_scores)), 4)
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = round(float(auc(r_curve, p_curve)), 4)
    ap = round(float(average_precision_score(y_true, y_scores)), 4)
    
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "total": total,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "roc_auc": roc_auc,
        "pr_auc_trapezoidal": pr_auc,
        "average_precision": ap
    }


def main():
    print("=" * 80)
    print("DEEPTRACE PHASE 2C: INPUT REPRESENTATION ABLATION (FULL CANVAS VS FACE CROP)")
    print("=" * 80)
    
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
        
    train_recs = baseline_data["development_train"]["evaluated_records"]
    val_recs = baseline_data["development_validation"]["evaluated_records"]
    
    print(f"Loaded {len(train_recs)} train records and {len(val_recs)} validation records.")
    
    # 1. Feature Extraction
    print("Extracting features: Full-Canvas and Primary-Face-Crop...")
    
    def process_records(records, split_folder):
        X_full = []
        X_crop = []
        y = []
        xc_scores = []
        pairs = []
        
        for r in records:
            p = BACKEND_DIR / "evaluation" / "datasets" / split_folder / r["filename"]
            img = cv2.imread(str(p))
            
            # Representation 1: Full Canvas
            f_full = extract_8d_frequency_features(img)
            X_full.append(f_full)
            
            # Representation 2: YuNet Primary Face Crop
            d_info = detect_and_crop_faces(str(p))
            if d_info["count"] > 0 and len(d_info["crops"]) > 0:
                crop = d_info["crops"][0]
                f_crop = extract_8d_frequency_features(crop)
            else:
                f_crop = f_full.copy()
            X_crop.append(f_crop)
            
            y.append(1 if r["ground_truth"] == "manipulated" else 0)
            xc_scores.append(r["pf_fake_probability"])
            pairs.append(r["pair_index"])
            
        return (
            np.array(X_full, dtype=np.float32),
            np.array(X_crop, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(xc_scores, dtype=np.float32),
            pairs
        )
        
    X_train_full, X_train_crop, y_train, train_xc_scores, train_pairs = process_records(train_recs, "development_train")
    X_val_full, X_val_crop, y_val, val_xc_scores, val_pairs = process_records(val_recs, "development_validation")
    
    # Convert Xception probabilities to logits for linear fusion: logit = ln(p / (1-p))
    train_xc_logits = np.array([np.log(np.clip(s, 1e-5, 1 - 1e-5) / (1 - np.clip(s, 1e-5, 1 - 1e-5))) for s in train_xc_scores]).reshape(-1, 1)
    val_xc_logits = np.array([np.log(np.clip(s, 1e-5, 1 - 1e-5) / (1 - np.clip(s, 1e-5, 1 - 1e-5))) for s in val_xc_scores]).reshape(-1, 1)
    
    # 2. Scaler Fitting strictly on training split
    scaler_full = StandardScaler().fit(X_train_full)
    X_train_full_scaled = scaler_full.transform(X_train_full)
    X_val_full_scaled = scaler_full.transform(X_val_full)
    
    scaler_crop = StandardScaler().fit(X_train_crop)
    X_train_crop_scaled = scaler_crop.transform(X_train_crop)
    X_val_crop_scaled = scaler_crop.transform(X_val_crop)
    
    # -------------------------------------------------------------
    # Variant A: Frozen Spatial Xception Baseline (Unchanged)
    # -------------------------------------------------------------
    val_preds_a = [1 if s >= 0.50 else 0 for s in val_xc_scores]
    val_scores_a = list(val_xc_scores)
    metrics_a = compute_metrics(list(y_val), val_preds_a, val_scores_a)
    
    # -------------------------------------------------------------
    # Variant B: Frequency-Only (Full Canvas)
    # -------------------------------------------------------------
    clf_b = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(X_train_full_scaled, y_train)
    val_scores_b = list(clf_b.predict_proba(X_val_full_scaled)[:, 1])
    val_preds_b = [1 if p >= 0.50 else 0 for p in val_scores_b]
    metrics_b = compute_metrics(list(y_val), val_preds_b, val_scores_b)
    
    # -------------------------------------------------------------
    # Variant C: Frequency-Only (Primary Face Crop)
    # -------------------------------------------------------------
    clf_c = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(X_train_crop_scaled, y_train)
    val_scores_c = list(clf_c.predict_proba(X_val_crop_scaled)[:, 1])
    val_preds_c = [1 if p >= 0.50 else 0 for p in val_scores_c]
    metrics_c = compute_metrics(list(y_val), val_preds_c, val_scores_c)
    
    # -------------------------------------------------------------
    # Variant D: Fused (Xception + Full Canvas Frequency)
    # -------------------------------------------------------------
    X_train_fused_d = np.hstack([train_xc_logits, X_train_full_scaled])
    X_val_fused_d = np.hstack([val_xc_logits, X_val_full_scaled])
    clf_d = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(X_train_fused_d, y_train)
    val_scores_d = list(clf_d.predict_proba(X_val_fused_d)[:, 1])
    val_preds_d = [1 if p >= 0.50 else 0 for p in val_scores_d]
    metrics_d = compute_metrics(list(y_val), val_preds_d, val_scores_d)
    
    # -------------------------------------------------------------
    # Variant E: Fused (Xception + Primary Face Crop Frequency)
    # -------------------------------------------------------------
    X_train_fused_e = np.hstack([train_xc_logits, X_train_crop_scaled])
    X_val_fused_e = np.hstack([val_xc_logits, X_val_crop_scaled])
    clf_e = LogisticRegression(C=1.0, random_state=42, max_iter=1000).fit(X_train_fused_e, y_train)
    val_scores_e = list(clf_e.predict_proba(X_val_fused_e)[:, 1])
    val_preds_e = [1 if p >= 0.50 else 0 for p in val_scores_e]
    metrics_e = compute_metrics(list(y_val), val_preds_e, val_scores_e)
    
    # 3. Pair-Aware Cluster Bootstrap (B=2,000, 50 pairs resampled)
    print("Running Pair-Aware Cluster Bootstrap on all 5 variants...")
    val_pair_records = {}
    for idx in range(len(val_recs)):
        pid = val_pairs[idx]
        if pid not in val_pair_records:
            val_pair_records[pid] = []
        val_pair_records[pid].append({
            "idx": idx, "ground_truth": int(y_val[idx]),
            "a_p": val_preds_a[idx], "a_s": val_scores_a[idx],
            "b_p": val_preds_b[idx], "b_s": val_scores_b[idx],
            "c_p": val_preds_c[idx], "c_s": val_scores_c[idx],
            "d_p": val_preds_d[idx], "d_s": val_scores_d[idx],
            "e_p": val_preds_e[idx], "e_s": val_scores_e[idx]
        })
        
    unique_pairs = list(val_pair_records.keys())
    np.random.seed(42)
    boot_accs = {k: [] for k in ["A", "B", "C", "D", "E"]}
    boot_recs = {k: [] for k in ["A", "B", "C", "D", "E"]}
    boot_specs = {k: [] for k in ["A", "B", "C", "D", "E"]}
    boot_aucs = {k: [] for k in ["A", "B", "C", "D", "E"]}
    delta_e_minus_a = {"acc": [], "rec": [], "spec": [], "auc": []}
    delta_e_minus_d = {"acc": [], "rec": [], "spec": [], "auc": []}
    
    for _ in range(2000):
        sampled_pids = np.random.choice(unique_pairs, size=50, replace=True)
        b_samples = []
        for pid in sampled_pids:
            b_samples.extend(val_pair_records[pid])
        by = [s["ground_truth"] for s in b_samples]
        
        for var_key, p_key, s_key in [
            ("A", "a_p", "a_s"), ("B", "b_p", "b_s"), ("C", "c_p", "c_s"),
            ("D", "d_p", "d_s"), ("E", "e_p", "e_s")
        ]:
            bp = [s[p_key] for s in b_samples]
            bs = [s[s_key] for s in b_samples]
            m = compute_metrics(by, bp, bs)
            boot_accs[var_key].append(m["accuracy"])
            boot_recs[var_key].append(m["recall"])
            boot_specs[var_key].append(m["specificity"])
            boot_aucs[var_key].append(m["roc_auc"])
            
        # Paired Deltas
        delta_e_minus_a["acc"].append(boot_accs["E"][-1] - boot_accs["A"][-1])
        delta_e_minus_a["rec"].append(boot_recs["E"][-1] - boot_recs["A"][-1])
        delta_e_minus_a["spec"].append(boot_specs["E"][-1] - boot_specs["A"][-1])
        delta_e_minus_a["auc"].append(boot_aucs["E"][-1] - boot_aucs["A"][-1])
        
        delta_e_minus_d["acc"].append(boot_accs["E"][-1] - boot_accs["D"][-1])
        delta_e_minus_d["rec"].append(boot_recs["E"][-1] - boot_recs["D"][-1])
        delta_e_minus_d["spec"].append(boot_specs["E"][-1] - boot_specs["D"][-1])
        delta_e_minus_d["auc"].append(boot_aucs["E"][-1] - boot_aucs["D"][-1])
        
    def ci95(arr):
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]
        
    ablation_results = {
        "ablation_title": "DEEPTRACE Phase 2C Input-Representation Ablation (Validation Split N=100)",
        "protocol": {
            "training_set": "development_train (Pairs 0050-0149, N=200)",
            "validation_set": "development_validation (Pairs 0150-0199, N=100)",
            "validation_status": "EXPLORATORY_WITHIN_GENERATOR_EVALUATION",
            "decision_threshold": DEEPFAKE_THRESHOLD,
            "resampling_protocol": "Pair-aware cluster bootstrap (B=2000 replicates, 50 pairs)"
        },
        "variants": {
            "variant_a_xception_baseline": {
                "description": "Frozen Spatial Xception Baseline (Unchanged production inference)",
                "metrics": metrics_a,
                "bootstrap_95ci": {
                    "accuracy": ci95(boot_accs["A"]),
                    "recall": ci95(boot_recs["A"]),
                    "specificity": ci95(boot_specs["A"]),
                    "roc_auc": ci95(boot_aucs["A"])
                }
            },
            "variant_b_freq_full_canvas": {
                "description": "Frequency-Only, Full 256x256 Image Canvas Features",
                "metrics": metrics_b,
                "bootstrap_95ci": {
                    "accuracy": ci95(boot_accs["B"]),
                    "recall": ci95(boot_recs["B"]),
                    "specificity": ci95(boot_specs["B"]),
                    "roc_auc": ci95(boot_aucs["B"])
                }
            },
            "variant_c_freq_primary_face_crop": {
                "description": "Frequency-Only, Primary Face Crop Features (YuNet + 0.20 padding crop)",
                "metrics": metrics_c,
                "bootstrap_95ci": {
                    "accuracy": ci95(boot_accs["C"]),
                    "recall": ci95(boot_recs["C"]),
                    "specificity": ci95(boot_specs["C"]),
                    "roc_auc": ci95(boot_aucs["C"])
                }
            },
            "variant_d_fused_full_canvas": {
                "description": "Xception Logit + Full Canvas Frequency Feature Fusion",
                "metrics": metrics_d,
                "bootstrap_95ci": {
                    "accuracy": ci95(boot_accs["D"]),
                    "recall": ci95(boot_recs["D"]),
                    "specificity": ci95(boot_specs["D"]),
                    "roc_auc": ci95(boot_aucs["D"])
                }
            },
            "variant_e_fused_primary_face_crop": {
                "description": "Xception Logit + Primary Face Crop Frequency Feature Fusion",
                "metrics": metrics_e,
                "bootstrap_95ci": {
                    "accuracy": ci95(boot_accs["E"]),
                    "recall": ci95(boot_recs["E"]),
                    "specificity": ci95(boot_specs["E"]),
                    "roc_auc": ci95(boot_aucs["E"])
                }
            }
        },
        "paired_comparisons": {
            "variant_e_vs_variant_a": {
                "description": "Fused Face-Crop (Variant E) minus Frozen Xception (Variant A)",
                "delta_accuracy_point": round(metrics_e["accuracy"] - metrics_a["accuracy"], 4),
                "delta_accuracy_95ci": ci95(delta_e_minus_a["acc"]),
                "delta_recall_point": round(metrics_e["recall"] - metrics_a["recall"], 4),
                "delta_recall_95ci": ci95(delta_e_minus_a["rec"]),
                "delta_specificity_point": round(metrics_e["specificity"] - metrics_a["specificity"], 4),
                "delta_specificity_95ci": ci95(delta_e_minus_a["spec"]),
                "delta_roc_auc_point": round(metrics_e["roc_auc"] - metrics_a["roc_auc"], 4),
                "delta_roc_auc_95ci": ci95(delta_e_minus_a["auc"])
            },
            "variant_e_vs_variant_d": {
                "description": "Fused Face-Crop (Variant E) minus Fused Full-Canvas (Variant D)",
                "delta_accuracy_point": round(metrics_e["accuracy"] - metrics_d["accuracy"], 4),
                "delta_accuracy_95ci": ci95(delta_e_minus_d["acc"]),
                "delta_recall_point": round(metrics_e["recall"] - metrics_d["recall"], 4),
                "delta_recall_95ci": ci95(delta_e_minus_d["rec"]),
                "delta_specificity_point": round(metrics_e["specificity"] - metrics_d["specificity"], 4),
                "delta_specificity_95ci": ci95(delta_e_minus_d["spec"]),
                "delta_roc_auc_point": round(metrics_e["roc_auc"] - metrics_d["roc_auc"], 4),
                "delta_roc_auc_95ci": ci95(delta_e_minus_d["auc"])
            }
        }
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
        
    print(f"\nAblation JSON written to: {OUTPUT_JSON_PATH}")
    print("\n--- Summary of 5 Variants on Validation Set ---")
    print(f"Variant A (Xception Baseline):       Acc={metrics_a['accuracy']:.2%}, Rec={metrics_a['recall']:.2%}, Spec={metrics_a['specificity']:.2%}, AUC={metrics_a['roc_auc']:.4f}")
    print(f"Variant B (Freq Full Canvas):        Acc={metrics_b['accuracy']:.2%}, Rec={metrics_b['recall']:.2%}, Spec={metrics_b['specificity']:.2%}, AUC={metrics_b['roc_auc']:.4f}")
    print(f"Variant C (Freq Face Crop):          Acc={metrics_c['accuracy']:.2%}, Rec={metrics_c['recall']:.2%}, Spec={metrics_c['specificity']:.2%}, AUC={metrics_c['roc_auc']:.4f}")
    print(f"Variant D (Fused Full Canvas):       Acc={metrics_d['accuracy']:.2%}, Rec={metrics_d['recall']:.2%}, Spec={metrics_d['specificity']:.2%}, AUC={metrics_d['roc_auc']:.4f}")
    print(f"Variant E (Fused Face Crop):         Acc={metrics_e['accuracy']:.2%}, Rec={metrics_e['recall']:.2%}, Spec={metrics_e['specificity']:.2%}, AUC={metrics_e['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
