"""
DEEPTRACE - Reproducible Model Evaluation & Prediction Reliability Audit
========================================================================
Performs rigorous, reproducible, and transparent model evaluation across
all unique media fixtures, evaluating Primary Face mode vs All Faces mode,
checking for data leakage, grouping by dataset and manipulation type,
and assessing probability calibration limits honestly.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.face_service import detect_and_crop_faces
from app.services.detection_service import (
    get_xception_detector,
    analyze_image,
    analyze_video,
    DEEPFAKE_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_audit")


def calculate_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    """Calculates deterministic classification metrics with safe division handling."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "total_samples": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4)
    }


def evaluate_manifest(manifest_path: Path) -> Dict[str, Any]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    logger.info(f"Loaded {len(samples)} total manifest entries from {manifest_path.name}")

    evaluated_records = []
    excluded_records = []

    for s in samples:
        sample_id = s["sample_id"]
        rel_path = s["file_path"]
        media_type = s["media_type"]
        gt = s["ground_truth"]
        eligible = s.get("is_eligible_for_detection_metrics", True)
        leakage = s.get("leakage_flag", False)

        resolved_path = (BACKEND_DIR / rel_path).resolve()
        if not resolved_path.exists():
            resolved_path = (REPO_ROOT / rel_path).resolve()

        if not resolved_path.exists():
            excluded_records.append({
                "sample_id": sample_id,
                "filename": s["filename"],
                "reason": f"File path does not exist on disk: {rel_path}"
            })
            continue

        if not eligible:
            excluded_records.append({
                "sample_id": sample_id,
                "filename": s["filename"],
                "reason": s.get("exclusion_reason", "Marked ineligible in manifest"),
                "ground_truth": gt,
                "leakage_flag": leakage
            })
            continue

        # Run inference
        if media_type == "image":
            d_info = detect_and_crop_faces(resolved_path)
            res = analyze_image(resolved_path, detection_info=d_info)

            if res.status == "no_face_detected":
                excluded_records.append({
                    "sample_id": sample_id,
                    "filename": s["filename"],
                    "reason": "No face detected by YuNet; model returns null probabilities.",
                    "ground_truth": gt
                })
                continue

            pf = res.primary_face_result
            af = res.all_faces_result

            pf_verdict = pf.prediction if pf else res.prediction
            pf_prob = pf.probability if pf else res.model_probability
            pf_fake_prob = pf.fake_probability if pf else (pf_prob if pf_verdict == "fake" else 1.0 - pf_prob)
            pf_real_prob = pf.real_probability if pf else (pf_prob if pf_verdict == "real" else 1.0 - pf_prob)
            pf_is_low_margin = pf.is_low_margin if pf else res.is_low_margin

            af_verdict = af.prediction if af else res.prediction
            af_prob = af.probability if af else res.model_probability
            af_fake_prob = af.mean_fake_probability if af else pf_fake_prob
            af_real_prob = af.mean_real_probability if af else pf_real_prob
            af_is_low_margin = af.is_low_margin if af else res.is_low_margin

            rec = {
                "sample_id": sample_id,
                "filename": s["filename"],
                "media_type": "image",
                "ground_truth": gt,
                "source_dataset": s.get("source_dataset", "Unknown"),
                "manipulation_type": s.get("manipulation_type", "none"),
                "verification_status": s.get("verification_status", "unverified"),
                "leakage_flag": leakage,
                "face_count": d_info["count"],
                # Primary Face results
                "pf_verdict": pf_verdict,
                "pf_prob": pf_prob,
                "pf_fake_prob": pf_fake_prob,
                "pf_real_prob": pf_real_prob,
                "pf_is_low_margin": pf_is_low_margin,
                "pf_correct": (pf_verdict == "fake" and gt == "manipulated") or (pf_verdict == "real" and gt == "authentic"),
                # All Faces results
                "af_verdict": af_verdict,
                "af_prob": af_prob,
                "af_fake_prob": af_fake_prob,
                "af_real_prob": af_real_prob,
                "af_is_low_margin": af_is_low_margin,
                "af_correct": (af_verdict == "fake" and gt == "manipulated") or (af_verdict == "real" and gt == "authentic")
            }
            evaluated_records.append(rec)

        elif media_type == "video":
            vres = analyze_video(resolved_path)
            if vres.status == "no_face_detected":
                excluded_records.append({
                    "sample_id": sample_id,
                    "filename": s["filename"],
                    "reason": "No faces detected across sampled video frames.",
                    "ground_truth": gt
                })
                continue

            v_pred = vres.prediction
            v_prob = vres.probability
            v_correct = (v_pred == "fake" and gt == "manipulated") or (v_pred == "real" and gt == "authentic")

            rec = {
                "sample_id": sample_id,
                "filename": s["filename"],
                "media_type": "video",
                "ground_truth": gt,
                "source_dataset": s.get("source_dataset", "Unknown"),
                "manipulation_type": s.get("manipulation_type", "none"),
                "verification_status": s.get("verification_status", "unverified"),
                "leakage_flag": leakage,
                "face_count": vres.faces_detected or 1,
                "pf_verdict": v_pred,
                "pf_prob": v_prob,
                "pf_fake_prob": v_prob if v_pred == "fake" else round(1.0 - v_prob, 4),
                "pf_real_prob": v_prob if v_pred == "real" else round(1.0 - v_prob, 4),
                "pf_is_low_margin": (abs(v_prob - 0.50) < 0.05),
                "pf_correct": v_correct,
                "af_verdict": v_pred,
                "af_prob": v_prob,
                "af_fake_prob": v_prob if v_pred == "fake" else round(1.0 - v_prob, 4),
                "af_real_prob": v_prob if v_pred == "real" else round(1.0 - v_prob, 4),
                "af_is_low_margin": (abs(v_prob - 0.50) < 0.05),
                "af_correct": v_correct
            }
            evaluated_records.append(rec)

    # Compute overall metrics for Primary Face mode
    pf_tp = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "fake")
    pf_fp = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["pf_verdict"] == "fake")
    pf_tn = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["pf_verdict"] == "real")
    pf_fn = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "real")
    pf_metrics = calculate_binary_metrics(pf_tp, pf_fp, pf_tn, pf_fn)

    # Compute overall metrics for All Faces mode
    af_tp = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["af_verdict"] == "fake")
    af_fp = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["af_verdict"] == "fake")
    af_tn = sum(1 for r in evaluated_records if r["ground_truth"] == "authentic" and r["af_verdict"] == "real")
    af_fn = sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated" and r["af_verdict"] == "real")
    af_metrics = calculate_binary_metrics(af_tp, af_fp, af_tn, af_fn)

    # Compute ROC-AUC if applicable
    y_true = [1 if r["ground_truth"] == "manipulated" else 0 for r in evaluated_records]
    y_scores_pf = [r["pf_fake_prob"] for r in evaluated_records]
    y_scores_af = [r["af_fake_prob"] for r in evaluated_records]

    from sklearn.metrics import roc_auc_score
    try:
        if len(set(y_true)) >= 2:
            roc_auc_pf = round(float(roc_auc_score(y_true, y_scores_pf)), 4)
            roc_auc_af = round(float(roc_auc_score(y_true, y_scores_af)), 4)
        else:
            roc_auc_pf = None
            roc_auc_af = None
    except Exception:
        roc_auc_pf = None
        roc_auc_af = None

    pf_metrics["roc_auc"] = roc_auc_pf
    af_metrics["roc_auc"] = roc_auc_af

    # Subgroup metrics: by source dataset
    dataset_groups = {}
    for r in evaluated_records:
        ds = r["source_dataset"]
        dataset_groups.setdefault(ds, []).append(r)

    dataset_metrics = {}
    for ds, recs in dataset_groups.items():
        tp = sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "fake")
        fp = sum(1 for r in recs if r["ground_truth"] == "authentic" and r["pf_verdict"] == "fake")
        tn = sum(1 for r in recs if r["ground_truth"] == "authentic" and r["pf_verdict"] == "real")
        fn = sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "real")
        dataset_metrics[ds] = calculate_binary_metrics(tp, fp, tn, fn)

    # Subgroup metrics: by manipulation type
    manip_groups = {}
    for r in evaluated_records:
        mt = r["manipulation_type"]
        manip_groups.setdefault(mt, []).append(r)

    manip_metrics = {}
    for mt, recs in manip_groups.items():
        tp = sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "fake")
        fp = sum(1 for r in recs if r["ground_truth"] == "authentic" and r["pf_verdict"] == "fake")
        tn = sum(1 for r in recs if r["ground_truth"] == "authentic" and r["pf_verdict"] == "real")
        fn = sum(1 for r in recs if r["ground_truth"] == "manipulated" and r["pf_verdict"] == "real")
        manip_metrics[mt] = calculate_binary_metrics(tp, fp, tn, fn)

    # Low-margin predictions check
    low_margin_count = sum(1 for r in evaluated_records if r["pf_is_low_margin"])

    report = {
        "manifest_name": manifest.get("dataset_name"),
        "total_manifest_samples": len(samples),
        "total_eligible_samples": len(evaluated_records),
        "total_excluded_samples": len(excluded_records),
        "class_counts": {
            "authentic": sum(1 for r in evaluated_records if r["ground_truth"] == "authentic"),
            "manipulated": sum(1 for r in evaluated_records if r["ground_truth"] == "manipulated")
        },
        "primary_face_metrics": pf_metrics,
        "all_faces_metrics": af_metrics,
        "subgroup_by_dataset": dataset_metrics,
        "subgroup_by_manipulation_type": manip_metrics,
        "low_margin_samples_count": low_margin_count,
        "evaluated_samples": evaluated_records,
        "excluded_samples": excluded_records,
        "scientific_integrity_assessment": {
            "sample_size_adequate_for_generalization": False,
            "sample_size_note": (
                f"With only N={len(evaluated_records)} eligible samples ({pf_metrics['tp'] + pf_metrics['fn']} manipulated, "
                f"{pf_metrics['tn'] + pf_metrics['fp']} authentic), the empirical sample size is far too small to establish "
                "statistically sound cross-dataset generalization accuracy or reliable confidence bounds."
            ),
            "probability_calibration_feasible": False,
            "calibration_note": (
                "Calibration assessment (e.g. Expected Calibration Error over 10 bins) cannot be meaningfully evaluated "
                f"on N={len(evaluated_records)} samples. Softmax probabilities reflect raw decision margins, not calibrated posteriors."
            ),
            "data_leakage_detected": True,
            "data_leakage_details": [
                "all_in_one.jpg resides directly in DeepfakeBench training library; excluded from clean evaluation.",
                "Nithin_sist_friend.jpeg was used as an ad-hoc target during earlier crop padding ratio tuning (0.15 -> 0.20)."
            ]
        }
    }

    return report


def print_evaluation_summary(report: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("DEEPTRACE INDEPENDENT MODEL EVALUATION & RELIABILITY AUDIT REPORT")
    print("=" * 80)
    print(f"Manifest: {report['manifest_name']}")
    print(f"Total Evaluated Eligible Samples: {report['total_eligible_samples']}")
    print(f"  Authentic: {report['class_counts']['authentic']}")
    print(f"  Manipulated: {report['class_counts']['manipulated']}")
    print(f"Total Excluded Samples: {report['total_excluded_samples']}")
    print("-" * 80)

    print("\n[1. PRIMARY FACE MODE METRICS (Default Standard)]")
    pf = report["primary_face_metrics"]
    print(f"  Confusion Matrix: TP={pf['tp']}, FP={pf['fp']}, TN={pf['tn']}, FN={pf['fn']}")
    print(f"  Accuracy:         {pf['accuracy'] * 100:.2f}%")
    print(f"  Precision:        {pf['precision'] * 100:.2f}%")
    print(f"  Recall (Sens):    {pf['recall'] * 100:.2f}%")
    print(f"  Specificity:      {pf['specificity'] * 100:.2f}%")
    print(f"  F1 Score:         {pf['f1_score']:.4f}")
    print(f"  FPR:              {pf['false_positive_rate'] * 100:.2f}%")
    print(f"  FNR:              {pf['false_negative_rate'] * 100:.2f}%")
    print(f"  ROC-AUC:          {pf['roc_auc'] if pf['roc_auc'] is not None else 'N/A (insufficient sample size)'}")

    print("\n[2. ALL FACES MODE METRICS (Comparison)]")
    af = report["all_faces_metrics"]
    print(f"  Confusion Matrix: TP={af['tp']}, FP={af['fp']}, TN={af['tn']}, FN={af['fn']}")
    print(f"  Accuracy:         {af['accuracy'] * 100:.2f}%")
    print(f"  Precision:        {af['precision'] * 100:.2f}%")
    print(f"  Recall (Sens):    {af['recall'] * 100:.2f}%")
    print(f"  Specificity:      {af['specificity'] * 100:.2f}%")
    print(f"  F1 Score:         {af['f1_score']:.4f}")
    print(f"  FPR:              {af['false_positive_rate'] * 100:.2f}%")
    print(f"  FNR:              {af['false_negative_rate'] * 100:.2f}%")
    print(f"  ROC-AUC:          {af['roc_auc'] if af['roc_auc'] is not None else 'N/A'}")

    print("\n[3. BREAKDOWN BY SOURCE DATASET]")
    for ds, m in report["subgroup_by_dataset"].items():
        print(f"  - {ds:30} (N={m['total_samples']}): Acc={m['accuracy']*100:.1f}%, Prec={m['precision']*100:.1f}%, Rec={m['recall']*100:.1f}%")

    print("\n[4. BREAKDOWN BY MANIPULATION TYPE]")
    for mt, m in report["subgroup_by_manipulation_type"].items():
        print(f"  - {mt:30} (N={m['total_samples']}): Acc={m['accuracy']*100:.1f}%, Prec={m['precision']*100:.1f}%, Rec={m['recall']*100:.1f}%")

    print("\n[5. EXCLUDED SAMPLES & REASONS]")
    for ex in report["excluded_samples"]:
        print(f"  - {ex['filename']:25} | Reason: {ex['reason']}")

    print("\n[6. LOW-MARGIN PREDICTIONS ADVISORY]")
    print(f"  Low-margin samples detected: {report['low_margin_samples_count']}")
    for r in report["evaluated_samples"]:
        if r["pf_is_low_margin"]:
            print(f"  - {r['filename']:20}: P(manipulated)={r['pf_fake_prob']:.4f}, Winning Verdict={r['pf_verdict']} (Low-margin product warning triggered)")

    print("\n[7. SCIENTIFIC REPRODUCIBILITY ASSESSMENT]")
    integ = report["scientific_integrity_assessment"]
    print(f"  Generalization Established: {integ['sample_size_adequate_for_generalization']}")
    print(f"  Note: {integ['sample_size_note']}")
    print(f"  Calibration Note: {integ['calibration_note']}")
    print(f"  Data Leakage Flags: {integ['data_leakage_details']}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DEEPTRACE Independent Model Evaluation")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "manifests" / "provenance_audit_manifest.json"),
        help="Path to evaluation dataset manifest JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BACKEND_DIR / "evaluation" / "results" / "reproducible_evaluation_results.json"),
        help="Path to save evaluation output JSON"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = evaluate_manifest(manifest_path)
    print_evaluation_summary(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report successfully saved to {output_path}")


if __name__ == "__main__":
    main()
