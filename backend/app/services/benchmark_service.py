import os
import json
import time
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from ..config import (
    EVALUATION_DIR,
    DATASETS_DIR,
    MANIFESTS_DIR,
    BENCHMARK_RESULTS_DIR,
    BENCHMARK_REPORTS_DIR,
    DEEPFAKE_THRESHOLD,
    MAX_BENCHMARK_SAMPLES,
    UPLOADS_DIR
)
from ..utils.file_utils import is_safe_path, sanitize_error_message
from ..models.benchmark_schemas import (
    ManifestSample,
    DatasetManifest,
    SampleEvaluationResult,
    ConfusionMatrix,
    ThresholdSweepItem,
    EvaluationMetrics,
    ErrorAnalysisItem,
    SubgroupMetricItem,
    BenchmarkRunSummary,
    BenchmarkRunDetail
)
from ..utils.logger import logger
from .face_service import detect_and_crop_faces
from .detection_service import predict_face_crop, analyze_video, get_xception_detector


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash digest of a file in 64KB streaming blocks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


APPROVED_BENCHMARK_ROOTS = [
    MANIFESTS_DIR.resolve(),
    DATASETS_DIR.resolve(),
    EVALUATION_DIR.resolve(),
    UPLOADS_DIR.resolve(),
    Path(tempfile.gettempdir()).resolve(),
]


def resolve_sample_path(file_path_str: str) -> Path:
    """
    Safely resolves a sample file path preventing path traversal outside approved trees.
    Requires target to reside strictly within MANIFESTS_DIR, DATASETS_DIR, EVALUATION_DIR, or UPLOADS_DIR.
    """
    candidate = Path(file_path_str)

    # If absolute, verify it resides strictly within APPROVED_BENCHMARK_ROOTS
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not any(resolved == root or root in resolved.parents for root in APPROVED_BENCHMARK_ROOTS):
            raise PermissionError(f"Access to sample path outside approved benchmark directories is forbidden: {file_path_str}")
        return resolved

    # Try relative to MANIFESTS_DIR, DATASETS_DIR, EVALUATION_DIR, UPLOADS_DIR
    for base in (MANIFESTS_DIR, DATASETS_DIR, EVALUATION_DIR, UPLOADS_DIR):
        resolved = (base / file_path_str).resolve()
        if any(resolved == root or root in resolved.parents for root in APPROVED_BENCHMARK_ROOTS) and resolved.exists():
            return resolved

    # Default relative to MANIFESTS_DIR and verify safety
    resolved = (MANIFESTS_DIR / file_path_str).resolve()
    if not any(resolved == root or root in resolved.parents for root in APPROVED_BENCHMARK_ROOTS):
        raise PermissionError(f"Access to sample path outside approved benchmark directories is forbidden: {file_path_str}")
    return resolved


def validate_manifest(manifest: DatasetManifest) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates manifest integrity, computes SHA-256 checks, detects duplicate files,
    and performs cross-split data leakage detection.
    """
    warnings: List[str] = []
    stats: Dict[str, Any] = {
        "total_samples": len(manifest.samples),
        "authentic_count": 0,
        "manipulated_count": 0,
        "image_count": 0,
        "video_count": 0,
        "splits": set()
    }

    if not manifest.samples:
        return False, ["Manifest contains zero samples."], stats

    sha_to_samples: Dict[str, List[ManifestSample]] = {}
    split_to_shas: Dict[str, set] = {}

    for s in manifest.samples:
        gt = s.ground_truth.lower().strip()
        if gt not in ("authentic", "manipulated"):
            return False, [f"Invalid ground_truth '{s.ground_truth}' for sample '{s.sample_id}'. Must be 'authentic' or 'manipulated'."], stats

        if gt == "authentic":
            stats["authentic_count"] += 1
        else:
            stats["manipulated_count"] += 1

        m_type = s.media_type.lower().strip()
        if m_type == "video":
            stats["video_count"] += 1
        else:
            stats["image_count"] += 1

        split = (s.split or "test").lower().strip()
        stats["splits"].add(split)

        # Hash tracking
        if s.sha256:
            sha_clean = s.sha256.lower().strip()
            sha_to_samples.setdefault(sha_clean, []).append(s)
            split_to_shas.setdefault(split, set()).add(sha_clean)

    # 1. Duplicate SHA-256 check
    duplicate_count = sum(len(samples) - 1 for samples in sha_to_samples.values() if len(samples) > 1)
    if duplicate_count > 0:
        dup_shas = [sha for sha, samples in sha_to_samples.items() if len(samples) > 1]
        warnings.append(
            f"Dataset duplicate warning: {duplicate_count} duplicate file(s) detected across {len(dup_shas)} distinct SHA-256 hash(es)."
        )

    # 2. Cross-split data leakage check
    if len(split_to_shas) > 1:
        leaked_shas = set()
        splits_list = list(split_to_shas.keys())
        for i in range(len(splits_list)):
            for j in range(i + 1, len(splits_list)):
                sp1, sp2 = splits_list[i], splits_list[j]
                intersection = split_to_shas[sp1].intersection(split_to_shas[sp2])
                if intersection:
                    leaked_shas.update(intersection)
                    warnings.append(
                        f"Train/Test leakage warning: {len(intersection)} identical file(s) appear in both '{sp1}' and '{sp2}' splits."
                    )

    stats["splits"] = list(stats["splits"])
    return True, warnings, stats


def compute_metrics_at_threshold(
    evaluated_samples: List[SampleEvaluationResult],
    threshold: float
) -> Tuple[EvaluationMetrics, ConfusionMatrix]:
    """
    Computes deterministic binary classification metrics and confusion matrix
    at a specific decision threshold T in [0, 1].
    Positive class = 'manipulated', Negative class = 'authentic'.
    Safe zero-denominator handling guarantees no NaN or misleading values.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    valid_for_metrics = [
        s for s in evaluated_samples
        if s.inference_status == "completed" and s.manipulated_probability is not None
    ]

    for s in valid_for_metrics:
        is_actual_manipulated = (s.ground_truth.lower().strip() == "manipulated")
        is_pred_manipulated = (s.manipulated_probability >= threshold)

        if is_actual_manipulated and is_pred_manipulated:
            tp += 1
        elif not is_actual_manipulated and not is_pred_manipulated:
            tn += 1
        elif not is_actual_manipulated and is_pred_manipulated:
            fp += 1
        elif is_actual_manipulated and not is_pred_manipulated:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Rates for confusion matrix
    tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fn_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    conf_matrix = ConfusionMatrix(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        tp_rate=round(tp_rate, 4),
        tn_rate=round(tn_rate, 4),
        fp_rate=round(fp_rate, 4),
        fn_rate=round(fn_rate, 4)
    )

    # Compute ROC-AUC and PR-AUC if continuous probabilities and both classes exist
    roc_auc = None
    pr_auc = None
    roc_status = "not_available"
    roc_explanation = None

    y_true = [1 if s.ground_truth.lower().strip() == "manipulated" else 0 for s in valid_for_metrics]
    y_scores = [s.manipulated_probability for s in valid_for_metrics]

    if len(set(y_true)) >= 2:
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_scores)), 4)
            p_curve, r_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = round(float(auc(r_curve, p_curve)), 4)
            roc_status = "available"
        except Exception as e:
            roc_status = "not_available"
            roc_explanation = f"ROC calculation encountered numerical issue: {e}"
    else:
        roc_status = "not_available"
        roc_explanation = "ROC-AUC and PR-AUC require samples from both authentic and manipulated classes; dataset contains only one class."

    metrics = EvaluationMetrics(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        specificity=round(specificity, 4),
        f1_score=round(f1, 4),
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        roc_auc_status=roc_status,
        roc_auc_explanation=roc_explanation
    )

    return metrics, conf_matrix


def run_threshold_sweep(
    evaluated_samples: List[SampleEvaluationResult],
    threshold_steps: Optional[List[float]] = None
) -> List[ThresholdSweepItem]:
    """
    Evaluates detector performance across multiple decision thresholds.
    Standard sweep: [0.10, 0.15, 0.20, ..., 0.90].
    """
    if threshold_steps is None:
        threshold_steps = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    sweep_results: List[ThresholdSweepItem] = []
    for t in threshold_steps:
        m, cm = compute_metrics_at_threshold(evaluated_samples, t)
        sweep_results.append(
            ThresholdSweepItem(
                threshold=t,
                accuracy=m.accuracy,
                precision=m.precision,
                recall=m.recall,
                specificity=m.specificity,
                f1=m.f1_score,
                fpr=m.false_positive_rate,
                fnr=m.false_negative_rate,
                tp=cm.tp,
                fp=cm.fp,
                tn=cm.tn,
                fn=cm.fn
            )
        )
    return sweep_results


def extract_error_analysis(
    evaluated_samples: List[SampleEvaluationResult],
    threshold: float
) -> List[ErrorAnalysisItem]:
    """
    Extracts all False Positives (authentic -> manipulated) and
    False Negatives (manipulated -> authentic), sorted by probability deviation.
    """
    errors: List[ErrorAnalysisItem] = []
    for s in evaluated_samples:
        if s.inference_status != "completed" or s.manipulated_probability is None:
            continue

        gt = s.ground_truth.lower().strip()
        is_pred_manipulated = (s.manipulated_probability >= threshold)

        if gt == "authentic" and is_pred_manipulated:
            errors.append(
                ErrorAnalysisItem(
                    sample_id=s.sample_id,
                    filename=s.filename,
                    error_type="false_positive",
                    ground_truth="authentic",
                    predicted_class="manipulated",
                    manipulated_probability=s.manipulated_probability,
                    face_count=s.face_count,
                    resolution=s.resolution,
                    sha256=s.sha256,
                    source=s.source,
                    dataset_name=None
                )
            )
        elif gt == "manipulated" and not is_pred_manipulated:
            errors.append(
                ErrorAnalysisItem(
                    sample_id=s.sample_id,
                    filename=s.filename,
                    error_type="false_negative",
                    ground_truth="manipulated",
                    predicted_class="authentic",
                    manipulated_probability=s.manipulated_probability,
                    face_count=s.face_count,
                    resolution=s.resolution,
                    sha256=s.sha256,
                    source=s.source,
                    dataset_name=None
                )
            )

    # Sort False Positives by highest manipulated probability first
    # Sort False Negatives by lowest manipulated probability first
    errors.sort(
        key=lambda e: (
            0 if e.error_type == "false_positive" else 1,
            -e.manipulated_probability if e.error_type == "false_positive" else e.manipulated_probability
        )
    )
    return errors


def compute_subgroup_analysis(
    evaluated_samples: List[SampleEvaluationResult],
    threshold: float
) -> List[SubgroupMetricItem]:
    """
    Calculates metrics across conditions: face count, resolution, and manipulation type
    when sufficient samples (>= 2) exist.
    """
    subgroups: List[SubgroupMetricItem] = []

    # 1. Face count subgroups
    face_groups: Dict[str, List[SampleEvaluationResult]] = {
        "Single Face": [],
        "Multiple Faces": []
    }
    for s in evaluated_samples:
        if s.inference_status == "completed":
            if s.face_count == 1:
                face_groups["Single Face"].append(s)
            elif s.face_count > 1:
                face_groups["Multiple Faces"].append(s)

    for cat, s_list in face_groups.items():
        if len(s_list) >= 2:
            m, _ = compute_metrics_at_threshold(s_list, threshold)
            subgroups.append(
                SubgroupMetricItem(
                    subgroup_type="face_count",
                    category=cat,
                    sample_count=len(s_list),
                    accuracy=m.accuracy,
                    precision=m.precision,
                    recall=m.recall,
                    f1=m.f1_score
                )
            )

    # 2. Manipulation type subgroups
    manip_groups: Dict[str, List[SampleEvaluationResult]] = {}
    for s in evaluated_samples:
        if s.inference_status == "completed" and s.manipulation_type and s.manipulation_type != "none":
            manip_groups.setdefault(s.manipulation_type, []).append(s)

    for m_type, s_list in manip_groups.items():
        if len(s_list) >= 2:
            m, _ = compute_metrics_at_threshold(s_list, threshold)
            subgroups.append(
                SubgroupMetricItem(
                    subgroup_type="manipulation_type",
                    category=m_type,
                    sample_count=len(s_list),
                    accuracy=m.accuracy,
                    precision=m.precision,
                    recall=m.recall,
                    f1=m.f1_score
                )
            )

    return subgroups


class BenchmarkRunner:
    """
    Reproducible benchmark execution harness.
    Evaluates labeled media manifests using the exact production Xception pipeline
    without modifying production thresholds or reloading model checkpoints per sample.
    """

    def __init__(self, threshold: float = DEEPFAKE_THRESHOLD):
        self.threshold = threshold
        # Warm the singleton Xception model once at runner initialization
        get_xception_detector()

    def evaluate_sample(self, sample: ManifestSample) -> SampleEvaluationResult:
        """Evaluates a single labeled sample with error containment."""
        start_time = time.perf_counter()
        target_path = resolve_sample_path(sample.file_path)

        if not target_path.exists():
            return SampleEvaluationResult(
                sample_id=sample.sample_id,
                filename=Path(sample.file_path).name,
                file_path=sample.file_path,
                media_type=sample.media_type,
                ground_truth=sample.ground_truth,
                inference_status="error",
                error_message=f"File not found: {sample.file_path}",
                inference_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                resolution=sample.resolution,
                source=sample.source,
                split=sample.split,
                manipulation_type=sample.manipulation_type,
                sha256=sample.sha256
            )

        # Compute SHA-256 if not provided
        file_sha = sample.sha256
        if not file_sha:
            try:
                file_sha = compute_file_sha256(target_path)
            except Exception:
                file_sha = None

        # Execute evaluation based on media type
        try:
            if sample.media_type.lower().strip() == "video":
                # Video evaluation using production analyze_video
                det_res = analyze_video(target_path)
                inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if det_res.status == "no_face_detected":
                    return SampleEvaluationResult(
                        sample_id=sample.sample_id,
                        filename=target_path.name,
                        file_path=str(target_path),
                        media_type="video",
                        ground_truth=sample.ground_truth,
                        predicted_class="no_face_detected",
                        manipulated_probability=None,
                        authentic_probability=None,
                        correct=False,
                        face_count=0,
                        inference_status="no_face_detected",
                        error_message="No facial structures detected in sampled video frames.",
                        inference_time_ms=inference_time_ms,
                        resolution=sample.resolution,
                        source=sample.source,
                        split=sample.split,
                        manipulation_type=sample.manipulation_type,
                        sha256=file_sha
                    )

                is_fake = (det_res.prediction == "fake")
                manip_prob = det_res.probability if is_fake else round(1.0 - (det_res.probability or 0.0), 4)
                auth_prob = round(1.0 - manip_prob, 4)
                pred_class = "manipulated" if manip_prob >= self.threshold else "authentic"
                correct = (pred_class == sample.ground_truth.lower().strip())

                return SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    filename=target_path.name,
                    file_path=str(target_path),
                    media_type="video",
                    ground_truth=sample.ground_truth,
                    predicted_class=pred_class,
                    manipulated_probability=manip_prob,
                    authentic_probability=auth_prob,
                    correct=correct,
                    face_count=det_res.faces_detected or 1,
                    inference_status="completed",
                    inference_time_ms=inference_time_ms,
                    resolution=sample.resolution,
                    source=sample.source,
                    split=sample.split,
                    manipulation_type=sample.manipulation_type,
                    sha256=file_sha
                )

            else:
                # Still image evaluation using production detector
                det_info = detect_and_crop_faces(target_path)
                face_count = det_info["count"]

                # Extract image dimensions for resolution tracking
                h, w = 0, 0
                if det_info.get("original_image") is not None:
                    h, w = det_info["original_image"].shape[:2]
                res_str = sample.resolution if (sample.resolution and sample.resolution != "unknown") else (f"{w}x{h}" if w > 0 else "unknown")

                inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

                if face_count == 0:
                    return SampleEvaluationResult(
                        sample_id=sample.sample_id,
                        filename=target_path.name,
                        file_path=str(target_path),
                        media_type="image",
                        ground_truth=sample.ground_truth,
                        predicted_class="no_face_detected",
                        manipulated_probability=None,
                        authentic_probability=None,
                        correct=False,
                        face_count=0,
                        inference_status="no_face_detected",
                        error_message="No facial structures detected by YuNet.",
                        inference_time_ms=inference_time_ms,
                        resolution=res_str,
                        source=sample.source,
                        split=sample.split,
                        manipulation_type=sample.manipulation_type,
                        sha256=file_sha
                    )

                # Predict each face crop
                crop_preds = [predict_face_crop(c, threshold=self.threshold) for c in det_info["crops"]]
                fake_probs = [p["fake_probability"] for p in crop_preds]
                real_probs = [p["real_probability"] for p in crop_preds]

                # Multi-face policy matching production:
                # If any face is fake, max(fake_probs). Otherwise average real prob.
                if any(p["prediction"] == "fake" for p in crop_preds):
                    manip_prob = round(max(fake_probs), 4)
                else:
                    avg_real = sum(real_probs) / len(real_probs) if real_probs else 0.5
                    manip_prob = round(1.0 - avg_real, 4)

                auth_prob = round(1.0 - manip_prob, 4)
                pred_class = "manipulated" if manip_prob >= self.threshold else "authentic"
                correct = (pred_class == sample.ground_truth.lower().strip())

                return SampleEvaluationResult(
                    sample_id=sample.sample_id,
                    filename=target_path.name,
                    file_path=str(target_path),
                    media_type="image",
                    ground_truth=sample.ground_truth,
                    predicted_class=pred_class,
                    manipulated_probability=manip_prob,
                    authentic_probability=auth_prob,
                    correct=correct,
                    face_count=face_count,
                    inference_status="completed",
                    inference_time_ms=inference_time_ms,
                    resolution=res_str,
                    source=sample.source,
                    split=sample.split,
                    manipulation_type=sample.manipulation_type,
                    sha256=file_sha
                )

        except Exception as e:
            logger.error(f"Inference error on benchmark sample '{sample.sample_id}': {e}", exc_info=True)
            return SampleEvaluationResult(
                sample_id=sample.sample_id,
                filename=target_path.name,
                file_path=str(target_path),
                media_type=sample.media_type,
                ground_truth=sample.ground_truth,
                inference_status="error",
                error_message=str(e),
                inference_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
                resolution=sample.resolution,
                source=sample.source,
                split=sample.split,
                manipulation_type=sample.manipulation_type,
                sha256=file_sha
            )

    def run_benchmark(
        self,
        manifest: DatasetManifest,
        max_samples: Optional[int] = None
    ) -> BenchmarkRunDetail:
        """
        Executes the full reproducible evaluation benchmark across manifest samples.
        """
        run_start = time.perf_counter()
        run_id = f"BENCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # 1. Manifest Validation
        is_valid, val_warnings, stats = validate_manifest(manifest)
        if not is_valid:
            raise ValueError(f"Manifest validation failed: {'; '.join(val_warnings)}")

        # Enforce maximum benchmark sample cap
        effective_max = min(max_samples, MAX_BENCHMARK_SAMPLES) if max_samples else MAX_BENCHMARK_SAMPLES
        samples_to_eval = manifest.samples[:effective_max]
        logger.info(f"[{run_id}] Commencing benchmark evaluation on '{manifest.dataset_name}' ({len(samples_to_eval)} samples, cap={effective_max}).")

        evaluated_samples: List[SampleEvaluationResult] = []
        for idx, s in enumerate(samples_to_eval):
            res = self.evaluate_sample(s)
            evaluated_samples.append(res)
            if (idx + 1) % 25 == 0 or (idx + 1) == len(samples_to_eval):
                logger.info(f"[{run_id}] Evaluated {idx + 1}/{len(samples_to_eval)} samples...")

        total_runtime = round(time.perf_counter() - run_start, 2)
        avg_infer = (
            round(sum(s.inference_time_ms for s in evaluated_samples) / len(evaluated_samples), 2)
            if evaluated_samples else 0.0
        )

        # 2. Metric calculations at selected benchmark threshold
        metrics, conf_matrix = compute_metrics_at_threshold(evaluated_samples, self.threshold)

        # 3. Threshold sweep analysis
        threshold_sweep = run_threshold_sweep(evaluated_samples)

        # 4. Error analysis
        errors = extract_error_analysis(evaluated_samples, self.threshold)

        # 5. Subgroup condition analysis
        subgroups = compute_subgroup_analysis(evaluated_samples, self.threshold)

        # Sample distribution breakdown
        authentic_count = sum(1 for s in evaluated_samples if s.ground_truth.lower().strip() == "authentic")
        manipulated_count = sum(1 for s in evaluated_samples if s.ground_truth.lower().strip() == "manipulated")
        no_face_count = sum(1 for s in evaluated_samples if s.inference_status == "no_face_detected")
        error_count = sum(1 for s in evaluated_samples if s.inference_status == "error")
        img_count = sum(1 for s in evaluated_samples if s.media_type.lower().strip() == "image")
        vid_count = sum(1 for s in evaluated_samples if s.media_type.lower().strip() == "video")

        run_detail = BenchmarkRunDetail(
            benchmark_run_id=run_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name="Xception",
            framework="DeepfakeBench",
            checkpoint="xception_best.pth",
            threshold=self.threshold,
            dataset_name=manifest.dataset_name,
            sample_count=len(evaluated_samples),
            image_count=img_count,
            video_count=vid_count,
            authentic_count=authentic_count,
            manipulated_count=manipulated_count,
            no_face_count=no_face_count,
            error_count=error_count,
            metrics=metrics,
            confusion_matrix=conf_matrix,
            total_runtime_seconds=total_runtime,
            avg_inference_time_ms=avg_infer,
            warnings=val_warnings,
            samples=evaluated_samples,
            threshold_sweep=threshold_sweep,
            errors=errors,
            subgroup_metrics=subgroups
        )

        # 6. Save persistent JSON artifact
        save_path = BENCHMARK_RESULTS_DIR / f"benchmark_{run_id}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(run_detail.model_dump_json(indent=2))

        logger.info(
            f"[{run_id}] Benchmark completed in {total_runtime}s: Accuracy={metrics.accuracy:.4f}, "
            f"F1={metrics.f1_score:.4f}, Saved to {save_path.name}"
        )
        return run_detail


def list_benchmark_runs() -> List[BenchmarkRunSummary]:
    """Retrieves summaries of all saved benchmark runs in reverse chronological order."""
    summaries: List[BenchmarkRunSummary] = []
    for file in sorted(BENCHMARK_RESULTS_DIR.glob("benchmark_BENCH-*.json"), reverse=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = BenchmarkRunSummary(**data)
                summaries.append(summary)
        except Exception as e:
            logger.warning(f"Failed to read benchmark run file {file.name}: {e}")
    return summaries


def get_benchmark_run(run_id: str) -> Optional[BenchmarkRunDetail]:
    """Retrieves full details for a specific benchmark run."""
    clean_id = run_id if run_id.startswith("BENCH-") else f"BENCH-{run_id}"
    file_path = BENCHMARK_RESULTS_DIR / f"benchmark_{clean_id}.json"
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return BenchmarkRunDetail(**data)
    except Exception as e:
        logger.error(f"Error parsing benchmark detail {file_path.name}: {e}")
        return None


def get_sample_manifest_template() -> Dict[str, Any]:
    """Returns a valid JSON manifest template clearly marked as a template."""
    return {
        "manifest_version": "1.0",
        "dataset_name": "TEMPLATE_EVALUATION_DATASET",
        "dataset_version": "1.0",
        "description": "Template manifest specification for DEEPTRACE supervised detector evaluation.",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "samples": [
            {
                "sample_id": "sample_0001",
                "file_path": "datasets/authentic_01.jpg",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "media_type": "image",
                "ground_truth": "authentic",
                "source": "CelebA-HQ",
                "dataset_name": "CelebA-HQ-Test",
                "split": "test",
                "manipulation_type": "none",
                "resolution": "1024x1024",
                "compression_level": "raw",
                "notes": "Reference authentic facial subject."
            },
            {
                "sample_id": "sample_0002",
                "file_path": "datasets/manipulated_01.jpg",
                "sha256": "cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34",
                "media_type": "image",
                "ground_truth": "manipulated",
                "source": "FaceForensics++",
                "dataset_name": "FF++-Deepfakes",
                "split": "test",
                "manipulation_type": "face_swap",
                "resolution": "1024x1024",
                "compression_level": "c23",
                "notes": "Face-swap manipulation."
            }
        ]
    }
