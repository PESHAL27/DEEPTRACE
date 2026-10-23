"""
Cross-Validation Service for DEEPTRACE Multi-Model Verification.
Provides image and video cross-validation across independent deepfake detectors,
frame-aligned video evaluation, and statistical agreement/correlation calculations.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from scipy import stats

from .model_registry import get_model_registry
from .face_service import detect_and_crop_faces
from ..models.cross_validation_schemas import (
    NormalizedDetectorPrediction,
    ImageCrossValidationResponse,
    VideoFrameComparisonItem,
    VideoCrossValidationResponse,
    ModelAgreementStatistics,
    ComparativeModelMetricItem,
    MultiModelBenchmarkComparison
)
from ..config import (
    VIDEO_SAMPLE_INTERVAL,
    MAX_ANALYZED_FRAMES,
    DEEPFAKE_THRESHOLD
)
from ..utils.logger import logger


def calculate_cohen_kappa(preds_a: List[str], preds_b: List[str]) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Calculates Cohen's Kappa coefficient (inter-rater agreement) between two categorical predictions.
    Categories: 'authentic' and 'manipulated'.
    Returns: (kappa_score, status, explanation)
    """
    if len(preds_a) != len(preds_b) or len(preds_a) == 0:
        return None, "not_available", "Insufficient sample count for Cohen's Kappa."

    n = len(preds_a)
    classes = ["authentic", "manipulated"]
    
    # 2x2 contingency table
    # matrix[i][j] = count where preds_a has classes[i] and preds_b has classes[j]
    matrix = np.zeros((2, 2), dtype=int)
    for a, b in zip(preds_a, preds_b):
        idx_a = 1 if a.lower() == "manipulated" else 0
        idx_b = 1 if b.lower() == "manipulated" else 0
        matrix[idx_a][idx_b] += 1

    # Observed agreement po
    po = (matrix[0][0] + matrix[1][1]) / n

    # Expected chance agreement pe
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = (row_sums[0] * col_sums[0]) + (row_sums[1] * col_sums[1])

    if math.isclose(pe, 1.0, abs_tol=1e-6):
        return 1.0, "available", None

    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4), "available", None


def calculate_correlation(
    scores_a: List[float],
    scores_b: List[float]
) -> Tuple[Optional[float], Optional[float], str, Optional[str]]:
    """
    Calculates Pearson linear correlation and Spearman rank correlation between two probability series.
    Returns: (pearson_r, spearman_rho, status, explanation)
    """
    valid_pairs = [
        (a, b) for a, b in zip(scores_a, scores_b)
        if a is not None and b is not None and not np.isnan(a) and not np.isnan(b)
    ]
    if len(valid_pairs) < 2:
        return None, None, "not_available", "At least 2 valid sample pairs required to calculate correlation."

    vals_a = [p[0] for p in valid_pairs]
    vals_b = [p[1] for p in valid_pairs]

    # Check for zero variance
    if np.std(vals_a) == 0 or np.std(vals_b) == 0:
        return None, None, "not_available", "One or both score series has zero variance (constant values)."

    try:
        pearson_r, _ = stats.pearsonr(vals_a, vals_b)
        spearman_rho, _ = stats.spearmanr(vals_a, vals_b)
        return round(float(pearson_r), 4), round(float(spearman_rho), 4), "available", None
    except Exception as e:
        return None, None, "not_available", f"Correlation calculation error: {e}"


def compute_agreement_statistics(
    primary_predictions: List[str],
    comparison_predictions: List[str],
    primary_scores: List[float],
    comparison_scores: List[float]
) -> ModelAgreementStatistics:
    """Computes comprehensive agreement, disagreement, correlation, and kappa statistics."""
    total = len(primary_predictions)
    if total == 0:
        return ModelAgreementStatistics(
            total_compared=0,
            agreeing_count=0,
            disagreeing_count=0,
            agreement_rate=1.0,
            disagreement_rate=0.0,
            cohen_kappa=None,
            cohen_kappa_status="not_available",
            pearson_correlation=None,
            spearman_correlation=None,
            correlation_status="not_available",
            correlation_explanation="No samples to compare."
        )

    agreeing = sum(1 for a, b in zip(primary_predictions, comparison_predictions) if a.lower() == b.lower())
    disagreeing = total - agreeing
    agr_rate = round(agreeing / total, 4)
    disagr_rate = round(disagreeing / total, 4)

    kappa, kappa_status, _ = calculate_cohen_kappa(primary_predictions, comparison_predictions)
    pearson_r, spearman_rho, corr_status, corr_exp = calculate_correlation(primary_scores, comparison_scores)

    return ModelAgreementStatistics(
        total_compared=total,
        agreeing_count=agreeing,
        disagreeing_count=disagreeing,
        agreement_rate=agr_rate,
        disagreement_rate=disagr_rate,
        cohen_kappa=kappa,
        cohen_kappa_status=kappa_status,
        pearson_correlation=pearson_r,
        spearman_correlation=spearman_rho,
        correlation_status=corr_status,
        correlation_explanation=corr_exp
    )


class CrossValidationService:
    """Service executing multi-model cross-validation on images and videos."""

    def __init__(self):
        self.registry = get_model_registry()

    def cross_validate_image(
        self,
        image_path: Path,
        comparison_model_ids: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> ImageCrossValidationResponse:
        """
        Executes multi-model cross-validation on a single image.
        1. Runs YuNet face detection once.
        2. Evaluates Primary detector (Xception).
        3. Evaluates selected Comparison detector(s) (Meso4) on identical face crop.
        4. Detects agreement vs disagreement.
        """
        thresholds = thresholds or {}
        primary_det = self.registry.get_primary_detector()

        # Resolve comparison detectors
        if comparison_model_ids is None:
            comparison_detectors = self.registry.get_comparison_detectors()
        else:
            comparison_detectors = [
                self.registry.get_detector(mid) for mid in comparison_model_ids
                if self.registry.get_detector(mid) is not None and self.registry.get_detector(mid).role == "comparison"
            ]

        # 1. Face detection
        det_info = detect_and_crop_faces(image_path)
        faces_count = det_info.get("count", 0)

        if faces_count == 0 or len(det_info.get("crops", [])) == 0:
            # Failure containment for zero-face images
            primary_res = NormalizedDetectorPrediction(
                model_id=primary_det.model_id,
                display_name=primary_det.display_name,
                role=primary_det.role,
                prediction="no_face_detected",
                manipulated_probability=None,
                authentic_probability=None,
                threshold=thresholds.get(primary_det.model_id, primary_det.default_threshold),
                status="no_face_detected",
                error_message="No face detected in the image."
            )
            comp_results = [
                NormalizedDetectorPrediction(
                    model_id=cd.model_id,
                    display_name=cd.display_name,
                    role=cd.role,
                    prediction="no_face_detected",
                    manipulated_probability=None,
                    authentic_probability=None,
                    threshold=thresholds.get(cd.model_id, cd.default_threshold),
                    status="no_face_detected",
                    error_message="No face detected in the image."
                )
                for cd in comparison_detectors
            ]
            return ImageCrossValidationResponse(
                filename=image_path.name,
                faces_detected=0,
                primary_model=primary_res,
                comparison_models=comp_results,
                agreement=True,
                disagreement=False,
                agreement_status="no_face_detected",
                summary_message="No facial structures detected. Cross-validation bypassed."
            )

        # 2. Extract largest detected face crop and convert to RGB (adhering to BaseDetector contract)
        primary_crop = det_info["crops"][0]
        primary_crop_rgb = cv2.cvtColor(primary_crop, cv2.COLOR_BGR2RGB)

        # 3. Evaluate Primary Model (Xception)
        primary_thresh = thresholds.get(primary_det.model_id, primary_det.default_threshold)
        primary_pred = primary_det.predict_face_crop(primary_crop_rgb, threshold=primary_thresh)

        # 4. Evaluate Comparison Models (Meso4) on IDENTICAL face crop
        comp_predictions: List[NormalizedDetectorPrediction] = []
        for cd in comparison_detectors:
            cd_thresh = thresholds.get(cd.model_id, cd.default_threshold)
            try:
                res = cd.predict_face_crop(primary_crop_rgb, threshold=cd_thresh)
            except Exception as e:
                logger.error(f"Comparison detector '{cd.model_id}' failed: {e}")
                res = NormalizedDetectorPrediction(
                    model_id=cd.model_id,
                    display_name=cd.display_name,
                    role=cd.role,
                    prediction=None,
                    manipulated_probability=None,
                    authentic_probability=None,
                    threshold=cd_thresh,
                    inference_time_ms=0.0,
                    status="error",
                    error_message=f"Model execution failed: {str(e)}"
                )
            comp_predictions.append(res)

        # 5. Determine Agreement / Disagreement
        active_comps = [c for c in comp_predictions if c.status == "success" and c.prediction is not None]

        if not active_comps:
            return ImageCrossValidationResponse(
                filename=image_path.name,
                faces_detected=faces_count,
                primary_model=primary_pred,
                comparison_models=comp_predictions,
                agreement=True,
                disagreement=False,
                agreement_status="comparison_unavailable",
                summary_message=f"Primary model completed ({primary_pred.prediction}). Comparison detector unavailable."
            )

        has_disagreement = any(
            c.prediction.lower() != primary_pred.prediction.lower()
            for c in active_comps if primary_pred.prediction is not None
        )

        if has_disagreement:
            disagreeing_names = [
                f"{c.display_name} ({c.prediction}, prob: {c.manipulated_probability})"
                for c in active_comps if c.prediction.lower() != primary_pred.prediction.lower()
            ]
            analysis = (
                f"Detector disagreement detected. Primary {primary_pred.display_name} predicted '{primary_pred.prediction}' "
                f"(probability: {primary_pred.manipulated_probability}), whereas {', '.join(disagreeing_names)}. "
                "Independent forensic inspection recommended."
            )
            return ImageCrossValidationResponse(
                filename=image_path.name,
                faces_detected=faces_count,
                primary_model=primary_pred,
                comparison_models=comp_predictions,
                agreement=False,
                disagreement=True,
                agreement_status="disagreement",
                summary_message="Detector disagreement detected.",
                disagreement_analysis=analysis
            )
        else:
            return ImageCrossValidationResponse(
                filename=image_path.name,
                faces_detected=faces_count,
                primary_model=primary_pred,
                comparison_models=comp_predictions,
                agreement=True,
                disagreement=False,
                agreement_status="agreement",
                summary_message=f"Model agreement: all detectors consistently classify the facial media as '{primary_pred.prediction}'."
            )

    def cross_validate_video(
        self,
        video_path: Path,
        sample_interval: int = VIDEO_SAMPLE_INTERVAL,
        max_frames: int = MAX_ANALYZED_FRAMES,
        comparison_model_ids: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None
    ) -> VideoCrossValidationResponse:
        """
        Executes multi-model cross-validation on a video.
        CRITICAL RULE: Evaluates ALL models on the EXACT SAME sampled frames and facial crops.
        """
        thresholds = thresholds or {}
        primary_det = self.registry.get_primary_detector()

        if comparison_model_ids is None:
            comparison_detectors = self.registry.get_comparison_detectors()
        else:
            comparison_detectors = [
                self.registry.get_detector(mid) for mid in comparison_model_ids
                if self.registry.get_detector(mid) is not None and self.registry.get_detector(mid).role == "comparison"
            ]

        # Open video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path.name}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = round(total_frames / fps, 2) if fps > 0 else 0.0

        # Uniform frame index sampling
        frame_indices = list(range(0, total_frames, max(1, sample_interval)))
        if max_frames and len(frame_indices) > max_frames:
            step = len(frame_indices) / max_frames
            frame_indices = [frame_indices[int(i * step)] for i in range(max_frames)]

        frame_items: List[VideoFrameComparisonItem] = []
        primary_manip_frames = 0
        comp_manip_frames_dict = {cd.model_id: 0 for cd in comparison_detectors}

        current_frame = 0
        idx_pointer = 0

        while cap.isOpened() and idx_pointer < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            target_frame_num = frame_indices[idx_pointer]
            if current_frame == target_frame_num:
                timestamp = round(current_frame / fps, 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection on the exact frame
                det_info = detect_and_crop_faces(frame_rgb)
                has_face = det_info.get("count", 0) > 0 and len(det_info.get("crops", [])) > 0

                if has_face:
                    face_crop = det_info["crops"][0]
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                    # 1. Primary prediction on frame crop (RGB)
                    p_thresh = thresholds.get(primary_det.model_id, primary_det.default_threshold)
                    p_pred = primary_det.predict_face_crop(face_crop_rgb, threshold=p_thresh)
                    if p_pred.prediction == "manipulated":
                        primary_manip_frames += 1

                    # 2. Comparison predictions on IDENTICAL frame crop (RGB)
                    c_preds = []
                    frame_agree = True
                    for cd in comparison_detectors:
                        c_thresh = thresholds.get(cd.model_id, cd.default_threshold)
                        try:
                            c_res = cd.predict_face_crop(face_crop_rgb, threshold=c_thresh)
                        except Exception as e:
                            logger.error(f"Comparison detector '{cd.model_id}' failed on video frame: {e}")
                            c_res = NormalizedDetectorPrediction(
                                model_id=cd.model_id,
                                display_name=cd.display_name,
                                role=cd.role,
                                prediction=None,
                                manipulated_probability=None,
                                authentic_probability=None,
                                threshold=c_thresh,
                                inference_time_ms=0.0,
                                status="error",
                                error_message=f"Model execution failed: {str(e)}"
                            )
                        c_preds.append(c_res)
                        if c_res.prediction == "manipulated":
                            comp_manip_frames_dict[cd.model_id] += 1

                        if c_res.status == "success" and p_pred.status == "success":
                            if c_res.prediction != p_pred.prediction:
                                frame_agree = False

                    frame_items.append(
                        VideoFrameComparisonItem(
                            frame_number=current_frame,
                            timestamp=timestamp,
                            face_detected=True,
                            primary_model=p_pred,
                            comparison_models=c_preds,
                            agreement=frame_agree,
                            disagreement=not frame_agree
                        )
                    )
                else:
                    # No face in frame
                    p_pred = NormalizedDetectorPrediction(
                        model_id=primary_det.model_id,
                        display_name=primary_det.display_name,
                        role=primary_det.role,
                        prediction="no_face_detected",
                        threshold=thresholds.get(primary_det.model_id, primary_det.default_threshold),
                        status="no_face_detected"
                    )
                    c_preds = [
                        NormalizedDetectorPrediction(
                            model_id=cd.model_id,
                            display_name=cd.display_name,
                            role=cd.role,
                            prediction="no_face_detected",
                            threshold=thresholds.get(cd.model_id, cd.default_threshold),
                            status="no_face_detected"
                        )
                        for cd in comparison_detectors
                    ]
                    frame_items.append(
                        VideoFrameComparisonItem(
                            frame_number=current_frame,
                            timestamp=timestamp,
                            face_detected=False,
                            primary_model=p_pred,
                            comparison_models=c_preds,
                            agreement=True,
                            disagreement=False
                        )
                    )

                idx_pointer += 1

            current_frame += 1

        cap.release()

        # Agreement metrics on facial frames
        face_frames = [f for f in frame_items if f.face_detected]
        compared_count = len(face_frames)
        agreeing_count = sum(1 for f in face_frames if f.agreement)
        disagreeing_count = compared_count - agreeing_count
        agreement_pct = round((agreeing_count / compared_count) * 100, 2) if compared_count > 0 else 100.0

        # Primary summary
        p_thresh = thresholds.get(primary_det.model_id, primary_det.default_threshold)
        p_manip_probs = [
            f.primary_model.manipulated_probability
            for f in face_frames if f.primary_model.manipulated_probability is not None
        ]
        # Video aggregation: maximum frame probability
        p_agg_prob = max(p_manip_probs) if p_manip_probs else 0.0
        p_verdict = "manipulated" if p_agg_prob >= p_thresh else "authentic"
        p_summary = NormalizedDetectorPrediction(
            model_id=primary_det.model_id,
            display_name=primary_det.display_name,
            role=primary_det.role,
            prediction=p_verdict,
            manipulated_probability=round(p_agg_prob, 4),
            authentic_probability=round(1.0 - p_agg_prob, 4),
            threshold=p_thresh,
            status="success"
        )

        # Comparison summaries
        comp_summaries = []
        for cd in comparison_detectors:
            cd_thresh = thresholds.get(cd.model_id, cd.default_threshold)
            c_probs = [
                comp.manipulated_probability
                for f in face_frames
                for comp in f.comparison_models
                if comp.model_id == cd.model_id and comp.manipulated_probability is not None
            ]
            c_agg = max(c_probs) if c_probs else 0.0
            c_verdict = "manipulated" if c_agg >= cd_thresh else "authentic"
            comp_summaries.append(
                NormalizedDetectorPrediction(
                    model_id=cd.model_id,
                    display_name=cd.display_name,
                    role=cd.role,
                    prediction=c_verdict,
                    manipulated_probability=round(c_agg, 4),
                    authentic_probability=round(1.0 - c_agg, 4),
                    threshold=cd_thresh,
                    status="success" if c_probs else "unavailable"
                )
            )

        video_agreement = all(c.prediction == p_summary.prediction for c in comp_summaries if c.status == "success")

        total_comp_manip_frames = sum(comp_manip_frames_dict.values())

        return VideoCrossValidationResponse(
            filename=video_path.name,
            duration=duration,
            fps=fps,
            frames_analyzed=len(frame_items),
            compared_frames=compared_count,
            agreeing_frames=agreeing_count,
            disagreeing_frames=disagreeing_count,
            agreement_percentage=agreement_pct,
            primary_manipulated_frames=primary_manip_frames,
            comparison_manipulated_frames=total_comp_manip_frames,
            primary_model_summary=p_summary,
            comparison_model_summaries=comp_summaries,
            agreement=video_agreement,
            disagreement=not video_agreement,
            frames=frame_items
        )


# Singleton instance
_cross_validation_service: Optional[CrossValidationService] = None


def get_cross_validation_service() -> CrossValidationService:
    """Returns singleton CrossValidationService instance."""
    global _cross_validation_service
    if _cross_validation_service is None:
        _cross_validation_service = CrossValidationService()
    return _cross_validation_service
