"""
FastAPI router for Multi-Model Cross-Validation and Comparison.
Provides image cross-validation, video frame comparison, and registry endpoints.
Hardened for upload validation, path confinement, and resource protection.
"""

import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status, Body
from pydantic import BaseModel

from ..services.cross_validation_service import get_cross_validation_service, compute_agreement_statistics
from ..services.model_registry import get_model_registry
from ..services.benchmark_service import resolve_sample_path, BenchmarkRunner
from ..models.cross_validation_schemas import (
    ModelRegistryEntry,
    ImageCrossValidationResponse,
    VideoCrossValidationResponse,
    MultiModelBenchmarkComparison,
    ComparativeModelMetricItem
)
from ..models.benchmark_schemas import DatasetManifest
from ..config import (
    UPLOADS_DIR,
    DEEPFAKE_THRESHOLD,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_BENCHMARK_SAMPLES
)
from ..utils.logger import logger
from ..utils.file_utils import (
    sanitize_filename,
    validate_file_content,
    sanitize_error_message
)

router = APIRouter(prefix="/cross-validation", tags=["Multi-Model Cross Validation"])

ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class CrossValidationPathRequest(BaseModel):
    file_path: Optional[str] = None
    case_id: Optional[str] = None
    comparison_models: Optional[List[str]] = None
    thresholds: Optional[Dict[str, float]] = None
    sample_interval: Optional[int] = 15
    max_frames: Optional[int] = 30


class MultiModelBenchmarkRequest(BaseModel):
    manifest_path: Optional[str] = None
    manifest_content: Optional[DatasetManifest] = None
    models: Optional[List[str]] = None
    thresholds: Optional[Dict[str, float]] = None
    max_samples: Optional[int] = None


@router.get("/models", response_model=List[ModelRegistryEntry], status_code=status.HTTP_200_OK)
async def get_registered_models():
    """Returns metadata and live operational status for all registered deepfake detectors."""
    registry = get_model_registry()
    return registry.list_models()


@router.post("/image", response_model=ImageCrossValidationResponse, status_code=status.HTTP_200_OK)
async def cross_validate_image_endpoint(payload: CrossValidationPathRequest):
    """
    Executes cross-validation on an image referenced by file_path or case_id.
    """
    cv_service = get_cross_validation_service()

    if payload.file_path:
        try:
            target_path = resolve_sample_path(payload.file_path)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to file outside approved directory is forbidden."
            )
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Requested file not found.")
    elif payload.case_id:
        if not ID_REGEX.match(payload.case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format.")
        matches = list(UPLOADS_DIR.glob(f"{payload.case_id}_*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Case ID not found in storage: {payload.case_id}")
        target_path = matches[0]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'case_id' must be provided."
        )

    return cv_service.cross_validate_image(
        image_path=target_path,
        comparison_model_ids=payload.comparison_models,
        thresholds=payload.thresholds
    )


@router.post("/image-upload", response_model=ImageCrossValidationResponse, status_code=status.HTTP_200_OK)
async def cross_validate_image_upload_endpoint(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None)
):
    """
    Executes cross-validation on a newly uploaded image file with security checks.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format .{ext}. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
        )

    header = await file.read(64)
    is_valid, err_msg = validate_file_content(safe_name, header)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    cv_service = get_cross_validation_service()
    temp_path = UPLOADS_DIR / f"cv_temp_{int(time.time())}_{safe_name}"

    total_bytes = len(header)
    try:
        with open(temp_path, "wb") as f_out:
            f_out.write(header)
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_IMAGE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Image exceeds maximum allowed size of {MAX_IMAGE_SIZE_BYTES / (1024 * 1024):.0f} MB."
                    )
                f_out.write(chunk)

        thresh_dict = {"xception_deepfakebench": threshold} if threshold is not None else None
        return cv_service.cross_validate_image(image_path=temp_path, thresholds=thresh_dict)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post("/video", response_model=VideoCrossValidationResponse, status_code=status.HTTP_200_OK)
async def cross_validate_video_endpoint(payload: CrossValidationPathRequest):
    """
    Executes video cross-validation on a video referenced by file_path or case_id.
    CRITICAL: Evaluates ALL models on the EXACT SAME sampled frames and detected face crops.
    """
    cv_service = get_cross_validation_service()

    if payload.file_path:
        try:
            target_path = resolve_sample_path(payload.file_path)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to file outside approved directory is forbidden."
            )
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Requested file not found.")
    elif payload.case_id:
        if not ID_REGEX.match(payload.case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format.")
        matches = list(UPLOADS_DIR.glob(f"{payload.case_id}_*"))
        if not matches:
            raise HTTPException(status_code=404, detail=f"Case ID not found in storage: {payload.case_id}")
        target_path = matches[0]
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'file_path' or 'case_id' must be provided."
        )

    return cv_service.cross_validate_video(
        video_path=target_path,
        sample_interval=payload.sample_interval or 15,
        max_frames=payload.max_frames or 30,
        comparison_model_ids=payload.comparison_models,
        thresholds=payload.thresholds
    )


@router.post("/video-upload", response_model=VideoCrossValidationResponse, status_code=status.HTTP_200_OK)
async def cross_validate_video_upload_endpoint(
    file: UploadFile = File(...),
    sample_interval: int = Form(15),
    max_frames: int = Form(30)
):
    """
    Executes video cross-validation on a newly uploaded video file with security checks.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower().lstrip(".")
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported video format .{ext}. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    header = await file.read(64)
    is_valid, err_msg = validate_file_content(safe_name, header)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)

    cv_service = get_cross_validation_service()
    temp_path = UPLOADS_DIR / f"cv_temp_{int(time.time())}_{safe_name}"

    total_bytes = len(header)
    try:
        with open(temp_path, "wb") as f_out:
            f_out.write(header)
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video exceeds maximum allowed size of {MAX_VIDEO_SIZE_BYTES / (1024 * 1024):.0f} MB."
                    )
                f_out.write(chunk)

        return cv_service.cross_validate_video(
            video_path=temp_path,
            sample_interval=sample_interval,
            max_frames=max_frames
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@router.post("/benchmark-comparison", response_model=MultiModelBenchmarkComparison, status_code=status.HTTP_200_OK)
async def compare_benchmark_models(request: MultiModelBenchmarkRequest):
    """
    Evaluates multiple detectors on the EXACT SAME labeled evaluation dataset.
    Computes comparative metrics, agreement rate, disagreement rate, correlation, and Cohen's Kappa.
    Does NOT create an ensemble, ranking, or claim a 'winner'.
    """
    registry = get_model_registry()
    manifest: Optional[DatasetManifest] = None

    if request.manifest_content:
        manifest = request.manifest_content
    elif request.manifest_path:
        try:
            target_path = resolve_sample_path(request.manifest_path)
        except PermissionError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access to manifest outside approved directory is forbidden."
            )
        if not target_path.exists():
            raise HTTPException(status_code=404, detail=f"Manifest not found: '{request.manifest_path}'")
        import json
        with open(target_path, "r", encoding="utf-8") as f:
            manifest = DatasetManifest(**json.load(f))
    else:
        raise HTTPException(status_code=400, detail="Either 'manifest_path' or 'manifest_content' required.")

    # Determine models to evaluate
    model_ids = request.models or ["xception_deepfakebench", "meso4_deepfakebench"]
    thresholds = request.thresholds or {}

    effective_max = min(request.max_samples, MAX_BENCHMARK_SAMPLES) if request.max_samples else MAX_BENCHMARK_SAMPLES
    samples_to_eval = manifest.samples[:effective_max]
    if not samples_to_eval:
        raise HTTPException(status_code=400, detail="Manifest contains zero samples.")

    # Evaluate each model on IDENTICAL samples
    from ..services.face_service import detect_and_crop_faces
    from ..services.benchmark_service import compute_metrics_at_threshold
    from ..models.benchmark_schemas import SampleEvaluationResult

    model_eval_results: Dict[str, List[SampleEvaluationResult]] = {mid: [] for mid in model_ids}

    for s in samples_to_eval:
        try:
            target_file = resolve_sample_path(s.file_path)
        except PermissionError:
            continue
        if not target_file.exists():
            continue

        # Detect face once per sample
        det_info = detect_and_crop_faces(target_file)
        has_face = det_info.get("count", 0) > 0 and len(det_info.get("crops", [])) > 0
        crop_bgr = det_info["crops"][0] if has_face else None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if (has_face and crop_bgr is not None) else None

        for mid in model_ids:
            det = registry.get_detector(mid)
            if det is None:
                continue

            t = thresholds.get(mid, det.default_threshold)
            if not has_face or crop_rgb is None:
                model_eval_results[mid].append(
                    SampleEvaluationResult(
                        sample_id=s.sample_id,
                        filename=target_file.name,
                        file_path=s.file_path,
                        media_type=s.media_type,
                        ground_truth=s.ground_truth,
                        predicted_class="no_face_detected",
                        manipulated_probability=None,
                        authentic_probability=None,
                        correct=False,
                        face_count=0,
                        inference_status="no_face_detected",
                        error_message="No facial structures detected."
                    )
                )
            else:
                pred = det.predict_face_crop(crop_rgb, threshold=t)
                is_correct = (pred.prediction == s.ground_truth.lower().strip())
                model_eval_results[mid].append(
                    SampleEvaluationResult(
                        sample_id=s.sample_id,
                        filename=target_file.name,
                        file_path=s.file_path,
                        media_type=s.media_type,
                        ground_truth=s.ground_truth,
                        predicted_class=pred.prediction,
                        manipulated_probability=pred.manipulated_probability,
                        authentic_probability=pred.authentic_probability,
                        correct=is_correct,
                        face_count=1,
                        inference_status="completed" if pred.status == "success" else "error",
                        error_message=pred.error_message,
                        inference_time_ms=pred.inference_time_ms
                    )
                )

    # Compute comparative metrics per model
    evaluated_metric_items: List[ComparativeModelMetricItem] = []
    for mid in model_ids:
        det = registry.get_detector(mid)
        if det is None:
            continue
        t = thresholds.get(mid, det.default_threshold)
        samples = model_eval_results[mid]
        metrics, _ = compute_metrics_at_threshold(samples, t)

        evaluated_metric_items.append(
            ComparativeModelMetricItem(
                model_id=mid,
                display_name=det.display_name,
                role=det.role,
                threshold=t,
                sample_count=len(samples),
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                specificity=metrics.specificity,
                f1_score=metrics.f1_score,
                false_positive_rate=metrics.false_positive_rate,
                false_negative_rate=metrics.false_negative_rate,
                roc_auc=metrics.roc_auc,
                pr_auc=metrics.pr_auc
            )
        )

    # Inter-model agreement statistics between primary and comparison
    p_id = "xception_deepfakebench"
    c_id = "meso4_deepfakebench"
    if p_id in model_eval_results and c_id in model_eval_results:
        p_samples = model_eval_results[p_id]
        c_samples = model_eval_results[c_id]
        common_len = min(len(p_samples), len(c_samples))
        p_preds = [p_samples[i].predicted_class or "unknown" for i in range(common_len)]
        c_preds = [c_samples[i].predicted_class or "unknown" for i in range(common_len)]
        p_scores = [p_samples[i].manipulated_probability or 0.0 for i in range(common_len)]
        c_scores = [c_samples[i].manipulated_probability or 0.0 for i in range(common_len)]
        agreement_stats = compute_agreement_statistics(p_preds, c_preds, p_scores, c_scores)
    else:
        agreement_stats = compute_agreement_statistics([], [], [], [])

    return MultiModelBenchmarkComparison(
        dataset_name=manifest.dataset_name,
        sample_count=len(samples_to_eval),
        evaluated_models=evaluated_metric_items,
        agreement_statistics=agreement_stats,
        identical_dataset_verified=True
    )
