import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from ..config import (
    UPLOADS_DIR,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_SIZE_BYTES,
    MAX_VIDEO_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_MIME_TYPES
)
from ..models.schemas import (
    AnalysisResponse,
    MediaMetadata,
    FaceDetectionResult,
    DetectionResult
)
from ..utils.logger import logger
from ..utils.file_utils import (
    generate_analysis_id,
    sanitize_filename,
    get_media_type_from_extension,
    validate_file_content,
    sanitize_error_message
)
from ..services.media_service import extract_media_metadata
from ..services.face_service import detect_faces
from ..services.detection_service import analyze_image, analyze_video
from ..services.metadata_service import analyze_metadata

router = APIRouter(tags=["Analysis"])

# In-memory case registry for persistent analysis lookup
analysis_store: Dict[str, AnalysisResponse] = {}

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_media(file: UploadFile = File(...)):
    """
    Ingests image or video evidence for forensic verification.
    
    1. Validates filename, extension, MIME type, and binary magic bytes.
    2. Enforces media-specific size limits (20MB image / 100MB video) in streaming chunks.
    3. Safely stores file with unique analysis ID (DT-2026-XXXXXX).
    4. Extracts concrete media telemetry using OpenCV and Pillow with resource bounds.
    5. Dispatches to face detection and deepfake classification interfaces.
    6. Ensures immediate cleanup of temporary files on analysis failure.
    """
    start_time = time.time()

    # 1. Filename presence check
    if not file.filename:
        logger.warning("Upload rejected: Missing filename.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower().lstrip(".")

    # 2. Extension verification
    if not ext or ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Upload rejected: Unsupported extension .{ext} for file {safe_name}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format .{ext}. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 3. Read initial header bytes for content validation
    header = await file.read(64)
    if not header:
        logger.warning(f"Upload rejected: Empty file {safe_name}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    is_valid, error_msg = validate_file_content(safe_name, header)
    if not is_valid:
        logger.warning(f"Upload rejected for {safe_name}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # 4. MIME type check if provided
    if file.content_type and file.content_type != "application/octet-stream":
        if file.content_type not in ALLOWED_MIME_TYPES:
            logger.warning(f"Upload rejected: Content-Type '{file.content_type}' not in allowed list.")
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"MIME type '{file.content_type}' is not supported."
            )

    # 5. Generate secure analysis ID & storage path
    analysis_id = generate_analysis_id()
    media_type = get_media_type_from_extension(ext)
    target_filename = f"{analysis_id}_{safe_name}"
    target_path = UPLOADS_DIR / target_filename

    # Media-specific size threshold
    max_size_allowed = MAX_IMAGE_SIZE_BYTES if media_type == "image" else MAX_VIDEO_SIZE_BYTES
    max_size_mb = max_size_allowed / (1024 * 1024)

    logger.info(f"[{analysis_id}] Ingesting {media_type.upper()} file: {safe_name} (max allowed: {max_size_mb:.0f}MB)")

    # 6. Stream content to disk with size constraint
    total_bytes = len(header)
    try:
        with open(target_path, "wb") as f_out:
            f_out.write(header)
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_bytes += len(chunk)
                if total_bytes > max_size_allowed:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {max_size_mb:.0f} MB for {media_type}."
                    )
                f_out.write(chunk)
    except HTTPException:
        if target_path.exists():
            target_path.unlink()
        raise
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        clean_err = sanitize_error_message(str(e))
        logger.error(f"[{analysis_id}] Failed to save file: {clean_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error saving evidence file."
        )

    # 7. Extract Media Metadata (OpenCV & Pillow) with resource bounds
    try:
        metadata = extract_media_metadata(target_path, media_type)
        logger.info(
            f"[{analysis_id}] Extracted metadata: {metadata.width}x{metadata.height}, "
            f"Size: {metadata.file_size_human}"
            + (f", FPS: {metadata.fps}, Frames: {metadata.frame_count}, Duration: {metadata.duration_formatted}" if media_type == "video" else "")
        )
    except ValueError as e:
        if target_path.exists():
            target_path.unlink()
        clean_err = sanitize_error_message(str(e))
        logger.error(f"[{analysis_id}] Invalid or corrupted media: {clean_err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=clean_err
        )
    except Exception as e:
        if target_path.exists():
            target_path.unlink()
        clean_err = sanitize_error_message(str(e))
        logger.error(f"[{analysis_id}] Unexpected media reading failure: {clean_err}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to read media telemetry from uploaded file."
        )

    # 8. Forensic Pipeline Execution (Metadata, Faces, Model Inference)
    try:
        metadata_analysis = analyze_metadata(target_path, media_type)
        if media_type == "image":
            face_result = detect_faces(target_path, media_type)
            detection_info = getattr(face_result, "_detection_info", None)
            detection_result = analyze_image(target_path, detection_info=detection_info)
        else:
            face_result = None
            detection_result = analyze_video(target_path)

        # Attach metadata analysis to detection result
        detection_result.metadata_analysis = metadata_analysis
    except Exception as e:
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception:
                pass
        clean_err = sanitize_error_message(str(e))
        logger.error(f"[{analysis_id}] Forensic pipeline execution failed: {clean_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {clean_err}"
        )

    # 9. Structured Analysis Record Construction
    created_at = datetime.now().strftime("%b %d, %Y • %H:%M")
    if media_type == "image":
        faces_count = len(detection_result.faces) if (detection_result.faces and detection_result.status != "no_face_detected") else 0
    else:
        faces_count = detection_result.faces_detected if detection_result.faces_detected is not None else 0
        if detection_result.status == "no_face_detected" or faces_count == 0:
            face_result = FaceDetectionResult(status="no_face_detected", faces=[], count=0)
        else:
            rep_boxes = [f.bounding_box for f in (detection_result.frame_results or []) if f.face_detected and f.bounding_box]
            face_result = FaceDetectionResult(
                status="completed",
                faces=rep_boxes[:faces_count] if rep_boxes else (face_result.faces if face_result and face_result.faces else []),
                count=faces_count
            )

    response = AnalysisResponse(
        analysis_id=analysis_id,
        filename=safe_name,
        media_type=media_type,
        media_url=f"/uploads/{target_filename}",
        file_size=metadata.file_size,
        file_size_human=metadata.file_size_human,
        status=detection_result.status,
        metadata=metadata,
        faces_detected=faces_count,
        face_detection=face_result,
        detection=detection_result,
        model=detection_result.model,
        prediction=detection_result.prediction,
        probability=detection_result.probability,
        detection_probability=detection_result.detection_probability,
        model_probability=detection_result.model_probability,
        video_probability=detection_result.video_probability,
        real_probability=detection_result.real_probability,
        fake_probability=detection_result.fake_probability,
        assessment=detection_result.assessment,
        faces=detection_result.faces,
        selected_face_id=detection_result.selected_face_id,
        analysis_mode=detection_result.analysis_mode or "primary_face",
        primary_face_result=detection_result.primary_face_result,
        all_faces_result=detection_result.all_faces_result,
        is_low_margin=detection_result.is_low_margin,
        margin_note=detection_result.margin_note,
        task_scope_note=detection_result.task_scope_note,
        frame_results=detection_result.frame_results,
        frames_analyzed=detection_result.frames_analyzed,
        duration=detection_result.duration or metadata.duration,
        fps=detection_result.fps or metadata.fps,
        gradcam_available=detection_result.gradcam_available,
        gradcam_heatmap_url=detection_result.gradcam_heatmap_url,
        gradcam_overlay_url=detection_result.gradcam_overlay_url,
        original_face_crop_url=detection_result.original_face_crop_url,
        metadata_analysis=metadata_analysis,
        frequency_analysis=detection_result.frequency_analysis,
        temporal_analysis=detection_result.temporal_analysis,
        created_at=created_at
    )

    # Store for case retrieval
    analysis_store[analysis_id] = response

    elapsed = time.time() - start_time
    logger.info(f"[{analysis_id}] Analysis ingestion completed in {elapsed:.2f}s. Model status: {detection_result.status}.")

    return response

@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_record(analysis_id: str):
    """Retrieves an existing analysis case by ID."""
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", analysis_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID format."
        )
    if analysis_id not in analysis_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis case {analysis_id} not found."
        )
    return analysis_store[analysis_id]

@router.get("/history", response_model=List[AnalysisResponse])
async def get_analysis_history():
    """Retrieves all stored analysis cases in reverse chronological order."""
    return list(reversed(list(analysis_store.values())))
