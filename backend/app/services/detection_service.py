from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import os
import sys
import types
import yaml
import math
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import threading

from ..config import DEEPFAKE_THRESHOLD, VIDEO_SAMPLE_INTERVAL, MAX_ANALYZED_FRAMES, THUMBNAILS_DIR
from ..models.schemas import (
    DetectionResult,
    FaceAnalysisItem,
    FaceBoundingBox,
    ModelDetails,
    VideoFrameAnalysisItem
)
from ..models.temporal_schemas import TemporalFrameItem
from ..utils.logger import logger
from ..utils.file_utils import sanitize_error_message
from .face_service import detect_and_crop_faces

# Path configuration
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = BACKEND_DIR.parent
DEEPFAKE_DIR = WORKSPACE_ROOT / "deepfake_model"
BENCH_DIR = DEEPFAKE_DIR / "DeepfakeBench"
TRAINING_DIR = BENCH_DIR / "training"
CONFIG_PATH = TRAINING_DIR / "config" / "detector" / "xception.yaml"
WEIGHTS_PATH = TRAINING_DIR / "weights" / "xception_best.pth"

# Setup python path for DeepfakeBench modules
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

# Ensure detectors registry module is mapped cleanly without loading unrelated detectors
from metrics.registry import DETECTOR
if 'detectors' not in sys.modules:
    det_mod = types.ModuleType('detectors')
    det_mod.__path__ = [str(TRAINING_DIR / 'detectors')]
    det_mod.DETECTOR = DETECTOR
    sys.modules['detectors'] = det_mod

from detectors.xception_detector import XceptionDetector

# Global cached model and device
_model = None
_model_config = None
_device = None
_inference_lock = threading.Lock()


def get_xception_detector():
    """Initializes and returns the singleton XceptionDetector loaded with official weights."""
    global _model, _model_config, _device
    if _model is not None:
        return _model, _model_config, _device

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Xception config file not found: {CONFIG_PATH}")
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Xception weights checkpoint not found: {WEIGHTS_PATH}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        _device = torch.device('cuda')
        logger.info(f"Hardware acceleration enabled: CUDA device '{gpu_name}' detected.")
    else:
        _device = torch.device('cpu')
        logger.info("CUDA acceleration unavailable; falling back gracefully to CPU inference.")

    with open(CONFIG_PATH, 'r') as f:
        _model_config = yaml.safe_load(f)

    # Ensure pretrained ImageNet backbone path is absolute
    pretrained_path = _model_config.get('pretrained', '')
    if not os.path.isabs(pretrained_path):
        _model_config['pretrained'] = str((BENCH_DIR / pretrained_path).resolve())

    # Build model using exact repository detector implementation
    detector = XceptionDetector(_model_config)
    detector = detector.to(_device)

    # Load official deepfake-trained checkpoint with strict=True
    ckpt = torch.load(str(WEIGHTS_PATH), map_location=_device)
    detector.load_state_dict(ckpt, strict=True)
    detector.eval()

    _model = detector
    logger.info("Pretrained DeepfakeBench Xception detector loaded and ready for inference.")
    return _model, _model_config, _device


def preprocess_face_crop(
    crop: np.ndarray,
    resolution: int = 256,
    mean: Tuple = (0.5, 0.5, 0.5),
    std: Tuple = (0.5, 0.5, 0.5)
) -> torch.Tensor:
    """
    Applies the exact DeepfakeBench Xception preprocessing:
    1. BGR -> RGB
    2. Resize to 256x256 using cv2.INTER_CUBIC
    3. ToTensor() scaled to [0.0, 1.0]
    4. Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) mapped to [-1.0, 1.0]
    """
    if crop is None or crop.size == 0:
        raise ValueError("Invalid or empty face crop.")

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std))
    ])
    tensor = transform(img_resized).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor


def predict_face_crop(crop: np.ndarray, threshold: float = DEEPFAKE_THRESHOLD) -> Dict[str, Any]:
    """
    Runs Xception inference on a single face crop.
    
    Decision Threshold:
    - DEEPFAKE_THRESHOLD (default: 0.5) separates authentic from manipulated classifications.
    
    Terminology Note:
    - Outputs express model detection probabilities, not forensic certainty.
    
    Returns:
    - prediction: "real" or "fake"
    - probability: model probability for the predicted class
    - real_probability: probability of real (class 0)
    - fake_probability: probability of fake (class 1)
    """
    detector, config, device = get_xception_detector()
    resolution = config.get('resolution', 256)
    mean = config.get('mean', [0.5, 0.5, 0.5])
    std = config.get('std', [0.5, 0.5, 0.5])

    tensor = preprocess_face_crop(crop, resolution=resolution, mean=mean, std=std)
    tensor = tensor.to(device)

    try:
        with _inference_lock:
            with torch.no_grad():
                output = detector({'image': tensor})

        # Convert logits to probability via softmax
        logits = output['cls']  # [1, 2]
        probs = torch.softmax(logits, dim=1)[0]
        real_prob = float(probs[0].item())
        fake_prob = float(probs[1].item())
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"CUDA out of memory during inference: {e}")
        raise RuntimeError("GPU out of memory during model inference.")
    except Exception as e:
        clean_err = sanitize_error_message(str(e))
        logger.error(f"Inference failure: {clean_err}")
        raise RuntimeError(f"Model inference failed: {clean_err}")

    if fake_prob >= threshold:
        prediction = "fake"
        confidence = fake_prob
    else:
        prediction = "real"
        confidence = real_prob

    return {
        "prediction": prediction,
        "probability": round(confidence, 4),
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4)
    }


def analyze_image(image_path: Path) -> DetectionResult:
    """
    Full real image pipeline:
    Uploaded Image -> Face Detection -> Face Bounding Box -> Face Crop
    -> Resize (256x256) -> Normalization ([-1, 1]) -> Xception -> Real/Fake Output
    
    Multi-Face Aggregation Strategy:
    - If ANY detected face exhibits fake_probability >= DEEPFAKE_THRESHOLD, the overall image
      is classified as manipulated (prediction="fake", assessment="Likely manipulated").
      The reported detection probability is the maximum manipulation probability observed across faces.
    - If ALL detected faces are below DEEPFAKE_THRESHOLD, the overall image is classified as authentic
      (prediction="real", assessment="Likely authentic").
      The reported model probability is the arithmetic mean of authentic probabilities across faces.
    - If NO face is detected, status="no_face_detected" is returned (the image is NOT classified as fake).
    """
    logger.info(f"Running deepfake analysis on image: {image_path.name}")

    # 1. Face Detection and Cropping
    detection_info = detect_and_crop_faces(image_path)

    # 2. Check if no face is detected
    if detection_info["count"] == 0:
        logger.info(f"No faces detected in {image_path.name}; returning no_face_detected.")
        from .frequency_service import analyze_frequency_domain
        full_bgr = cv2.imread(str(image_path))
        freq_res = analyze_frequency_domain(
            full_bgr,
            identifier_prefix=f"{image_path.stem}_full",
            analysis_region="full_image"
        ) if full_bgr is not None else None

        return DetectionResult(
            status="no_face_detected",
            model=ModelDetails(name="Xception", source="DeepfakeBench"),
            prediction=None,
            probability=None,
            detection_probability=None,
            model_probability=None,
            assessment="No face detected",
            faces=[],
            frequency_analysis=freq_res
        )

    # 3. Analyze each detected face crop independently
    face_items: List[FaceAnalysisItem] = []
    fake_confidences: List[float] = []
    real_confidences: List[float] = []

    for idx, (bbox_dict, crop) in enumerate(zip(detection_info["bounding_boxes"], detection_info["crops"])):
        pred_result = predict_face_crop(crop)
        bbox = FaceBoundingBox(**bbox_dict)

        face_item = FaceAnalysisItem(
            face_index=idx,
            bounding_box=bbox,
            prediction=pred_result["prediction"],
            probability=pred_result["probability"],
            real_probability=pred_result["real_probability"],
            fake_probability=pred_result["fake_probability"]
        )
        face_items.append(face_item)

        if pred_result["prediction"] == "fake":
            fake_confidences.append(pred_result["fake_probability"])
        else:
            real_confidences.append(pred_result["real_probability"])

    # 4. Multi-Face Aggregation (Conservative Any-Face Policy)
    if len(fake_confidences) > 0:
        overall_pred = "fake"
        overall_prob = max(fake_confidences)
        assessment = "Likely manipulated"
        # Select the face crop that exhibited the highest manipulation score
        selected_face_idx = max(range(len(face_items)), key=lambda i: face_items[i].fake_probability or 0.0)
    else:
        overall_pred = "real"
        overall_prob = sum(real_confidences) / len(real_confidences) if real_confidences else 0.5
        assessment = "Likely authentic"
        selected_face_idx = 0

    # 5. Generate real Xception Grad-CAM explainability for the selected face crop
    from .gradcam_service import generate_gradcam_for_face
    cam_crop = detection_info["crops"][selected_face_idx]
    cam_prefix = f"{image_path.stem}_face{selected_face_idx}"
    cam_res = generate_gradcam_for_face(cam_crop, identifier_prefix=cam_prefix, target_class=1)

    # 6. Frequency-domain forensic analysis on the analyzed face crop
    from .frequency_service import analyze_frequency_domain
    freq_res = analyze_frequency_domain(
        cam_crop,
        identifier_prefix=cam_prefix,
        analysis_region="face_region"
    )

    logger.info(
        f"Analysis complete for {image_path.name}: {len(face_items)} face(s) analyzed. "
        f"Verdict: {overall_pred} ({assessment}) with model probability {overall_prob:.4f}. "
        f"Grad-CAM available: {cam_res.get('gradcam_available', False)}, Frequency spectrum: {freq_res.frequency_spectrum_url}."
    )

    return DetectionResult(
        status="completed",
        model=ModelDetails(name="Xception", source="DeepfakeBench"),
        prediction=overall_pred,
        probability=round(overall_prob, 4),
        detection_probability=round(overall_prob, 4),
        model_probability=round(overall_prob, 4),
        assessment=assessment,
        faces=face_items,
        gradcam_available=cam_res.get("gradcam_available", False),
        gradcam_heatmap_url=cam_res.get("gradcam_heatmap_url"),
        gradcam_overlay_url=cam_res.get("gradcam_overlay_url"),
        original_face_crop_url=cam_res.get("original_face_crop_url"),
        frequency_analysis=freq_res
    )


def analyze_image_dict(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Runs the full image analysis pipeline and returns the result as a dictionary
    matching the exact specification:
    
    If no face is detected:
        {"status": "no_face_detected"}
        
    If face(s) are detected:
        {
            "status": "completed",
            "model": {
                "name": "Xception",
                "source": "DeepfakeBench"
            },
            "faces_detected": 1,
            "faces": [
                {
                    "face_index": 0,
                    "bounding_box": {"x": ..., "y": ..., "width": ..., "height": ...},
                    "prediction": "real_or_fake",
                    "probability": ...
                }
            ]
        }
    """
    res = analyze_image(Path(image_path))
    if res.status == "no_face_detected":
        return {"status": "no_face_detected"}

    model_info = res.model.model_dump() if hasattr(res.model, 'model_dump') else {"name": "Xception", "source": "DeepfakeBench"}

    return {
        "status": res.status,
        "media_type": "image",
        "filename": Path(image_path).name,
        "model": model_info,
        "faces_detected": len(res.faces or []),
        "faces": [
            {
                "face_index": f.face_index,
                "bounding_box": f.bounding_box.model_dump(),
                "prediction": f.prediction,
                "probability": f.probability
            }
            for f in (res.faces or [])
        ],
        "prediction": res.prediction,
        "probability": res.probability,
        "assessment": res.assessment
    }



def analyze_video(
    video_path: Path,
    sample_interval: int = VIDEO_SAMPLE_INTERVAL,
    max_frames: int = MAX_ANALYZED_FRAMES
) -> DetectionResult:
    """
    Video deepfake forensic pipeline:
    VIDEO -> OpenCV -> Read metadata -> Sample frames -> Face Detection
    -> Face Crop -> Xception -> Frame Probability -> Video-level Aggregation

    For every sampled frame, extracts:
    - frame_number: int
    - timestamp: float (seconds)
    - face_detected: bool
    - bounding_box: Optional[FaceBoundingBox]
    - prediction: Optional[str] ("fake" | "real")
    - probability: Optional[float]
    - thumbnail_url: Optional[str]

    Video-Level Statistical Aggregation (Top-K Suspicious Quantile Aggregation):
    - Video manipulation attacks often target specific segments (e.g. 2-second face-swap).
      A simple global arithmetic mean dilutes genuine manipulations, while a single maximum (max)
      is overly sensitive to detector outliers.
    - We collect all valid frames where face_detected == True.
    - Let M be the count of valid face frames.
    - If M == 0:
        Returns status="no_face_detected", prediction=None, video_probability=None.
    - If M > 0:
        Sort frames descending by manipulation probability P(fake).
        Select top k = max(1, math.ceil(0.20 * M)) (top 20% most suspicious frames).
        P_video_fake = mean(top_k(fake_probabilities)).
        If P_video_fake >= DEEPFAKE_THRESHOLD:
            prediction = "fake"
            video_probability = round(top_k_suspicion, 4)
            assessment = "Likely manipulated"
        Else:
            P_video_real = 1.0 - (sum(fake_probabilities) / M)
            prediction = "real"
            video_probability = round(P_video_real, 4)
            assessment = "Likely authentic"
    """
    video_path = Path(video_path)
    logger.info(
        f"Running video deepfake analysis on {video_path.name} "
        f"(sample_interval={sample_interval}, max_frames={max_frames})"
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video stream at {video_path.name}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        duration = round(total_frames / fps, 2) if total_frames > 0 else 0.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if total_frames <= 0:
            raise ValueError(f"Video {video_path.name} contains no readable frames.")

        # 1. Calculate sampled frame indices
        step = max(1, sample_interval)
        candidate_indices = list(range(0, total_frames, step))
        if len(candidate_indices) > max_frames:
            sampled_indices = [int(round(i)) for i in np.linspace(0, total_frames - 1, max_frames)]
        else:
            sampled_indices = candidate_indices

        # Resolve comparison detector Meso4 for multi-model temporal alignment
        from .model_registry import get_model_registry
        registry = get_model_registry()
        meso4_det = registry.get_detector("meso4_deepfakebench")

        logger.info(
            f"Sampling {len(sampled_indices)} of {total_frames} total frames "
            f"from {video_path.name} ({duration}s @ {fps:.1f} FPS)."
        )

        frame_results: List[VideoFrameAnalysisItem] = []
        temporal_frame_items: List[TemporalFrameItem] = []
        frame_face_crops: Dict[int, np.ndarray] = {}
        max_faces_detected = 0

        # 2. Iterate through sampled frames
        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue

            timestamp = round(frame_idx / fps, 2)

            # Save frame thumbnail for frontend timeline / suspicious frames display
            thumb_filename = f"{video_path.stem}_f{frame_idx}.jpg"
            thumb_path = THUMBNAILS_DIR / thumb_filename
            thumb_h, thumb_w = frame_bgr.shape[:2]
            scale = min(1.0, 320.0 / thumb_w)
            if scale < 1.0:
                thumb_img = cv2.resize(frame_bgr, (int(thumb_w * scale), int(thumb_h * scale)), interpolation=cv2.INTER_AREA)
            else:
                thumb_img = frame_bgr
            cv2.imwrite(str(thumb_path), thumb_img)
            thumbnail_url = f"/uploads/thumbnails/{thumb_filename}"

            # 3. Face Detection on frame
            detection_info = detect_and_crop_faces(frame_bgr, save_crops=False)
            face_count = detection_info["count"]
            if face_count > max_faces_detected:
                max_faces_detected = face_count

            if face_count == 0:
                frame_results.append(
                    VideoFrameAnalysisItem(
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        face_detected=False,
                        bounding_box=None,
                        prediction=None,
                        probability=None,
                        thumbnail_url=thumbnail_url
                    )
                )
                temporal_frame_items.append(
                    TemporalFrameItem(
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        valid=False,
                        face_count=0,
                        face_bbox=None,
                        thumbnail_url=thumbnail_url
                    )
                )
            else:
                # 4. Predict each face crop in frame
                frame_bboxes = []
                frame_preds = []
                frame_fake_probs = []

                for bbox_dict, crop in zip(detection_info["bounding_boxes"], detection_info["crops"]):
                    pred_res = predict_face_crop(crop)
                    bbox = FaceBoundingBox(**bbox_dict)
                    frame_bboxes.append(bbox)
                    frame_preds.append(pred_res)
                    frame_fake_probs.append(pred_res["fake_probability"])

                # Representative face selection for this frame:
                # If any face is classified as fake, highlight that manipulated face.
                # Otherwise, track the primary subject (the largest face by area in the frame).
                face_areas = [b["width"] * b["height"] for b in detection_info["bounding_boxes"]]
                fake_indices = [i for i, p in enumerate(frame_preds) if p["prediction"] == "fake"]

                if fake_indices:
                    selected_idx = max(fake_indices, key=lambda i: frame_preds[i]["fake_probability"])
                else:
                    selected_idx = int(np.argmax(face_areas))

                primary_pred = frame_preds[selected_idx]
                primary_bbox = frame_bboxes[selected_idx]
                selected_face_crop = detection_info["crops"][selected_idx]
                frame_face_crops[frame_idx] = selected_face_crop

                # Secondary comparison model Meso4 inference on identical face crop
                meso_prob = None
                meso_pred = None
                if meso4_det is not None:
                    try:
                        meso_res = meso4_det.predict_face_crop(selected_face_crop)
                        if meso_res.status == "success":
                            meso_prob = meso_res.manipulated_probability
                            meso_pred = meso_res.prediction
                    except Exception as e:
                        logger.warning(f"Meso4 inference failed on frame {frame_idx}: {e}")

                x_manip_prob = primary_pred["fake_probability"]
                x_cat_pred = "manipulated" if primary_pred["prediction"] == "fake" else "authentic"

                temporal_frame_items.append(
                    TemporalFrameItem(
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        valid=True,
                        xception_probability=round(float(x_manip_prob), 4),
                        xception_prediction=x_cat_pred,
                        meso4_probability=round(float(meso_prob), 4) if meso_prob is not None else None,
                        meso4_prediction=meso_pred,
                        face_count=face_count,
                        face_bbox=[primary_bbox.x, primary_bbox.y, primary_bbox.width, primary_bbox.height],
                        thumbnail_url=thumbnail_url
                    )
                )

                frame_results.append(
                    VideoFrameAnalysisItem(
                        frame_number=frame_idx,
                        timestamp=timestamp,
                        face_detected=True,
                        bounding_box=primary_bbox,
                        prediction=primary_pred["prediction"],
                        probability=primary_pred["probability"],
                        thumbnail_url=thumbnail_url
                    )
                )

    finally:
        cap.release()

    # 5. Video-Level Statistical Aggregation (Official DeepfakeBench Frame-Average Aggregation)
    valid_frames = [f for f in frame_results if f.face_detected and f.probability is not None]
    m_valid = len(valid_frames)

    if m_valid == 0:
        logger.info(f"No faces detected across {len(frame_results)} sampled frames in {video_path.name}.")
        from .temporal_analysis_service import get_temporal_analysis_service
        temporal_service = get_temporal_analysis_service()
        empty_temporal = temporal_service.analyze_temporal_consistency(
            frames=temporal_frame_items,
            frame_width=frame_width,
            frame_height=frame_height
        )
        return DetectionResult(
            status="no_face_detected",
            model=ModelDetails(name="Xception", source="DeepfakeBench"),
            prediction=None,
            probability=None,
            detection_probability=None,
            model_probability=None,
            video_probability=None,
            assessment="No face detected",
            faces=[],
            frame_results=frame_results,
            frames_analyzed=len(frame_results),
            faces_detected=0,
            duration=duration,
            fps=fps,
            gradcam_available=False,
            gradcam_heatmap_url=None,
            gradcam_overlay_url=None,
            original_face_crop_url=None,
            temporal_analysis=empty_temporal
        )

    # Calculate frame manipulation probabilities P(fake | frame_t)
    fake_probs = []
    for f in valid_frames:
        if f.prediction == "fake":
            fake_probs.append(f.probability)
        else:
            fake_probs.append(round(1.0 - f.probability, 4))

    # Mean frame manipulation probability
    mean_fake_prob = sum(fake_probs) / m_valid

    if mean_fake_prob >= DEEPFAKE_THRESHOLD:
        overall_pred = "fake"
        video_prob = round(mean_fake_prob, 4)
        assessment = "Likely manipulated"
    else:
        overall_pred = "real"
        video_prob = round(1.0 - mean_fake_prob, 4)
        assessment = "Likely authentic"

    # 6. Generate real Grad-CAM explainability and Frequency-domain analysis for top suspicious frames
    from .gradcam_service import generate_gradcam_for_face
    from .frequency_service import analyze_frequency_domain

    top_cam_overlay = None
    top_cam_heatmap = None
    top_cam_crop = None
    top_cam_available = False
    top_freq_analysis = None

    # Sort valid frames descending by manipulation probability
    sorted_suspicious = sorted(
        valid_frames,
        key=lambda f: f.probability if f.prediction == "fake" else round(1.0 - f.probability, 4),
        reverse=True
    )

    for rank, susp_frame in enumerate(sorted_suspicious[:3]):
        f_num = susp_frame.frame_number
        if f_num in frame_face_crops:
            crop_img = frame_face_crops[f_num]
            cam_prefix = f"{video_path.stem}_f{f_num}"
            cam_res = generate_gradcam_for_face(crop_img, identifier_prefix=cam_prefix, target_class=1)
            susp_frame.gradcam_available = cam_res.get("gradcam_available", False)
            susp_frame.gradcam_overlay_url = cam_res.get("gradcam_overlay_url")
            susp_frame.gradcam_heatmap_url = cam_res.get("gradcam_heatmap_url")
            susp_frame.original_face_crop_url = cam_res.get("original_face_crop_url")

            freq_res = analyze_frequency_domain(crop_img, identifier_prefix=cam_prefix, analysis_region="face_region")
            susp_frame.frequency_analysis = freq_res
            susp_frame.frequency_spectrum_url = freq_res.frequency_spectrum_url

            if rank == 0 and cam_res.get("gradcam_available"):
                top_cam_available = True
                top_cam_overlay = cam_res.get("gradcam_overlay_url")
                top_cam_heatmap = cam_res.get("gradcam_heatmap_url")
                top_cam_crop = cam_res.get("original_face_crop_url")
                top_freq_analysis = freq_res

            for t_item in temporal_frame_items:
                if t_item.frame_number == f_num:
                    t_item.gradcam_available = susp_frame.gradcam_available or False
                    t_item.frequency_available = susp_frame.frequency_analysis is not None

    # 7. Execute Temporal Video Consistency Forensics
    from .temporal_analysis_service import get_temporal_analysis_service
    temporal_service = get_temporal_analysis_service()
    temporal_analysis = temporal_service.analyze_temporal_consistency(
        frames=temporal_frame_items,
        frame_width=frame_width,
        frame_height=frame_height
    )

    logger.info(
        f"Video analysis complete for {video_path.name}: {len(frame_results)} frames analyzed ({m_valid} face frames). "
        f"Verdict: {overall_pred} ({assessment}) with video probability {video_prob:.4f}. "
        f"Temporal analysis status: {temporal_analysis.status} ({len(temporal_analysis.temporal_events)} events). "
        f"Grad-CAM generated: {top_cam_available}, Frequency analysis: {top_freq_analysis is not None}."
    )

    return DetectionResult(
        status="completed",
        model=ModelDetails(name="Xception", source="DeepfakeBench"),
        prediction=overall_pred,
        probability=video_prob,
        detection_probability=video_prob,
        model_probability=video_prob,
        video_probability=video_prob,
        assessment=assessment,
        faces=[],
        frame_results=frame_results,
        frames_analyzed=len(frame_results),
        faces_detected=max_faces_detected,
        duration=duration,
        fps=fps,
        gradcam_available=top_cam_available,
        gradcam_heatmap_url=top_cam_heatmap,
        gradcam_overlay_url=top_cam_overlay,
        original_face_crop_url=top_cam_crop,
        frequency_analysis=top_freq_analysis,
        temporal_analysis=temporal_analysis
    )

