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
    VideoFrameAnalysisItem,
    PrimaryFaceResult,
    AllFacesResult
)
from ..models.temporal_schemas import TemporalFrameItem
from ..utils.logger import logger
from ..utils.file_utils import sanitize_error_message
from .face_service import detect_and_crop_faces

from ..config import (
    XCEPTION_CONFIG_PATH as CONFIG_PATH,
    XCEPTION_WEIGHTS_PATH as WEIGHTS_PATH,
    BENCH_DIR,
    TRAINING_DIR,
    BACKEND_DIR,
    WORKSPACE_ROOT
)

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

TASK_SCOPE_DISCLOSURE = (
    "DEEPTRACE Xception is specialized for localized facial-manipulation detection (face-swaps, deepfake reenactments). "
    "It has not been validated as a general-purpose detector of full-scene AI-generated imagery (e.g. latent diffusion "
    "or GAN synthesis). A REAL classification indicates the absence of localized facial manipulation artifacts, not certified non-AI provenance."
)

ALL_FACES_LIMITATION_DISCLOSURE = (
    "All-Faces Limitation: The unweighted arithmetic mean assigns equal weight to every detected face regardless of size. "
    "Small background detections or false positives can influence the aggregate. Primary Face Analysis is the default standard for examining the primary subject."
)


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
    std: Tuple = (0.5, 0.5, 0.5),
    is_bgr: bool = True
) -> torch.Tensor:
    """
    Applies the exact DeepfakeBench Xception preprocessing:
    1. BGR -> RGB (if input is BGR)
    2. Resize to 256x256 using cv2.INTER_CUBIC
    3. ToTensor() scaled to [0.0, 1.0]
    4. Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) mapped to [-1.0, 1.0]
    """
    if crop is None or crop.size == 0:
        raise ValueError("Invalid or empty face crop.")

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if is_bgr else crop
    img_resized = cv2.resize(img_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=list(mean), std=list(std))
    ])
    tensor = transform(img_resized).unsqueeze(0)  # [1, 3, 256, 256]
    return tensor


def predict_face_crop(crop: np.ndarray, threshold: float = DEEPFAKE_THRESHOLD, is_bgr: bool = True) -> Dict[str, Any]:
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

    tensor = preprocess_face_crop(crop, resolution=resolution, mean=mean, std=std, is_bgr=is_bgr)
    tensor = tensor.to(device)

    try:
        with _inference_lock:
            with torch.inference_mode():
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

    raw_logits_list = [round(float(logits[0][0].item()), 4), round(float(logits[0][1].item()), 4)]

    # Comprehensive forensic diagnostic logging
    logger.info(
        f"[Xception Forensic Diagnostic] Model: Xception | Checkpoint: {WEIGHTS_PATH.name} | "
        f"Input: {resolution}x{resolution} RGB | Raw Logits: {raw_logits_list} | "
        f"Class 0 (Authentic): {real_prob:.4f} | Class 1 (Manipulated): {fake_prob:.4f} | "
        f"Selected: {prediction} (prob={confidence:.4f}) | Decision Threshold: {threshold:.2f}"
    )

    return {
        "prediction": prediction,
        "probability": round(confidence, 4),
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "raw_logits": raw_logits_list
    }


def analyze_image(
    image_path: Path,
    detection_info: Optional[Dict[str, Any]] = None
) -> DetectionResult:
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

    # 1. Face Detection and Cropping (Reuse detection_info if already available)
    if detection_info is None:
        detection_info = detect_and_crop_faces(image_path)

    # 2. Check if no face is detected
    if detection_info["count"] == 0:
        logger.info(f"No faces detected in {image_path.name}; returning no_face_detected.")
        from .frequency_service import analyze_frequency_domain
        full_bgr = detection_info.get("source_image_bgr")
        if full_bgr is None:
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
    boxes = detection_info["bounding_boxes"]
    crops = detection_info["crops"]

    # Image geometry for relative bounding-box area calculation
    full_bgr = detection_info.get("source_image_bgr")
    if full_bgr is not None:
        img_h, img_w = full_bgr.shape[:2]
    else:
        img_temp = cv2.imread(str(image_path))
        img_h, img_w = img_temp.shape[:2] if img_temp is not None else (1, 1)
    img_area = max(1, img_w * img_h)

    # Primary face selection: Select largest valid face by area (width * height)
    areas = [b["width"] * b["height"] for b in boxes]
    primary_idx = int(np.argmax(areas)) if areas else 0

    from .gradcam_service import generate_gradcam_for_face

    for idx, (bbox_dict, crop) in enumerate(zip(boxes, crops)):
        pred_result = predict_face_crop(crop)
        bbox = FaceBoundingBox(**bbox_dict)
        area = bbox.width * bbox.height
        rel_area = round((area / img_area) * 100, 2)
        is_primary = (idx == primary_idx)
        face_id = f"face_{idx}"

        # Generate per-face Grad-CAM visual attribution
        cam_prefix = f"{image_path.stem}_face{idx}"
        cam_res = generate_gradcam_for_face(crop, identifier_prefix=cam_prefix, target_class=1)

        face_item = FaceAnalysisItem(
            face_id=face_id,
            face_index=idx,
            bounding_box=bbox,
            bbox=bbox,
            area=area,
            relative_area=rel_area,
            is_primary=is_primary,
            prediction=pred_result["prediction"],
            probability=pred_result["probability"],
            real_probability=pred_result["real_probability"],
            fake_probability=pred_result["fake_probability"],
            raw_logits=pred_result.get("raw_logits"),
            gradcam_available=cam_res.get("gradcam_available", False),
            gradcam_heatmap_url=cam_res.get("gradcam_heatmap_url"),
            gradcam_overlay_url=cam_res.get("gradcam_overlay_url"),
            original_face_crop_url=cam_res.get("original_face_crop_url")
        )
        face_items.append(face_item)

    # 4. Primary-Face Mode (Default)
    primary_face = face_items[primary_idx]
    
    # Margin note for primary face if near 0.50 decision threshold
    margin_diff = abs((primary_face.fake_probability or 0.5) - DEEPFAKE_THRESHOLD)
    is_low_margin = bool(margin_diff < 0.10)
    margin_note = (
        f"Low-margin model prediction ({round((primary_face.probability or 0.5)*100, 1)}% confidence, "
        f"near {int(DEEPFAKE_THRESHOLD*100)}% decision boundary)"
        if is_low_margin
        else None
    )

    primary_face_result = PrimaryFaceResult(
        face_id=primary_face.face_id or f"face_{primary_idx}",
        face_index=primary_face.face_index,
        prediction=primary_face.prediction,
        probability=primary_face.probability,
        real_probability=primary_face.real_probability or 0.5,
        fake_probability=primary_face.fake_probability or 0.5,
        raw_logits=primary_face.raw_logits,
        bounding_box=primary_face.bounding_box,
        area=primary_face.area or (primary_face.bounding_box.width * primary_face.bounding_box.height),
        relative_area=primary_face.relative_area or 0.0,
        margin_note=margin_note,
        is_low_margin=is_low_margin,
        task_scope_note=TASK_SCOPE_DISCLOSURE
    )

    # 5. All-Faces Aggregation
    # Compute mean_fake and mean_real directly from the exact same valid face set
    valid_fakes = [f.fake_probability for f in face_items if f.fake_probability is not None]
    valid_reals = [f.real_probability for f in face_items if f.real_probability is not None]
    mean_fake_p = round(float(np.mean(valid_fakes)), 4) if valid_fakes else 0.5
    mean_real_p = round(float(np.mean(valid_reals)), 4) if valid_reals else 0.5

    all_faces_pred = "fake" if mean_fake_p >= DEEPFAKE_THRESHOLD else "real"
    # Displayed probability must strictly match the assigned class:
    # If verdict is 'fake', it is mean_fake_p; if verdict is 'real', it is mean_real_p.
    all_faces_prob = mean_fake_p if all_faces_pred == "fake" else mean_real_p

    all_faces_low_margin = bool(abs(mean_fake_p - DEEPFAKE_THRESHOLD) < 0.10)
    all_faces_margin_note = (
        f"Low-margin aggregate prediction ({round(all_faces_prob * 100, 1)}% {all_faces_pred.upper()}, "
        f"near {int(DEEPFAKE_THRESHOLD * 100)}% decision boundary)"
        if all_faces_low_margin
        else None
    )

    all_faces_result = AllFacesResult(
        face_count=len(face_items),
        prediction=all_faces_pred,
        probability=all_faces_prob,
        mean_fake_probability=mean_fake_p,
        mean_real_probability=mean_real_p,
        aggregation_rule="Unweighted mean of individual face probabilities across all detected faces",
        margin_note=all_faces_margin_note,
        is_low_margin=all_faces_low_margin,
        limitation_disclosure=ALL_FACES_LIMITATION_DISCLOSURE
    )

    # Default image verdict reflects Primary Face Analysis
    overall_pred = primary_face.prediction
    overall_prob = primary_face.probability
    image_real_prob = primary_face.real_probability
    image_fake_prob = primary_face.fake_probability
    assessment = "Likely authentic" if overall_pred == "real" else "Likely manipulated"

    # 6. Frequency-domain forensic analysis on the primary face crop
    from .frequency_service import analyze_frequency_domain
    cam_crop = crops[primary_idx]
    cam_prefix = f"{image_path.stem}_face{primary_idx}"
    freq_res = analyze_frequency_domain(
        cam_crop,
        identifier_prefix=cam_prefix,
        analysis_region="face_region"
    )

    logger.info(
        f"Analysis complete for {image_path.name}: {len(face_items)} face(s) analyzed. "
        f"Primary Face #{primary_idx+1} ({primary_face.area}px, {primary_face.relative_area}% of image): "
        f"Verdict={overall_pred} ({assessment}) with model probability {overall_prob:.4f}. "
        f"All-Faces Mean Fake Prob={mean_fake_p:.4f}."
    )

    return DetectionResult(
        status="completed",
        model=ModelDetails(name="Xception", source="DeepfakeBench"),
        prediction=overall_pred,
        probability=round(overall_prob, 4),
        detection_probability=round(overall_prob, 4),
        model_probability=round(overall_prob, 4),
        real_probability=image_real_prob,
        fake_probability=image_fake_prob,
        assessment=assessment,
        faces=face_items,
        selected_face_id=f"face_{primary_idx}",
        analysis_mode="primary_face",
        primary_face_result=primary_face_result,
        all_faces_result=all_faces_result,
        is_low_margin=is_low_margin,
        margin_note=margin_note,
        task_scope_note=TASK_SCOPE_DISCLOSURE,
        gradcam_available=primary_face.gradcam_available or False,
        gradcam_heatmap_url=primary_face.gradcam_heatmap_url,
        gradcam_overlay_url=primary_face.gradcam_overlay_url,
        original_face_crop_url=primary_face.original_face_crop_url,
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
                        real_probability=None,
                        fake_probability=None,
                        raw_logits=None,
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
                        selected_face_crop_rgb = cv2.cvtColor(selected_face_crop, cv2.COLOR_BGR2RGB)
                        meso_res = meso4_det.predict_face_crop(selected_face_crop_rgb)
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
                        real_probability=primary_pred.get("real_probability"),
                        fake_probability=primary_pred.get("fake_probability"),
                        raw_logits=primary_pred.get("raw_logits"),
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
            real_probability=None,
            fake_probability=None,
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

    # 5. Video-Level Statistical Aggregation (Official DeepfakeBench Frame-Average Aggregation)
    # Explicit semantic definitions:
    # - Eligible frames: valid frames where a face was detected and inference succeeded (m_valid).
    # - Excluded frames: frames where no face was detected (excluded_count).
    #   Non-face frames are not assigned 0.0 fake probability so they do not artificially skew the mean.
    # - frame_fake_probability: P(fake | frame_t) from model softmax.
    # - frame_real_probability: 1.0 - frame_fake_probability.
    # - video_fake_probability: arithmetic mean of frame_fake_probability over all m_valid eligible frames.
    # - video_real_probability: 1.0 - video_fake_probability.
    # - final video verdict: "fake" if video_fake_probability >= DEEPFAKE_THRESHOLD (0.50) else "real".
    # - video_probability: confidence of the final verdict (video_fake_prob for fake, video_real_prob for real).
    excluded_count = len(frame_results) - m_valid

    fake_probs = []
    for f in valid_frames:
        if f.fake_probability is not None:
            fake_probs.append(f.fake_probability)
        elif f.prediction == "fake":
            fake_probs.append(f.probability)
        else:
            fake_probs.append(round(1.0 - f.probability, 4))

    # Mean frame manipulation probability
    mean_fake_prob = sum(fake_probs) / m_valid
    video_fake_prob = round(mean_fake_prob, 4)
    video_real_prob = round(1.0 - mean_fake_prob, 4)

    if mean_fake_prob >= DEEPFAKE_THRESHOLD:
        overall_pred = "fake"
        video_prob = video_fake_prob
        assessment = "Likely manipulated"
    else:
        overall_pred = "real"
        video_prob = video_real_prob
        assessment = "Likely authentic"

    logger.info(
        f"[Video Aggregation Audit] Video: {video_path.name} | Total Frames: {len(frame_results)} | "
        f"Eligible Face Frames (m_valid): {m_valid} | Excluded (no-face): {excluded_count} | "
        f"Video Fake Probability: {video_fake_prob:.4f} | Video Real Probability: {video_real_prob:.4f} | "
        f"Decision Threshold: {DEEPFAKE_THRESHOLD:.2f} | Final Verdict: {overall_pred} ({assessment}) | "
        f"Reported Probability: {video_prob:.4f}"
    )

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
        key=lambda f: f.fake_probability if f.fake_probability is not None else (f.probability if f.prediction == "fake" else round(1.0 - f.probability, 4)),
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

    # Free in-memory face crop arrays
    frame_face_crops.clear()

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
        detection_probability=video_fake_prob,
        model_probability=video_prob,
        video_probability=video_prob,
        real_probability=video_real_prob,
        fake_probability=video_fake_prob,
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

