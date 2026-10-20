from pathlib import Path
from typing import Dict, Any, List, Union, Tuple, Optional
import os
import cv2
import numpy as np
from ..models.schemas import FaceDetectionResult, FaceBoundingBox
from ..utils.logger import logger

from ..config import YUNET_MODEL_PATH

_face_detector = None


def get_face_detector():
    """Initializes and returns the singleton FaceDetectorYN detector."""
    global _face_detector
    if _face_detector is not None:
        return _face_detector

    if not YUNET_MODEL_PATH.exists():
        logger.error(f"YuNet model not found at {YUNET_MODEL_PATH}")
        raise FileNotFoundError(f"Face detection model not found at {YUNET_MODEL_PATH}")

    # Create OpenCV FaceDetectorYN
    # Default input size (320, 320) will be dynamically adjusted per image
    _face_detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL_PATH),
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    logger.info("OpenCV YuNet face detector initialized successfully.")
    return _face_detector


def detect_and_crop_faces(
    image_input: Union[Path, str, np.ndarray],
    save_crops: bool = False,
    crops_dir: Optional[Path] = None,
    padding_ratio: float = 0.20
) -> Dict[str, Any]:
    """
    Detects faces in an image, extracts bounding boxes, and generates face crops.

    Args:
        image_input: Path to image file or numpy BGR array.
        save_crops: Whether to persist face crops to disk.
        crops_dir: Directory where crops are saved (if save_crops is True).
        padding_ratio: Margin padding around the face box (0.20 = 20% contextual padding).

    Returns:
        Dict containing:
        - "count": int (number of detected faces)
        - "bounding_boxes": List[Dict[str, int]] (list of {"x", "y", "width", "height"})
        - "crops": List[np.ndarray] (list of BGR face crop arrays)
        - "crop_paths": List[str] (list of file paths if saved, else empty)
        - "status": "faces_detected" | "no_face_detected"
    """
    if isinstance(image_input, (Path, str)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image at {image_path}")
        source_name = image_path.stem
    elif isinstance(image_input, np.ndarray):
        img_bgr = image_input
        source_name = "crop"
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img_h, img_w = img_bgr.shape[:2]
    detector = get_face_detector()
    detector.setInputSize((img_w, img_h))

    _, detections = detector.detect(img_bgr)

    if detections is None or len(detections) == 0:
        logger.info(f"No face detected in {source_name} ({img_w}x{img_h}).")
        return {
            "count": 0,
            "bounding_boxes": [],
            "crops": [],
            "crop_paths": [],
            "status": "no_face_detected",
            "source_image_bgr": img_bgr
        }

    bounding_boxes: List[Dict[str, int]] = []
    crops: List[np.ndarray] = []
    crop_paths: List[str] = []

    if save_crops and crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for idx, det in enumerate(detections):
        # Bounding box coordinates from YuNet
        x, y, w_box, h_box = [int(v) for v in det[:4]]
        confidence = float(det[-1])

        # Validate bounding box boundaries
        if w_box <= 0 or h_box <= 0:
            continue

        # Add padding around face box
        pad_x = int(w_box * padding_ratio)
        pad_y = int(h_box * padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w_box + pad_x)
        y2 = min(img_h, y + h_box + pad_y)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        bbox_dict = {
            "x": x,
            "y": y,
            "width": w_box,
            "height": h_box
        }
        bounding_boxes.append(bbox_dict)
        crops.append(crop)

        if save_crops and crops_dir:
            crop_filename = f"{source_name}_face_{idx}.jpg"
            crop_file_path = crops_dir / crop_filename
            cv2.imwrite(str(crop_file_path), crop)
            crop_paths.append(str(crop_file_path))

    count = len(crops)
    status = "faces_detected" if count > 0 else "no_face_detected"
    logger.info(f"Detected {count} face(s) in {source_name}.")

    return {
        "count": count,
        "bounding_boxes": bounding_boxes,
        "crops": crops,
        "crop_paths": crop_paths,
        "status": status,
        "source_image_bgr": img_bgr
    }


def detect_faces(media_path: Path, media_type: str) -> FaceDetectionResult:
    """
    Standard interface used by FastAPI endpoints for face detection.
    Supports both image and video media inputs using OpenCV YuNet.
    """
    try:
        if media_type == "video":
            cap = cv2.VideoCapture(str(media_path))
            if not cap.isOpened():
                return FaceDetectionResult(status="error", faces=[], count=0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // 10) if total_frames > 0 else 1
            sample_indices = list(range(0, min(total_frames, 100), step))[:10]

            detected_boxes = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                detection_info = detect_and_crop_faces(frame)
                if detection_info["count"] > 0:
                    detected_boxes = [FaceBoundingBox(**b) for b in detection_info["bounding_boxes"]]
                    break
            cap.release()

            if len(detected_boxes) == 0:
                return FaceDetectionResult(status="no_face_detected", faces=[], count=0)
            return FaceDetectionResult(status="completed", faces=detected_boxes, count=len(detected_boxes))

        # Image detection branch
        detection_info = detect_and_crop_faces(media_path)
        if detection_info["count"] == 0:
            res = FaceDetectionResult(status="no_face_detected", faces=[], count=0)
        else:
            boxes = [FaceBoundingBox(**b) for b in detection_info["bounding_boxes"]]
            res = FaceDetectionResult(status="completed", faces=boxes, count=len(boxes))
        res._detection_info = detection_info
        return res
    except Exception as e:
        logger.error(f"Face detection error on {media_path}: {e}")
        return FaceDetectionResult(status="error", faces=[], count=0)
