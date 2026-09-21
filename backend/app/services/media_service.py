import os
from pathlib import Path
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import cv2

from ..config import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_DIMENSION,
    MAX_VIDEO_DIMENSION,
    MAX_VIDEO_DURATION_SECONDS,
    MAX_VIDEO_FPS,
)
from ..models.schemas import MediaMetadata
from ..utils.logger import logger
from ..utils.file_utils import sanitize_error_message

def format_file_size(size_bytes: int) -> str:
    """Returns human-readable representation of file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def format_duration(seconds: float) -> str:
    """Formats seconds into MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def extract_image_metadata(file_path: Path) -> MediaMetadata:
    """
    Extracts resolution, format, and size for still image files.
    Enforces pixel and dimension safety bounds against decompression bombs.
    """
    file_size = os.path.getsize(file_path)
    file_size_human = format_file_size(file_size)

    try:
        with Image.open(file_path) as img:
            img.verify()  # Check for corruption

        # Reopen to read dimensions after verify() closes the file
        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format or file_path.suffix.upper().lstrip(".")

            # Resource bounds validation
            total_pixels = width * height
            if total_pixels > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Image pixel count ({total_pixels:,}) exceeds maximum safety limit ({MAX_IMAGE_PIXELS:,} pixels)."
                )

            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension ({MAX_IMAGE_DIMENSION}px)."
                )

            return MediaMetadata(
                width=width,
                height=height,
                format=img_format,
                file_size=file_size,
                file_size_human=file_size_human,
                fps=None,
                frame_count=1,
                duration=None,
                duration_formatted=None
            )
    except Image.DecompressionBombError as e:
        logger.error(f"Decompression bomb detected in image {file_path.name}: {e}")
        raise ValueError("Image exceeds safe decompression limits (possible decompression bomb).")
    except (UnidentifiedImageError, OSError) as e:
        clean_err = sanitize_error_message(str(e))
        logger.error(f"Failed to read image file {file_path.name}: {clean_err}")
        raise ValueError(f"Corrupted or unreadable image file: {clean_err}")

def extract_video_metadata(file_path: Path) -> MediaMetadata:
    """
    Extracts resolution, fps, frame count, and duration using OpenCV.
    Enforces resolution, duration, and frame rate safety bounds.
    """
    file_size = os.path.getsize(file_path)
    file_size_human = format_file_size(file_size)

    cap = cv2.VideoCapture(str(file_path))
    try:
        if not cap.isOpened():
            raise ValueError("Unable to open video stream. Codec may be corrupted or unsupported.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Check readability of first frame
        ret, _ = cap.read()
        if not ret or frame_count <= 0:
            raise ValueError("Video contains no readable frames.")

        duration = (frame_count / fps) if fps > 0 else 0.0
        duration_formatted = format_duration(duration)
        video_format = file_path.suffix.upper().lstrip(".")

        # Video safety checks
        if width > MAX_VIDEO_DIMENSION or height > MAX_VIDEO_DIMENSION:
            raise ValueError(
                f"Video resolution ({width}x{height}) exceeds maximum allowed dimension ({MAX_VIDEO_DIMENSION}px)."
            )

        if duration > MAX_VIDEO_DURATION_SECONDS:
            raise ValueError(
                f"Video duration ({duration:.1f}s) exceeds maximum allowed duration ({MAX_VIDEO_DURATION_SECONDS:.0f}s)."
            )

        if fps > MAX_VIDEO_FPS:
            raise ValueError(
                f"Video framerate ({fps:.1f} FPS) exceeds maximum allowed framerate ({MAX_VIDEO_FPS:.0f} FPS)."
            )

        return MediaMetadata(
            width=width,
            height=height,
            format=video_format,
            file_size=file_size,
            file_size_human=file_size_human,
            fps=round(fps, 2),
            frame_count=frame_count,
            duration=round(duration, 2),
            duration_formatted=duration_formatted
        )
    finally:
        cap.release()

def extract_media_metadata(file_path: Path, media_type: str) -> MediaMetadata:
    """Dispatches metadata extraction based on media type."""
    if media_type == "image":
        return extract_image_metadata(file_path)
    elif media_type == "video":
        return extract_video_metadata(file_path)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")
