from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import datetime
from PIL import Image, ExifTags
import cv2

from ..models.schemas import MetadataAnalysisResult, MetadataFinding
from ..utils.logger import logger

# Recognized editing software signatures for heuristic detection
EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "canva", "premiere",
    "after effects", "final cut", "capcut", "snapseed", "vsco",
    "pixlr", "corel", "faceapp", "remini", "paint.net", "affinity"
]


def _format_timestamp(ts: float) -> str:
    """Formats a unix timestamp into an ISO-like readable string."""
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def analyze_image_metadata(file_path: Path) -> MetadataAnalysisResult:
    """
    Performs forensic inspection on an uploaded still image file.
    Extracts EXIF attributes, camera parameters, software traces, and timestamp metadata.
    """
    findings: List[MetadataFinding] = []
    raw_metadata: Dict[str, Any] = {}
    
    file_stat = file_path.stat()
    file_size_bytes = file_stat.st_size
    file_mtime_str = _format_timestamp(file_stat.st_mtime)
    file_ctime_str = _format_timestamp(file_stat.st_ctime)

    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    creation_timestamp: Optional[str] = None
    gps_present: bool = False
    editing_detected: bool = False
    metadata_fields_count: int = 0
    metadata_present: bool = False

    try:
        with Image.open(file_path) as img:
            img_format = img.format or file_path.suffix.upper().lstrip(".")
            color_mode = img.mode
            width, height = img.size

            raw_metadata["format"] = img_format
            raw_metadata["mode"] = color_mode
            raw_metadata["dimensions"] = f"{width}x{height}"
            raw_metadata["file_size_bytes"] = file_size_bytes

            # Attempt EXIF extraction
            exif_data = img.getexif()
            if exif_data and len(exif_data) > 0:
                metadata_present = True
                metadata_fields_count = len(exif_data)
                
                # Decode numeric EXIF tags to human-readable names
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Clean byte / tuple values to serializable representations
                    if isinstance(value, bytes):
                        clean_val = value.decode("utf-8", errors="ignore").strip("\x00")
                    elif isinstance(value, tuple):
                        clean_val = str(value)
                    else:
                        clean_val = str(value).strip("\x00")

                    raw_metadata[tag_name] = clean_val

                    # Extract camera details
                    if tag_name == "Make" and not camera_make:
                        camera_make = clean_val.strip()
                    elif tag_name == "Model" and not camera_model:
                        camera_model = clean_val.strip()
                    elif tag_name == "Software" and not software:
                        software = clean_val.strip()
                    elif tag_name in ("DateTimeOriginal", "DateTime", "DateTimeDigitized") and not creation_timestamp:
                        creation_timestamp = clean_val.strip()
                    elif tag_name == "GPSInfo":
                        gps_present = True

                # Check for IFD extensions (e.g. ExifIFD for detailed camera timestamps)
                try:
                    for ifd_id in ExifTags.IFD:
                        try:
                            ifd_data = exif_data.get_ifd(ifd_id)
                            if ifd_data:
                                for sub_tag_id, sub_val in ifd_data.items():
                                    sub_tag_name = ExifTags.TAGS.get(sub_tag_id, str(sub_tag_id))
                                    clean_sub = str(sub_val).strip("\x00")
                                    raw_metadata[f"{ifd_id.name}_{sub_tag_name}"] = clean_sub

                                    if sub_tag_name == "Software" and not software:
                                        software = clean_sub
                                    if sub_tag_name in ("DateTimeOriginal", "DateTimeDigitized") and not creation_timestamp:
                                        creation_timestamp = clean_sub
                        except Exception:
                            pass
                except Exception:
                    pass

    except Exception as e:
        logger.warning(f"Error during image metadata extraction for {file_path.name}: {e}")
        return MetadataAnalysisResult(
            available=False,
            metadata_present=False,
            findings=[
                MetadataFinding(
                    type="file_access_error",
                    message=f"Unable to read image header metadata: {e}",
                    severity="warning"
                )
            ]
        )

    # 1. Evaluate metadata presence
    if metadata_present:
        findings.append(
            MetadataFinding(
                type="exif_presence",
                message=f"Embedded EXIF metadata detected ({metadata_fields_count} fields identified).",
                severity="info"
            )
        )
    else:
        findings.append(
            MetadataFinding(
                type="exif_presence",
                message="EXIF metadata was not present in the uploaded file (standard for social media uploads or processed images).",
                severity="info"
            )
        )

    # 2. Camera hardware profile
    if camera_make or camera_model:
        device_str = " ".join(filter(None, [camera_make, camera_model]))
        findings.append(
            MetadataFinding(
                type="camera_profile",
                message=f"Camera hardware profile identified: {device_str}.",
                severity="info"
            )
        )

    # 3. Software signature & editing tool detection
    if software:
        lower_soft = software.lower()
        matched_editors = [kw for kw in EDITING_SOFTWARE_KEYWORDS if kw in lower_soft]
        if matched_editors:
            editing_detected = True
            findings.append(
                MetadataFinding(
                    type="editing_software",
                    message=f"Editing or post-processing software signature identified in metadata: '{software}'.",
                    severity="warning"
                )
            )
        else:
            findings.append(
                MetadataFinding(
                    type="software_trace",
                    message=f"Software tag recorded: '{software}'.",
                    severity="info"
                )
            )

    # 4. Timestamp analysis
    if creation_timestamp:
        findings.append(
            MetadataFinding(
                type="timestamp_analysis",
                message=f"Original capture timestamp recorded in metadata: {creation_timestamp}.",
                severity="info"
            )
        )

    # 5. Geolocation / GPS
    if gps_present:
        findings.append(
            MetadataFinding(
                type="gps_presence",
                message="Geographic positioning (GPS) metadata coordinates are embedded in evidence file.",
                severity="info"
            )
        )

    return MetadataAnalysisResult(
        available=True,
        metadata_present=metadata_present,
        metadata_fields_count=metadata_fields_count,
        camera_make=camera_make,
        camera_model=camera_model,
        software=software,
        editing_software_detected=editing_detected,
        gps_present=gps_present,
        creation_timestamp=creation_timestamp,
        modification_timestamp=file_mtime_str,
        raw_metadata=raw_metadata,
        findings=findings
    )


def analyze_video_metadata(file_path: Path) -> MetadataAnalysisResult:
    """
    Performs forensic inspection on an uploaded video container.
    Extracts stream properties, codec headers, dimensions, framerate, and container timestamps.
    """
    findings: List[MetadataFinding] = []
    raw_metadata: Dict[str, Any] = {}

    file_stat = file_path.stat()
    file_size_bytes = file_stat.st_size
    file_mtime_str = _format_timestamp(file_stat.st_mtime)

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return MetadataAnalysisResult(
            available=False,
            metadata_present=False,
            findings=[
                MetadataFinding(
                    type="stream_error",
                    message="Unable to parse video stream container.",
                    severity="warning"
                )
            ]
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Decode FourCC codec integer
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if not fourcc_str or any(ord(c) < 32 or ord(c) > 126 for c in fourcc_str):
            fourcc_str = "Unknown"

        duration = round(frame_count / fps, 2) if fps > 0 else 0.0

        raw_metadata["dimensions"] = f"{width}x{height}"
        raw_metadata["fps"] = round(fps, 2)
        raw_metadata["frame_count"] = frame_count
        raw_metadata["duration_seconds"] = duration
        raw_metadata["codec_fourcc"] = fourcc_str
        raw_metadata["container_format"] = file_path.suffix.upper().lstrip(".")
        raw_metadata["file_size_bytes"] = file_size_bytes

        findings.append(
            MetadataFinding(
                type="container_format",
                message=f"Container format: {file_path.suffix.upper().lstrip('.')}, Dimensions: {width}x{height}, Codec FourCC: '{fourcc_str}'.",
                severity="info"
            )
        )

        findings.append(
            MetadataFinding(
                type="temporal_stream",
                message=f"Temporal stream telemetry: {frame_count} frames @ {fps:.1f} FPS (Duration: {duration}s).",
                severity="info"
            )
        )

    finally:
        cap.release()

    return MetadataAnalysisResult(
        available=True,
        metadata_present=True,
        metadata_fields_count=len(raw_metadata),
        camera_make=None,
        camera_model=None,
        software=None,
        editing_software_detected=False,
        gps_present=False,
        creation_timestamp=None,
        modification_timestamp=file_mtime_str,
        raw_metadata=raw_metadata,
        findings=findings
    )


def analyze_metadata(file_path: Path, media_type: str) -> MetadataAnalysisResult:
    """
    Main forensic metadata entrypoint.
    Dispatches inspection based on media type.
    """
    if not file_path.exists():
        return MetadataAnalysisResult(
            available=False,
            metadata_present=False,
            findings=[
                MetadataFinding(
                    type="file_not_found",
                    message=f"Target evidence file {file_path.name} does not exist.",
                    severity="warning"
                )
            ]
        )

    if media_type == "image":
        return analyze_image_metadata(file_path)
    elif media_type == "video":
        return analyze_video_metadata(file_path)
    else:
        return MetadataAnalysisResult(
            available=False,
            metadata_present=False,
            findings=[
                MetadataFinding(
                    type="unsupported_media",
                    message=f"Unsupported media type for metadata extraction: {media_type}",
                    severity="warning"
                )
            ]
        )
