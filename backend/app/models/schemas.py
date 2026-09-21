from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from .temporal_schemas import TemporalAnalysisResult

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "DEEPTRACE backend"
    version: str = "1.0.0"

class MediaMetadata(BaseModel):
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    file_size: int = Field(description="File size in bytes")
    file_size_human: str = Field(description="Formatted human-readable size, e.g. 14.2 MB")
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    duration: Optional[float] = None
    duration_formatted: Optional[str] = None

class FaceBoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class FaceDetectionResult(BaseModel):
    status: str = Field(
        default="face_detector_not_loaded", 
        description="Explicit detector status: 'face_detector_not_loaded' or 'completed'"
    )
    faces: List[FaceBoundingBox] = Field(default_factory=list)
    count: int = 0

class ModelDetails(BaseModel):
    name: str = "Xception"
    source: str = "DeepfakeBench"

class FaceAnalysisItem(BaseModel):
    face_index: int
    bounding_box: FaceBoundingBox
    prediction: str
    probability: float
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None

class MetadataFinding(BaseModel):
    type: str
    message: str
    severity: str  # "info" | "warning"

class MetadataAnalysisResult(BaseModel):
    available: bool = True
    metadata_present: bool = False
    metadata_fields_count: int = 0
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    editing_software_detected: bool = False
    gps_present: bool = False
    creation_timestamp: Optional[str] = None
    modification_timestamp: Optional[str] = None
    raw_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    findings: List[MetadataFinding] = Field(default_factory=list)

class FrequencyFinding(BaseModel):
    type: str
    message: str
    severity: str  # "info" | "warning"

class FrequencyAnalysisResult(BaseModel):
    available: bool = True
    analysis_region: str = "face_region"  # "face_region" | "full_image"
    frequency_spectrum_url: Optional[str] = None
    mean_magnitude: Optional[float] = None
    median_magnitude: Optional[float] = None
    low_frequency_energy: Optional[float] = None
    mid_frequency_energy: Optional[float] = None
    high_frequency_energy: Optional[float] = None
    high_frequency_ratio: Optional[float] = None
    spectral_centroid: Optional[float] = None
    total_energy: Optional[float] = None
    findings: List[FrequencyFinding] = Field(default_factory=list)

class VideoFrameAnalysisItem(BaseModel):
    frame_number: int
    timestamp: float
    face_detected: bool
    bounding_box: Optional[FaceBoundingBox] = None
    prediction: Optional[str] = None
    probability: Optional[float] = None
    thumbnail_url: Optional[str] = None
    gradcam_available: Optional[bool] = False
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    original_face_crop_url: Optional[str] = None
    frequency_analysis: Optional[FrequencyAnalysisResult] = None
    frequency_spectrum_url: Optional[str] = None

class DetectionResult(BaseModel):
    status: str = Field(
        default="model_not_loaded", 
        description="Explicit model state: 'no_face_detected', 'completed', 'model_not_loaded'"
    )
    model: Optional[Union[ModelDetails, str]] = None
    prediction: Optional[str] = None
    probability: Optional[float] = None
    detection_probability: Optional[float] = None
    model_probability: Optional[float] = None
    video_probability: Optional[float] = None
    assessment: Optional[str] = None
    faces: Optional[List[FaceAnalysisItem]] = Field(default_factory=list)
    frame_results: Optional[List[VideoFrameAnalysisItem]] = None
    frames_analyzed: Optional[int] = None
    faces_detected: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    gradcam_available: Optional[bool] = False
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    original_face_crop_url: Optional[str] = None
    metadata_analysis: Optional[MetadataAnalysisResult] = None
    frequency_analysis: Optional[FrequencyAnalysisResult] = None
    temporal_analysis: Optional[TemporalAnalysisResult] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    media_type: str  # 'image' or 'video'
    file_size: int
    file_size_human: str
    status: str
    metadata: MediaMetadata
    faces_detected: Optional[int] = None
    face_detection: FaceDetectionResult
    detection: DetectionResult
    model: Optional[Union[ModelDetails, str]] = None
    prediction: Optional[str] = None
    probability: Optional[float] = None
    detection_probability: Optional[float] = None
    model_probability: Optional[float] = None
    video_probability: Optional[float] = None
    assessment: Optional[str] = None
    faces: Optional[List[FaceAnalysisItem]] = Field(default_factory=list)
    frame_results: Optional[List[VideoFrameAnalysisItem]] = None
    frames_analyzed: Optional[int] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    gradcam_available: Optional[bool] = False
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    original_face_crop_url: Optional[str] = None
    metadata_analysis: Optional[MetadataAnalysisResult] = None
    frequency_analysis: Optional[FrequencyAnalysisResult] = None
    temporal_analysis: Optional[TemporalAnalysisResult] = None
    created_at: str

class ErrorResponse(BaseModel):
    error: str
    detail: str
    code: int

class ModelInfoResponse(BaseModel):
    id: str
    name: str
    category: str
    purpose: str
    status: str
    parameters: str
    inputResolution: str
    description: str
    accuracyMetric: Optional[str] = None
    architecture: Optional[str] = None
    checkpoint: Optional[str] = None
    device: Optional[str] = None
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    numClasses: Optional[int] = None

class ForensicReportResponse(BaseModel):
    id: str
    caseId: str
    filename: str
    mediaType: str
    result: str
    confidence: float
    created: str
    hashSha256: str
    examiner: str
    summary: str
    methodology: str

