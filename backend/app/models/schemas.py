from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from .temporal_schemas import TemporalAnalysisResult

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "DEEPTRACE backend"
    version: str = "1.0.0"

class ModelReadinessInfo(BaseModel):
    xception_primary: str
    meso4_comparison: str
    yunet_face_detector: str
    smaug_ai_detector: Optional[str] = "ready"

class StorageReadinessInfo(BaseModel):
    uploads: str
    results: str

class StatusResponse(BaseModel):
    status: str
    service: str = "DEEPTRACE Media Forensics API"
    version: str = "1.0.0"
    device: str
    models: ModelReadinessInfo
    storage: StorageReadinessInfo

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

class PrimaryFaceResult(BaseModel):
    face_id: str
    face_index: int
    prediction: str
    probability: float
    real_probability: float
    fake_probability: float
    raw_logits: Optional[List[float]] = None
    bounding_box: FaceBoundingBox
    area: int
    relative_area: float
    margin_note: Optional[str] = None
    is_low_margin: Optional[bool] = False
    task_scope_note: Optional[str] = None

class AllFacesResult(BaseModel):
    face_count: int
    prediction: str
    probability: float
    mean_fake_probability: float
    mean_real_probability: float
    aggregation_rule: str
    margin_note: Optional[str] = None
    is_low_margin: Optional[bool] = False
    limitation_disclosure: Optional[str] = None

class FaceAnalysisItem(BaseModel):
    face_id: Optional[str] = None
    face_index: int
    bounding_box: FaceBoundingBox
    bbox: Optional[FaceBoundingBox] = None
    area: Optional[int] = None
    relative_area: Optional[float] = None
    is_primary: Optional[bool] = False
    prediction: str
    probability: float
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    raw_logits: Optional[List[float]] = None
    gradcam_available: Optional[bool] = False
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    original_face_crop_url: Optional[str] = None

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
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    raw_logits: Optional[List[float]] = None
    thumbnail_url: Optional[str] = None
    gradcam_available: Optional[bool] = False
    gradcam_heatmap_url: Optional[str] = None
    gradcam_overlay_url: Optional[str] = None
    original_face_crop_url: Optional[str] = None
    frequency_analysis: Optional[FrequencyAnalysisResult] = None
    frequency_spectrum_url: Optional[str] = None

class AIGenerationResult(BaseModel):
    model_name: str = "Smaug AI Image Detector"
    model_id: str = "abacusai/Smaug-AI-Image-Detector"
    score: Optional[float] = Field(default=None, description="Raw model probability score [0.0 - 1.0] that image is AI-generated")
    predicted_label: Optional[str] = Field(default=None, description="Classification verdict: likely_ai_generated, likely_camera_captured, or inconclusive")
    assessment: str = Field(default="Analysis unavailable", description="Human-readable assessment string")
    confidence: Optional[float] = Field(default=None, description="Confidence in predicted label [0.0 - 1.0]")
    threshold: float = Field(default=0.50, description="Decision threshold applied to the raw score")
    crops_evaluated: int = Field(default=1, description="Number of crop views evaluated (1 or 3)")
    device: Optional[str] = Field(default="cpu", description="Execution device (cpu or cuda)")
    status: str = Field(default="completed", description="Execution status: completed, error, or unavailable")
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None

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
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    assessment: Optional[str] = None
    faces: Optional[List[FaceAnalysisItem]] = Field(default_factory=list)
    selected_face_id: Optional[str] = None
    analysis_mode: Optional[str] = "primary_face"
    primary_face_result: Optional[PrimaryFaceResult] = None
    all_faces_result: Optional[AllFacesResult] = None
    is_low_margin: Optional[bool] = False
    margin_note: Optional[str] = None
    task_scope_note: Optional[str] = None
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
    ai_generation: Optional[AIGenerationResult] = None
    face_manipulation: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    media_type: str  # 'image' or 'video'
    media_url: Optional[str] = None
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
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    assessment: Optional[str] = None
    faces: Optional[List[FaceAnalysisItem]] = Field(default_factory=list)
    selected_face_id: Optional[str] = None
    analysis_mode: Optional[str] = "primary_face"
    primary_face_result: Optional[PrimaryFaceResult] = None
    all_faces_result: Optional[AllFacesResult] = None
    is_low_margin: Optional[bool] = False
    margin_note: Optional[str] = None
    task_scope_note: Optional[str] = None
    frame_results: Optional[List[VideoFrameAnalysisItem]] = None
    ai_generation: Optional[AIGenerationResult] = None
    face_manipulation: Optional[Dict[str, Any]] = None
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

