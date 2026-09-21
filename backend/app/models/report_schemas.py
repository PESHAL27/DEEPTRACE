"""
Pydantic schemas for DEEPTRACE Professional Forensic Case Reports (Stage 8G).
Defines structured data contracts for case identification, environment telemetry,
primary detector outputs, face localization, Grad-CAM attention, frequency-domain FFT,
metadata, video frame analysis, temporal consistency, cross-model comparison,
methodology, limitations, and evidence integrity.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class CaseIdentification(BaseModel):
    """Core administrative and evidentiary identification parameters."""
    case_id: str
    report_id: str
    report_version: str = "1.0"
    analysis_id: str
    generated_at: str
    filename: str
    media_type: str  # 'image' or 'video'
    file_size_bytes: int
    file_size_human: str
    sha256_hash: str
    analysis_status: str  # 'completed', 'no_face_detected', etc.


class AnalysisEnvironment(BaseModel):
    """Operational telemetry of the forensic runtime environment (safe identifiers only)."""
    primary_detector_name: str = "Xception Deepfake Detector"
    primary_detector_architecture: str = "Xception (DeepfakeBench)"
    primary_detector_checkpoint: str = "xception_best.pth"
    primary_detector_version: str = "DeepfakeBench FF++ c23 Pretrained"
    comparison_detector_name: str = "MesoNet (Meso4)"
    comparison_detector_checkpoint: str = "meso4_best.pth"
    face_detector: str = "OpenCV YuNet (libfacedetection)"
    input_resolution: str = "256x256 RGB"
    inference_device: str = "CPU (Host Processor)"
    normalization_mean: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    normalization_std: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    face_crop_padding: str = "15% expanded bounding box"
    video_sampling_interval: Optional[int] = 15
    video_max_frames: Optional[int] = 30


class PrimaryDetectorFindings(BaseModel):
    """Official primary model classification outputs and confidence metrics."""
    detector_name: str = "Xception Deepfake Detector"
    verdict: str  # 'LIKELY MANIPULATED', 'REAL', 'NO FACE DETECTED', 'INCONCLUSIVE'
    model_probability: Optional[float] = None
    manipulated_class_softmax_probability: Optional[float] = None
    raw_probability: Optional[float] = None
    confidence_score: float = 0.0
    classification_threshold: float = 0.50
    assessment: str = "Automated neural network classification"
    faces_detected: int = 0
    classification_note: str = (
        "Primary model output represents algorithmic pattern classification from the "
        "DeepfakeBench Xception convolutional neural network. The probability is the uncalibrated "
        "manipulated-class softmax probability output, indicating learned statistical correlation "
        "with training artifacts, and is not deterministic proof of media provenance."
    )


class FaceDetectionItem(BaseModel):
    """Individual localized facial subject."""
    face_index: int
    bounding_box: Dict[str, Any]  # {x, y, width, height}
    prediction: Optional[str] = None
    probability: Optional[float] = None
    is_primary: bool = False


class FaceDetectionEvidence(BaseModel):
    """Facial localization findings and individual subject bounding telemetry."""
    status: str
    faces_detected: int
    faces: List[FaceDetectionItem] = Field(default_factory=list)
    primary_face_index: Optional[int] = None
    detection_limitations: str = (
        "Facial localization operates via OpenCV YuNet. False negatives can occur under extreme "
        "angles, low illumination, heavy motion blur, or partial facial occlusions."
    )


class GradCAMEvidence(BaseModel):
    """Grad-CAM model visual attribution and attention feature maps."""
    available: bool = False
    target_layer: str = "conv4 (DeepfakeBench Xception block)"
    face_crop_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    interpretation_disclaimer: str = (
        "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes regions that contributed "
        "most strongly to the neural network's activation logits. It indicates model attention, "
        "not ground-truth forensic segmentation or verified tamper boundaries."
    )
    video_frame_gradcam: Optional[List[Dict[str, Any]]] = None


class FrequencyEvidence(BaseModel):
    """2D Fast Fourier Transform frequency-domain spectrum analysis."""
    available: bool = False
    analysis_region: str = "face_region"  # 'face_region' or 'full_image'
    spectrum_url: Optional[str] = None
    high_frequency_ratio: Optional[float] = None
    high_frequency_percentage: Optional[float] = None
    low_frequency_energy: Optional[float] = None
    mid_frequency_energy: Optional[float] = None
    high_frequency_energy: Optional[float] = None
    spectral_centroid: Optional[float] = None
    total_energy: Optional[float] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    interpretation_disclaimer: str = (
        "Frequency-domain characteristics can be influenced by lossy compression (JPEG/H.264), "
        "bilinear/bicubic resizing, sharpening filters, and sensor characteristics. High-frequency "
        "energy ratios provide supporting physical evidence and are not uniquely specific to synthetic media."
    )


class MetadataEvidence(BaseModel):
    """Container and image EXIF metadata inspection."""
    available: bool = False
    metadata_present: bool = False
    fields_count: int = 0
    format: Optional[str] = None
    dimensions: Optional[str] = None
    color_mode: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    editing_software_detected: bool = False
    creation_timestamp: Optional[str] = None
    modification_timestamp: Optional[str] = None
    gps_present: bool = False
    video_container: Optional[str] = None
    video_fps: Optional[float] = None
    video_frames: Optional[int] = None
    video_duration: Optional[str] = None
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    interpretation_disclaimer: str = (
        "Metadata presence reflects file container header information. Stripped or missing EXIF metadata "
        "is standard for media transmitted across web platforms and social networks, and does not, "
        "by itself, indicate malicious tampering."
    )


class VideoFrameSummary(BaseModel):
    """Summary of video temporal sampling and frame-level evaluations."""
    duration_seconds: float = 0.0
    duration_formatted: str = "00:00"
    fps: float = 30.0
    total_video_frames: int = 0
    sampled_frames_count: int = 0
    valid_face_frames_count: int = 0
    suspicious_frames_count: int = 0
    aggregation_method: str = "Top-20% suspicious quantile aggregation"
    sample_interval: int = 15
    max_analyzed_frames: int = 30
    frame_results: List[Dict[str, Any]] = Field(default_factory=list)


class TemporalEvidence(BaseModel):
    """Advanced sequence temporal consistency findings (Stage 8F)."""
    status: str = "not_available"
    face_presence_percentage: float = 0.0
    longest_face_present_streak: int = 0
    longest_face_missing_streak: int = 0
    probability_mean: float = 0.0
    probability_median: float = 0.0
    probability_std: float = 0.0
    mean_absolute_delta: float = 0.0
    max_absolute_delta: float = 0.0
    prediction_transitions_count: int = 0
    mean_center_displacement: Optional[float] = None
    max_center_displacement: Optional[float] = None
    max_area_delta: Optional[float] = None
    temporal_events_count: int = 0
    temporal_events: List[Dict[str, Any]] = Field(default_factory=list)
    model_agreement_rate: Optional[float] = None
    cohen_kappa: Optional[float] = None
    pearson_correlation: Optional[float] = None
    interpretation_disclaimer: str = (
        "Temporal analysis identifies inter-frame fluctuations in detector probabilities and spatial "
        "tracking. These signals can arise from video compression, camera motion, lighting variation, "
        "occlusion, or face detector jitter, and do not constitute unilateral proof of synthetic manipulation."
    )


class CrossModelEvidence(BaseModel):
    """Independent multi-model comparative validation (Stage 8E)."""
    evaluated: bool = False
    primary_model_name: str = "Xception (DeepfakeBench)"
    primary_prediction: Optional[str] = None
    primary_probability: Optional[float] = None
    comparison_model_name: str = "MesoNet-4 (Afchar et al.)"
    comparison_prediction: Optional[str] = None
    comparison_probability: Optional[float] = None
    agreement_status: str = "not_evaluated"  # 'aligned', 'divergent', 'not_evaluated'
    agreement_rate: Optional[float] = None
    cohen_kappa: Optional[float] = None
    pearson_correlation: Optional[float] = None
    governance_note: str = (
        "DEEPTRACE maintains independent model evaluation. Models are not ranked, averaged, or ensembled. "
        "The primary Xception verdict remains sovereign; MesoNet-4 serves as an orthogonal comparison."
    )


class EvidenceSummaryCategory(BaseModel):
    """
    Independent factual evidence domain status entry.
    Contains purely factual availability and analytical findings without
    weighted severity scores, manual risk rankings, or synthetic combined ratings.
    """
    category: str
    status: str  # Factual status: 'AVAILABLE', 'UNAVAILABLE', 'PRESENT', 'ABSENT', 'COMPLETED', etc.
    headline: str
    details: str
    supporting_artifacts: List[str] = Field(default_factory=list)


class DetailedForensicReport(BaseModel):
    """
    Comprehensive, reproducible professional forensic case report for DEEPTRACE Media Forensics.
    Satisfies both backward compatibility with ForensicReportResponse and complete Stage 8G specifications.
    Contains NO overall weighted forensic score or synthetic evidence ratings.
    """
    # Backward compatible fields
    id: str
    caseId: str
    filename: str
    mediaType: str
    result: str
    confidence: float
    created: str
    hashSha256: str
    examiner: str = "DEEPTRACE Automated Analysis Engine (Xception-DeepfakeBench)"
    summary: str
    methodology: str

    # Professional Stage 8G Structured Sections
    case_identification: CaseIdentification
    environment: AnalysisEnvironment
    primary_detector: PrimaryDetectorFindings
    face_detection: FaceDetectionEvidence
    gradcam_evidence: GradCAMEvidence
    frequency_evidence: FrequencyEvidence
    metadata_evidence: MetadataEvidence
    video_evidence: Optional[VideoFrameSummary] = None
    temporal_evidence: Optional[TemporalEvidence] = None
    cross_model_comparison: Optional[CrossModelEvidence] = None
    evidence_summary_categories: List[EvidenceSummaryCategory] = Field(default_factory=list)
    executive_summary: str
    methodology_details: List[Dict[str, str]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence_integrity_reference: Dict[str, str] = Field(default_factory=dict)
    automated_analysis_statement: str = (
        "This report summarizes automated algorithmic media analysis performed by DEEPTRACE "
        "and does not constitute human examiner certification or legal proof of media provenance."
    )
