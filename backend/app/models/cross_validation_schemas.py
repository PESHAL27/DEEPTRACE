"""
Pydantic schemas for DEEPTRACE Stage 8E — Multi-Model Cross Validation.
Defines normalized schemas for model registry entries, detector outputs,
image/video cross-validation, and comparative multi-model benchmarking.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class ModelRegistryEntry(BaseModel):
    """Metadata descriptor for a supported deepfake detector in the registry."""
    model_id: str
    display_name: str
    architecture: str
    framework: str
    task: str = "binary deepfake classification"
    role: str  # "primary" | "comparison"
    checkpoint: str
    checkpoint_size_kb: Optional[float] = None
    input_size: str = "256x256"
    preprocessing_mean: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    preprocessing_std: List[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    classes: List[str] = Field(default_factory=lambda: ["authentic", "manipulated"])
    default_threshold: float = 0.50
    license: str = "CC BY-NC 4.0"
    availability_status: str = "available"  # "available" | "unavailable"
    availability_notes: Optional[str] = None
    source_url: Optional[str] = None
    paper_citation: Optional[str] = None


class NormalizedDetectorPrediction(BaseModel):
    """
    Standardized detector prediction output.
    All registered detectors normalize their outputs to this deterministic structure.
    Positive class = 'manipulated', Negative class = 'authentic'.
    """
    model_id: str
    display_name: str
    role: str  # "primary" | "comparison"
    prediction: Optional[str] = None  # "authentic" | "manipulated" | "no_face_detected"
    manipulated_probability: Optional[float] = None
    authentic_probability: Optional[float] = None
    threshold: float = 0.50
    status: str = "success"  # "success" | "unavailable" | "error" | "no_face_detected"
    error_message: Optional[str] = None
    inference_time_ms: float = 0.0


class ImageCrossValidationResponse(BaseModel):
    """Multi-model cross-validation result for an uploaded image."""
    media_type: str = "image"
    filename: str
    faces_detected: int
    primary_model: NormalizedDetectorPrediction
    comparison_models: List[NormalizedDetectorPrediction] = Field(default_factory=list)
    agreement: bool
    disagreement: bool
    agreement_status: str  # "agreement" | "disagreement" | "comparison_unavailable" | "no_face_detected"
    summary_message: str
    disagreement_analysis: Optional[str] = None


class VideoFrameComparisonItem(BaseModel):
    """Aligned frame comparison result across primary and comparison detectors."""
    frame_number: int
    timestamp: float
    face_detected: bool
    primary_model: NormalizedDetectorPrediction
    comparison_models: List[NormalizedDetectorPrediction] = Field(default_factory=list)
    agreement: bool
    disagreement: bool


class VideoCrossValidationResponse(BaseModel):
    """Multi-model cross-validation result for an uploaded video."""
    media_type: str = "video"
    filename: str
    duration: float
    fps: float
    frames_analyzed: int
    compared_frames: int
    agreeing_frames: int
    disagreeing_frames: int
    agreement_percentage: float
    primary_manipulated_frames: int
    comparison_manipulated_frames: int
    primary_model_summary: NormalizedDetectorPrediction
    comparison_model_summaries: List[NormalizedDetectorPrediction] = Field(default_factory=list)
    agreement: bool
    disagreement: bool
    frames: List[VideoFrameComparisonItem] = Field(default_factory=list)


class ModelAgreementStatistics(BaseModel):
    """Descriptive agreement and correlation statistics between detectors."""
    total_compared: int
    agreeing_count: int
    disagreeing_count: int
    agreement_rate: float
    disagreement_rate: float
    cohen_kappa: Optional[float] = None
    cohen_kappa_status: str = "available"  # "available" | "not_available"
    pearson_correlation: Optional[float] = None
    spearman_correlation: Optional[float] = None
    correlation_status: str = "available"  # "available" | "not_available"
    correlation_explanation: Optional[str] = None


class ComparativeModelMetricItem(BaseModel):
    """Evaluation metrics for an individual detector on an evaluation dataset."""
    model_id: str
    display_name: str
    role: str
    threshold: float
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None


class MultiModelBenchmarkComparison(BaseModel):
    """Side-by-side performance evaluation on the exact same manifest samples."""
    dataset_name: str
    sample_count: int
    evaluated_models: List[ComparativeModelMetricItem] = Field(default_factory=list)
    agreement_statistics: ModelAgreementStatistics
    identical_dataset_verified: bool = True
