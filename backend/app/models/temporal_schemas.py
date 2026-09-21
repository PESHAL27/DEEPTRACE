from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TemporalFrameItem(BaseModel):
    """Data representation of an individual analyzed video frame in a temporal sequence."""
    frame_number: int
    timestamp: float
    valid: bool = False
    xception_probability: Optional[float] = None
    xception_prediction: Optional[str] = None
    meso4_probability: Optional[float] = None
    meso4_prediction: Optional[str] = None
    face_count: int = 0
    face_bbox: Optional[List[int]] = None  # [x, y, width, height]
    thumbnail_url: Optional[str] = None
    gradcam_available: bool = False
    frequency_available: bool = False


class TemporalProbabilityDelta(BaseModel):
    """Measured probability step between consecutive valid frames."""
    from_frame: int
    to_frame: int
    from_timestamp: float
    to_timestamp: float
    delta: float
    absolute_delta: float


class TemporalProbabilityVolatility(BaseModel):
    """Statistical dispersion and volatility metrics over valid frame sequence."""
    mean_probability: float
    median_probability: float
    min_probability: float
    max_probability: float
    std_probability: float
    mean_absolute_delta: float
    max_absolute_delta: float


class TemporalPredictionTransition(BaseModel):
    """Transition of categorical prediction between consecutive valid frames."""
    frame_before: int
    frame_after: int
    timestamp: float
    from_prediction: str
    to_prediction: str
    from_probability: float
    to_probability: float
    description: str = "Prediction transition detected."


class TemporalFacePresence(BaseModel):
    """Facial presence consistency and detection sequence metrics."""
    total_sampled_frames: int
    frames_with_face: int
    frames_without_face: int
    face_presence_percentage: float
    longest_consecutive_face_present: int
    longest_consecutive_face_missing: int


class TemporalBBoxTracking(BaseModel):
    """Spatial displacement and scale dynamics of detected facial bounding boxes."""
    mean_center_displacement: float
    max_center_displacement: float
    mean_face_area: float
    min_face_area: float
    max_face_area: float
    max_area_delta: float
    tracking_note: str = (
        "Spatial tracking operates on the primary representative face per frame. "
        "Independent identity tracking is not applied."
    )


class TemporalEvent(BaseModel):
    """A discrete, measurable temporal observation or threshold anomaly."""
    event_id: str
    frame_number: int
    timestamp: float
    event_type: str  # "probability_jump" | "prediction_transition" | "face_presence_gap" | "face_bbox_change" | "face_scale_change" | "detector_disagreement"
    value: float
    threshold: Optional[float] = None
    severity: str = "informational"  # "informational" | "notable"
    description: str


class TemporalModelAgreement(BaseModel):
    """Inter-model temporal alignment between Xception and Meso4 across shared frames."""
    frames_compared: int
    agreeing_frames: int
    disagreeing_frames: int
    agreement_rate: float
    cohen_kappa: Optional[float] = None
    pearson_correlation: Optional[float] = None


class TemporalAnalysisResult(BaseModel):
    """Comprehensive temporal video consistency analysis report."""
    status: str = "available"  # "available" | "not_available"
    reason: Optional[str] = None
    analyzed_frames: int = 0
    valid_frames: int = 0
    face_presence: TemporalFacePresence
    statistics: Optional[TemporalProbabilityVolatility] = None
    deltas: List[TemporalProbabilityDelta] = Field(default_factory=list)
    prediction_transitions: List[TemporalPredictionTransition] = Field(default_factory=list)
    bbox_tracking: Optional[TemporalBBoxTracking] = None
    temporal_events: List[TemporalEvent] = Field(default_factory=list)
    model_agreement: Optional[TemporalModelAgreement] = None
    frame_series: List[TemporalFrameItem] = Field(default_factory=list)
    analysis_disclaimer: str = (
        "Forensic Disclaimer: Temporal inconsistencies and fluctuations can arise from compression artifacts, "
        "camera motion, lighting variations, or detector limitations, and do not constitute unilateral proof of manipulation. "
        "Primary production verdict remains sovereign."
    )
