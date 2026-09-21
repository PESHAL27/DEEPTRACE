from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, computed_field


class ManifestSample(BaseModel):
    sample_id: str
    file_path: str
    sha256: Optional[str] = None
    media_type: str = "image"  # "image" | "video"
    ground_truth: str  # "authentic" | "manipulated"
    source: Optional[str] = "unknown"
    dataset_name: Optional[str] = "unknown"
    split: Optional[str] = "test"  # "train" | "val" | "test"
    manipulation_type: Optional[str] = "none"
    resolution: Optional[str] = "unknown"
    compression_level: Optional[str] = "unknown"
    notes: Optional[str] = ""


class DatasetManifest(BaseModel):
    manifest_version: str = "1.0"
    dataset_name: str
    dataset_version: Optional[str] = "1.0"
    description: Optional[str] = ""
    created_at: Optional[str] = None
    samples: List[ManifestSample] = Field(default_factory=list)


class SampleEvaluationResult(BaseModel):
    sample_id: str
    filename: str
    file_path: str
    media_type: str = "image"
    ground_truth: str
    predicted_class: Optional[str] = None
    manipulated_probability: Optional[float] = None
    authentic_probability: Optional[float] = None
    correct: Optional[bool] = None
    face_count: int = 0
    inference_status: str = "completed"  # "completed" | "no_face_detected" | "error"
    error_message: Optional[str] = None
    inference_time_ms: float = 0.0
    resolution: Optional[str] = None
    source: Optional[str] = None
    split: Optional[str] = None
    manipulation_type: Optional[str] = None
    sha256: Optional[str] = None


class ConfusionMatrix(BaseModel):
    tp: int
    tn: int
    fp: int
    fn: int
    tp_rate: float
    tn_rate: float
    fp_rate: float
    fn_rate: float


class ThresholdSweepItem(BaseModel):
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    fpr: float
    fnr: float
    tp: int
    fp: int
    tn: int
    fn: int


class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    roc_auc_status: str = "available"  # "available" | "not_available"
    roc_auc_explanation: Optional[str] = None


class ErrorAnalysisItem(BaseModel):
    sample_id: str
    filename: str
    error_type: str  # "false_positive" | "false_negative"
    ground_truth: str
    predicted_class: str
    manipulated_probability: float
    face_count: int
    resolution: Optional[str] = None
    sha256: Optional[str] = None
    source: Optional[str] = None
    dataset_name: Optional[str] = None


class SubgroupMetricItem(BaseModel):
    subgroup_type: str  # "resolution" | "face_count" | "manipulation_type"
    category: str
    sample_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float


class BenchmarkRunSummary(BaseModel):
    benchmark_run_id: str
    timestamp: str
    model_name: str = "Xception"
    framework: str = "DeepfakeBench"
    checkpoint: str = "xception_best.pth"
    threshold: float = 0.5
    dataset_name: str
    sample_count: int
    image_count: int
    video_count: int
    authentic_count: int
    manipulated_count: int
    no_face_count: int
    error_count: int
    metrics: EvaluationMetrics
    confusion_matrix: ConfusionMatrix
    total_runtime_seconds: float
    avg_inference_time_ms: float
    warnings: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy

    @computed_field
    @property
    def f1_score(self) -> float:
        return self.metrics.f1_score

    @computed_field
    @property
    def roc_auc(self) -> Optional[float]:
        return self.metrics.roc_auc


class BenchmarkRunDetail(BenchmarkRunSummary):
    samples: List[SampleEvaluationResult] = Field(default_factory=list)
    threshold_sweep: List[ThresholdSweepItem] = Field(default_factory=list)
    errors: List[ErrorAnalysisItem] = Field(default_factory=list)
    subgroup_metrics: List[SubgroupMetricItem] = Field(default_factory=list)


class BenchmarkRunRequest(BaseModel):
    manifest_path: Optional[str] = None
    manifest_content: Optional[DatasetManifest] = None
    threshold: float = 0.5
    max_samples: Optional[int] = None
