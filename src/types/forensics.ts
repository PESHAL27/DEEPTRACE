export type MediaType = 'image' | 'video';

export type PredictionVerdict = 'LIKELY MANIPULATED' | 'REAL' | 'INCONCLUSIVE' | 'NO FACE DETECTED';

export type RiskSeverity = 'High' | 'Medium' | 'Low';

export interface FaceBox {
  faceId?: string;
  faceIndex?: number;
  top: number;
  left: number;
  width: number;
  height: number;
  area?: number;
  relativeArea?: number;
  isPrimary?: boolean;
  label: string;
  prediction?: string;
  probability?: number;
  realProbability?: number;
  fakeProbability?: number;
  rawLogits?: number[];
  gradcamOverlayUrl?: string;
  gradcamHeatmapUrl?: string;
  originalFaceCropUrl?: string;
}

export interface SuspiciousFrame {
  id: string;
  frameNumber?: number;
  timestamp: string;
  seconds: number;
  confidence: number;
  fakeProbability?: number;
  realProbability?: number;
  prediction?: 'fake' | 'real';
  thumbnailUrl: string;
  status: string;
  anomalyType: string;
  boundingBox?: FaceBox;
  gradcamAvailable?: boolean;
  gradcamOverlayUrl?: string;
  gradcamHeatmapUrl?: string;
  originalFaceCropUrl?: string;
  frequencyAnalysis?: FrequencyAnalysis;
  frequencySpectrumUrl?: string;
}

export interface MetadataFinding {
  type: string;
  message: string;
  severity: 'info' | 'warning';
}

export interface MetadataAnalysis {
  available: boolean;
  metadataPresent: boolean;
  metadataFieldsCount: number;
  cameraMake?: string;
  cameraModel?: string;
  software?: string;
  editingSoftwareDetected: boolean;
  gpsPresent: boolean;
  creationTimestamp?: string;
  modificationTimestamp?: string;
  rawMetadata?: Record<string, any>;
  findings: MetadataFinding[];
}

export interface FrequencyFinding {
  type: string;
  message: string;
  severity: 'info' | 'warning';
}

export interface FrequencyAnalysis {
  available: boolean;
  analysisRegion: 'face_region' | 'full_image' | string;
  frequencySpectrumUrl?: string;
  meanMagnitude?: number;
  medianMagnitude?: number;
  lowFrequencyEnergy?: number;
  midFrequencyEnergy?: number;
  highFrequencyEnergy?: number;
  highFrequencyRatio?: number;
  spectralCentroid?: number;
  totalEnergy?: number;
  findings: FrequencyFinding[];
}

export interface ForensicIndicator {
  id: string;
  name: string;
  description: string;
  severity: RiskSeverity;
  detected: boolean;
  confidence?: number;
  iconName: 'boundary' | 'texture' | 'lighting' | 'temporal' | 'compression';
}

export interface TimelinePoint {
  timestamp: string;
  seconds: number;
  probability: number; // 0.0 - 1.0
  isSuspicious: boolean;
}

export interface Analysis {
  id: string;
  filename: string;
  fileSize: string;
  mediaType: MediaType;
  mediaUrl: string;
  createdAt: string;
  status: 'pending' | 'analyzing' | 'completed' | 'failed';
  prediction: PredictionVerdict;
  riskLevel?: string;
  confidenceScore: number; // e.g. 92.8
  model: string;
  duration?: string;
  framesAnalyzed?: number;
  facesDetected: number;
  suspiciousFrames: SuspiciousFrame[];
  indicators: ForensicIndicator[];
  timelineData?: TimelinePoint[];
  suspiciousRegion?: {
    start: string;
    end: string;
  };
  originalFrameUrl: string;
  heatmapUrl: string;
  gradcamAvailable?: boolean;
  gradcamOverlayUrl?: string;
  gradcamHeatmapUrl?: string;
  originalFaceCropUrl?: string;
  comparisonFrameUrl?: string;
  faceBoundingBox?: FaceBox;
  faceBoundingBoxes?: FaceBox[];
  selectedFaceId?: string;
  analysisMode?: 'primary_face' | 'all_faces';
  primaryFaceResult?: {
    faceId: string;
    faceIndex: number;
    prediction: string;
    probability: number;
    realProbability: number;
    fakeProbability: number;
    rawLogits?: number[];
    area: number;
    relativeArea: number;
    marginNote?: string;
    isLowMargin?: boolean;
    taskScopeNote?: string;
  };
  allFacesResult?: {
    faceCount: number;
    prediction: string;
    probability: number;
    meanFakeProbability: number;
    meanRealProbability: number;
    aggregationRule: string;
    marginNote?: string;
    isLowMargin?: boolean;
    limitationDisclosure?: string;
  };
  isLowMargin?: boolean;
  marginNote?: string;
  taskScopeNote?: string;
  rawProbability?: number;
  realProbability?: number;
  fakeProbability?: number;
  metadataAnalysis?: MetadataAnalysis;
  frequencyAnalysis?: FrequencyAnalysis;
  rawFile?: File;
  crossValidation?: ImageCrossValidationData | VideoCrossValidationData;
  temporalAnalysis?: TemporalAnalysisResultData;
}

export interface ModelInfo {
  id: string;
  name: string;
  category: string;
  purpose: string;
  status: 'Ready' | 'Not Loaded' | 'Planned';
  parameters: string;
  inputResolution: string;
  description: string;
  accuracyMetric?: string;
  architecture?: string;
  checkpoint?: string;
  device?: string;
  mean?: number[];
  std?: number[];
  numClasses?: number;
  role?: 'primary' | 'comparison' | string;
}

export interface ForensicReport {
  id: string;
  caseId: string;
  filename: string;
  mediaType: MediaType;
  result: PredictionVerdict;
  confidence: number;
  created: string;
  hashSha256: string;
  examiner: string;
  summary: string;
  methodology: string;
  detailedReport?: DetailedForensicReportData;
}

export interface AnalysisStepProgress {
  step: string;
  status: 'waiting' | 'running' | 'completed' | 'failed';
  message?: string;
}

// Stage 8D: Detection Reliability & Benchmarking Interfaces
export interface ConfusionMatrixData {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  tp_rate: number;
  tn_rate: number;
  fp_rate: number;
  fn_rate: number;
}

export interface EvaluationMetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1_score: number;
  false_positive_rate: number;
  false_negative_rate: number;
  roc_auc: number | null;
  pr_auc: number | null;
  roc_auc_status: 'available' | 'not_available';
  roc_auc_explanation?: string | null;
}

export interface ThresholdSweepData {
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  fpr: number;
  fnr: number;
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface ErrorAnalysisRecord {
  sample_id: string;
  filename: string;
  error_type: 'false_positive' | 'false_negative';
  ground_truth: string;
  predicted_class: string;
  manipulated_probability: number;
  face_count: number;
  resolution?: string | null;
  sha256?: string | null;
  source?: string | null;
  dataset_name?: string | null;
}

export interface SubgroupMetricRecord {
  category: string;
  subgroup_value: string;
  total_samples: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface BenchmarkSampleResult {
  sample_id: string;
  filename: string;
  file_path: string;
  media_type: string;
  ground_truth: string;
  predicted_class?: string | null;
  manipulated_probability?: number | null;
  authentic_probability?: number | null;
  correct?: boolean | null;
  face_count: number;
  inference_status: 'completed' | 'no_face_detected' | 'error';
  error_message?: string | null;
  inference_time_ms: number;
  resolution?: string | null;
  source?: string | null;
  split?: string | null;
  manipulation_type?: string | null;
  sha256?: string | null;
}

export interface BenchmarkRunSummaryItem {
  benchmark_run_id: string;
  timestamp: string;
  model_name: string;
  framework: string;
  checkpoint: string;
  threshold: number;
  dataset_name: string;
  sample_count: number;
  accuracy: number;
  f1_score: number;
  roc_auc: number | null;
  total_runtime_seconds: number;
}

export interface BenchmarkRunDetailData {
  benchmark_run_id: string;
  timestamp: string;
  model_name: string;
  framework: string;
  checkpoint: string;
  threshold: number;
  dataset_name: string;
  sample_count: number;
  image_count: number;
  video_count: number;
  authentic_count: number;
  manipulated_count: number;
  no_face_count: number;
  error_count: number;
  metrics: EvaluationMetricsData;
  confusion_matrix: ConfusionMatrixData;
  total_runtime_seconds: number;
  avg_inference_time_ms: number;
  warnings: string[];
  samples: BenchmarkSampleResult[];
  threshold_sweep: ThresholdSweepData[];
  errors: ErrorAnalysisRecord[];
  subgroup_metrics: SubgroupMetricRecord[];
}

export interface BenchmarkTemplateSample {
  sample_id: string;
  file_path: string;
  sha256: string;
  media_type: string;
  ground_truth: string;
  source?: string;
  dataset_name?: string;
  split?: string;
  manipulation_type?: string;
  resolution?: string;
  compression_level?: string;
  notes?: string;
}

export interface BenchmarkTemplate {
  manifest_version: string;
  dataset_name: string;
  dataset_version: string;
  description: string;
  created_at: string;
  samples: BenchmarkTemplateSample[];
}

export interface ModelRegistryItem {
  model_id: string;
  name: string;
  architecture: string;
  role: 'primary' | 'comparison' | string;
  source: string;
  checkpoint: string;
  status: 'ready' | 'unavailable' | 'error' | string;
  resolution: number;
  device: string;
  supported_modalities: string[];
  mean: number[];
  std: number[];
  parameters_count?: number | null;
  description?: string | null;
  citation?: string | null;
}

export interface NormalizedDetectorPredictionData {
  model_id: string;
  model_name: string;
  role: 'primary' | 'comparison' | string;
  architecture: string;
  prediction: 'authentic' | 'manipulated' | 'inconclusive' | 'error' | string;
  manipulated_probability: number;
  authentic_probability: number;
  threshold_used: number;
  confidence: number;
  inference_time_ms: number;
  status: 'completed' | 'no_face_detected' | 'error' | string;
  error_message?: string | null;
}

export interface ImageCrossValidationData {
  analysis_id: string;
  timestamp: string;
  media_type: 'image';
  filename: string;
  faces_detected: number;
  face_bounding_box?: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
  primary_detector: NormalizedDetectorPredictionData;
  comparison_detectors: NormalizedDetectorPredictionData[];
  agreement_status: 'unanimous_agreement' | 'disagreement_detected' | 'partial_agreement' | 'no_face' | 'error' | string;
  agreement_percentage: number;
  disagreement_detected: boolean;
  analysis_summary: string;
}

export interface VideoFrameComparisonItemData {
  frame_number: number;
  timestamp: number;
  face_detected: boolean;
  bounding_box?: {
    x: number;
    y: number;
    width: number;
    height: number;
  } | null;
  primary_prediction: NormalizedDetectorPredictionData;
  comparison_predictions: NormalizedDetectorPredictionData[];
  frame_agreement: boolean;
}

export interface ModelAgreementStatisticsData {
  pair: string;
  model_a: string;
  model_b: string;
  frames_compared: number;
  agreement_count: number;
  disagreement_count: number;
  agreement_rate: number;
  cohen_kappa?: number | null;
  pearson_correlation?: number | null;
  spearman_correlation?: number | null;
}

export interface VideoCrossValidationData {
  analysis_id: string;
  timestamp: string;
  media_type: 'video';
  filename: string;
  duration: number;
  fps: number;
  frames_analyzed: number;
  primary_video_prediction: string;
  primary_video_probability: number;
  comparison_video_predictions: Array<Record<string, any>>;
  agreement_status: string;
  video_agreement_rate: number;
  disagreement_detected: boolean;
  frame_comparisons: VideoFrameComparisonItemData[];
  agreement_statistics: ModelAgreementStatisticsData[];
  analysis_summary: string;
}

export interface ComparativeModelMetricItemData {
  model_id: string;
  display_name: string;
  role: string;
  threshold: number;
  sample_count: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1_score: number;
  false_positive_rate: number;
  false_negative_rate: number;
  roc_auc?: number | null;
  pr_auc?: number | null;
}

export interface MultiModelBenchmarkComparisonData {
  dataset_name: string;
  sample_count: number;
  evaluated_models: ComparativeModelMetricItemData[];
  agreement_statistics: ModelAgreementStatisticsData;
  identical_dataset_verified: boolean;
}

// Stage 8F: Advanced Temporal Video Analysis Interfaces
export interface TemporalProbabilityDeltaData {
  from_frame: number;
  to_frame: number;
  from_timestamp: number;
  to_timestamp: number;
  delta: number;
  absolute_delta: number;
}

export interface TemporalProbabilityVolatilityData {
  mean_probability: number;
  median_probability: number;
  min_probability: number;
  max_probability: number;
  std_probability: number;
  mean_absolute_delta: number;
  max_absolute_delta: number;
}

export interface TemporalPredictionTransitionData {
  frame_before: number;
  frame_after: number;
  timestamp: number;
  from_prediction: string;
  to_prediction: string;
  from_probability: number;
  to_probability: number;
  description: string;
}

export interface TemporalFacePresenceData {
  total_sampled_frames: number;
  frames_with_face: number;
  frames_without_face: number;
  face_presence_percentage: number;
  longest_consecutive_face_present: number;
  longest_consecutive_face_missing: number;
}

export interface TemporalBBoxTrackingData {
  mean_center_displacement: number;
  max_center_displacement: number;
  mean_face_area: number;
  min_face_area: number;
  max_face_area: number;
  max_area_delta: number;
  tracking_note: string;
}

export interface TemporalEventData {
  event_id: string;
  frame_number: number;
  timestamp: number;
  event_type: 'probability_jump' | 'prediction_transition' | 'face_presence_gap' | 'face_bbox_change' | 'face_scale_change' | 'detector_disagreement' | string;
  value: number;
  threshold?: number | null;
  severity: 'informational' | 'notable';
  description: string;
}

export interface TemporalModelAgreementData {
  frames_compared: number;
  agreeing_frames: number;
  disagreeing_frames: number;
  agreement_rate: number;
  cohen_kappa?: number | null;
  pearson_correlation?: number | null;
}

export interface TemporalFrameItemData {
  frame_number: number;
  timestamp: number;
  valid: boolean;
  xception_probability?: number | null;
  xception_prediction?: string | null;
  meso4_probability?: number | null;
  meso4_prediction?: string | null;
  face_count: number;
  face_bbox?: number[] | null;
  thumbnail_url?: string | null;
  gradcam_available: boolean;
  frequency_available: boolean;
}

export interface TemporalAnalysisResultData {
  status: 'available' | 'not_available' | string;
  reason?: string | null;
  analyzed_frames: number;
  valid_frames: number;
  face_presence: TemporalFacePresenceData;
  statistics?: TemporalProbabilityVolatilityData | null;
  deltas: TemporalProbabilityDeltaData[];
  prediction_transitions: TemporalPredictionTransitionData[];
  bbox_tracking?: TemporalBBoxTrackingData | null;
  temporal_events: TemporalEventData[];
  model_agreement?: TemporalModelAgreementData | null;
  frame_series: TemporalFrameItemData[];
  analysis_disclaimer: string;
}

// Stage 8G: Professional Forensic Case Report Interfaces
export interface CaseIdentificationData {
  case_id: string;
  report_id: string;
  report_version: string;
  analysis_id: string;
  generated_at: string;
  filename: string;
  media_type: string;
  file_size_bytes: number;
  file_size_human: string;
  sha256_hash: string;
  analysis_status: string;
}

export interface AnalysisEnvironmentData {
  primary_detector_name: string;
  primary_detector_architecture: string;
  primary_detector_checkpoint: string;
  primary_detector_version: string;
  comparison_detector_name: string;
  comparison_detector_checkpoint: string;
  face_detector: string;
  input_resolution: string;
  inference_device: string;
  normalization_mean: number[];
  normalization_std: number[];
  face_crop_padding: string;
  video_sampling_interval?: number;
  video_max_frames?: number;
}

export interface PrimaryDetectorFindingsData {
  detector_name: string;
  verdict: string;
  model_probability?: number | null;
  manipulated_class_softmax_probability?: number | null;
  raw_probability?: number | null;
  confidence_score: number;
  classification_threshold: number;
  assessment: string;
  faces_detected: number;
  classification_note: string;
}

export interface FaceDetectionItemData {
  face_index: number;
  bounding_box: { x: number; y: number; width: number; height: number };
  prediction?: string | null;
  probability?: number | null;
  is_primary: boolean;
}

export interface FaceDetectionEvidenceData {
  status: string;
  faces_detected: number;
  faces: FaceDetectionItemData[];
  primary_face_index?: number | null;
  detection_limitations: string;
}

export interface GradCAMEvidenceData {
  available: boolean;
  target_layer: string;
  face_crop_url?: string | null;
  heatmap_url?: string | null;
  overlay_url?: string | null;
  interpretation_disclaimer: string;
  video_frame_gradcam?: Array<{
    frame_number: number;
    timestamp: number;
    probability?: number;
    gradcam_overlay_url?: string;
    heatmap_url?: string;
    face_crop_url?: string;
  }> | null;
}

export interface FrequencyEvidenceData {
  available: boolean;
  analysis_region: string;
  spectrum_url?: string | null;
  high_frequency_ratio?: number | null;
  high_frequency_percentage?: number | null;
  low_frequency_energy?: number | null;
  mid_frequency_energy?: number | null;
  high_frequency_energy?: number | null;
  spectral_centroid?: number | null;
  total_energy?: number | null;
  findings: Array<Record<string, any>>;
  interpretation_disclaimer: string;
}

export interface MetadataEvidenceData {
  available: boolean;
  metadata_present: boolean;
  fields_count: number;
  format?: string | null;
  dimensions?: string | null;
  color_mode?: string | null;
  camera_make?: string | null;
  camera_model?: string | null;
  software?: string | null;
  editing_software_detected: boolean;
  creation_timestamp?: string | null;
  modification_timestamp?: string | null;
  gps_present: boolean;
  video_container?: string | null;
  video_fps?: number | null;
  video_frames?: number | null;
  video_duration?: string | null;
  findings: Array<Record<string, any>>;
  interpretation_disclaimer: string;
}

export interface VideoFrameSummaryData {
  duration_seconds: number;
  duration_formatted: string;
  fps: number;
  total_video_frames: number;
  sampled_frames_count: number;
  valid_face_frames_count: number;
  suspicious_frames_count: number;
  aggregation_method: string;
  sample_interval: number;
  max_analyzed_frames: number;
  frame_results: Array<Record<string, any>>;
}

export interface TemporalEvidenceData {
  status: string;
  face_presence_percentage: number;
  longest_face_present_streak: number;
  longest_face_missing_streak: number;
  probability_mean: number;
  probability_median: number;
  probability_std: number;
  mean_absolute_delta: number;
  max_absolute_delta: number;
  prediction_transitions_count: number;
  mean_center_displacement?: number | null;
  max_center_displacement?: number | null;
  max_area_delta?: number | null;
  temporal_events_count: number;
  temporal_events: Array<Record<string, any>>;
  model_agreement_rate?: number | null;
  cohen_kappa?: number | null;
  pearson_correlation?: number | null;
  interpretation_disclaimer: string;
}

export interface CrossModelEvidenceData {
  evaluated: boolean;
  primary_model_name: string;
  primary_prediction?: string | null;
  primary_probability?: number | null;
  comparison_model_name: string;
  comparison_prediction?: string | null;
  comparison_probability?: number | null;
  agreement_status: string;
  agreement_rate?: number | null;
  cohen_kappa?: number | null;
  pearson_correlation?: number | null;
  governance_note: string;
}

export interface EvidenceSummaryCategoryData {
  category: string;
  status: string;
  headline: string;
  details: string;
  supporting_artifacts: string[];
}

export interface DetailedForensicReportData {
  id: string;
  caseId: string;
  filename: string;
  mediaType: string;
  result: string;
  confidence: number;
  created: string;
  hashSha256: string;
  examiner: string;
  summary: string;
  methodology: string;

  case_identification: CaseIdentificationData;
  environment: AnalysisEnvironmentData;
  primary_detector: PrimaryDetectorFindingsData;
  face_detection: FaceDetectionEvidenceData;
  gradcam_evidence: GradCAMEvidenceData;
  frequency_evidence: FrequencyEvidenceData;
  metadata_evidence: MetadataEvidenceData;
  video_evidence?: VideoFrameSummaryData | null;
  temporal_evidence?: TemporalEvidenceData | null;
  cross_model_comparison?: CrossModelEvidenceData | null;
  evidence_summary_categories: EvidenceSummaryCategoryData[];
  executive_summary: string;
  methodology_details: Array<{ step: string; description: string }>;
  limitations: string[];
  evidence_integrity_reference: Record<string, string>;
  automated_analysis_statement?: string;
}



