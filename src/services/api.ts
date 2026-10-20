import type {
  Analysis,
  ForensicReport,
  ModelInfo,
  BenchmarkRunSummaryItem,
  BenchmarkRunDetailData,
  BenchmarkTemplate,
  ModelRegistryItem,
  ImageCrossValidationData,
  VideoCrossValidationData,
  MultiModelBenchmarkComparisonData,
  DetailedForensicReportData
} from '../types/forensics';

// Configurable backend and media URLs for development and production deployments
export const USE_LIVE_BACKEND = true;
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string) || 'http://127.0.0.1:8000/api';

export const MEDIA_BASE_URL: string =
  (import.meta.env.VITE_MEDIA_BASE_URL as string) ??
  (API_BASE_URL.startsWith('http') ? API_BASE_URL.replace(/\/api\/?$/, '') : '');

export function resolveMediaUrl(url: string | undefined | null): string {
  if (!url) return '';
  if (url.startsWith('http://') || url.startsWith('https://') || url.startsWith('blob:') || url.startsWith('data:')) {
    return url;
  }
  return `${MEDIA_BASE_URL}${url.startsWith('/') ? '' : '/'}${url}`;
}

// In-memory store for session additions
let sessionAnalyses: Analysis[] = [];
let sessionReports: ForensicReport[] = [];

export interface UploadResponse {
  fileId: string;
  filename: string;
  fileSize: string;
  mediaType: 'image' | 'video';
  previewUrl: string;
  rawFile?: File;
}

function formatSecondsToMMSS(sec: number): string {
  const mins = Math.floor(sec / 60);
  const secs = Math.floor(sec % 60);
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function transformBackendAnalysis(data: any, fallbackPreviewUrl?: string): Analysis {
  const isModelLoaded = data.detection && data.detection.status !== 'model_not_loaded';
  const facesCount = data.faces_detected ?? (data.faces ? data.faces.length : 0);
  const isNoFace = data.status === 'no_face_detected' || facesCount === 0;

  const modelName = typeof data.model === 'object' && data.model !== null
    ? `${data.model.name} (${data.model.source})`
    : (data.model ?? 'Xception (DeepfakeBench)');

  const imgW = data.metadata?.width || 1;
  const imgH = data.metadata?.height || 1;
  const rawFaces = data.faces || [];

  const faceBoxes = rawFaces.map((f: any) => {
    const bbox = f.bounding_box || {};
    const fPred = f.prediction ? f.prediction.toUpperCase() : 'FACE';
    const fProb = typeof f.probability === 'number' ? (f.probability * 100).toFixed(1) : '0.0';
    const faceId = f.face_id || `face_${f.face_index}`;
    return {
      faceId: faceId,
      faceIndex: f.face_index,
      left: +((bbox.x / imgW) * 100).toFixed(2),
      top: +((bbox.y / imgH) * 100).toFixed(2),
      width: +((bbox.width / imgW) * 100).toFixed(2),
      height: +((bbox.height / imgH) * 100).toFixed(2),
      area: f.area,
      relativeArea: f.relative_area,
      isPrimary: f.is_primary,
      label: `Face #${f.face_index + 1}: ${fPred} (${fProb}%)`,
      prediction: f.prediction,
      probability: f.probability,
      realProbability: f.real_probability,
      fakeProbability: f.fake_probability,
      rawLogits: f.raw_logits,
      gradcamOverlayUrl: resolveMediaUrl(f.gradcam_overlay_url),
      gradcamHeatmapUrl: resolveMediaUrl(f.gradcam_heatmap_url),
      originalFaceCropUrl: resolveMediaUrl(f.original_face_crop_url)
    };
  });

  const effectiveProb = typeof data.video_probability === 'number'
    ? data.video_probability
    : (typeof data.probability === 'number' ? data.probability : 0);

  let predictionStr: any = 'INCONCLUSIVE';
  let riskLevel = 'Inconclusive';
  let confidenceScore = 0;

  if (isNoFace) {
    predictionStr = 'NO FACE DETECTED';
    riskLevel = 'No Face Detected';
    confidenceScore = 0;
  } else if (isModelLoaded) {
    const pred = (data.prediction || '').toLowerCase();
    if (pred === 'fake' || data.prediction === 'LIKELY MANIPULATED') {
      predictionStr = 'LIKELY MANIPULATED';
      confidenceScore = +(effectiveProb * 100).toFixed(1);
      riskLevel = `Model Probability: ${confidenceScore}%`;
    } else if (pred === 'real' || data.prediction === 'REAL') {
      predictionStr = 'REAL';
      confidenceScore = +(effectiveProb * 100).toFixed(1);
      riskLevel = `Model Probability: ${confidenceScore}%`;
    }
  }

  // Map video frame results to suspiciousFrames and timelineData
  const rawFrameResults: any[] = data.frame_results || [];

  // Explicit selection criterion for Suspicious Frames:
  // A frame is suspicious if and only if a face is detected AND the frame is classified as manipulated:
  // (prediction === 'fake' OR fake_probability >= 0.50).
  // Authentic/real frames must never be included in the suspicious frames panel.
  const suspiciousFrames = rawFrameResults
    .filter((f: any) => {
      if (!f.face_detected) return false;
      const fakeP = typeof f.fake_probability === 'number'
        ? f.fake_probability
        : (f.prediction === 'fake' ? f.probability : (typeof f.probability === 'number' ? +(1.0 - f.probability).toFixed(4) : 0));
      return f.prediction === 'fake' || fakeP >= 0.5;
    })
    .map((f: any) => {
      const fakeP = typeof f.fake_probability === 'number'
        ? f.fake_probability
        : (f.prediction === 'fake' ? f.probability : (typeof f.probability === 'number' ? +(1.0 - f.probability).toFixed(4) : 0));
      const realP = typeof f.real_probability === 'number'
        ? f.real_probability
        : +(1.0 - fakeP).toFixed(4);

      const bbox = f.bounding_box;
      const box = bbox ? {
        left: +((bbox.x / imgW) * 100).toFixed(2),
        top: +((bbox.y / imgH) * 100).toFixed(2),
        width: +((bbox.width / imgW) * 100).toFixed(2),
        height: +((bbox.height / imgH) * 100).toFixed(2),
        label: `Frame #${f.frame_number}: FAKE (${(fakeP * 100).toFixed(1)}%)`,
        prediction: 'fake',
        probability: fakeP
      } : undefined;

      return {
        id: `f-${f.frame_number}`,
        frameNumber: f.frame_number,
        timestamp: formatSecondsToMMSS(f.timestamp),
        seconds: f.timestamp,
        confidence: fakeP,
        fakeProbability: fakeP,
        realProbability: realP,
        prediction: 'fake' as const,
        thumbnailUrl: resolveMediaUrl(f.thumbnail_url || `/uploads/${data.analysis_id}_${data.filename}`),
        status: 'Manipulated Frame',
        anomalyType: 'Facial Boundary Discrepancy',
        boundingBox: box,
        gradcamAvailable: f.gradcam_available ?? false,
        gradcamOverlayUrl: resolveMediaUrl(f.gradcam_overlay_url),
        gradcamHeatmapUrl: resolveMediaUrl(f.gradcam_heatmap_url),
        originalFaceCropUrl: resolveMediaUrl(f.original_face_crop_url),
        frequencyAnalysis: f.frequency_analysis ? {
          available: f.frequency_analysis.available,
          analysisRegion: f.frequency_analysis.analysis_region,
          frequencySpectrumUrl: resolveMediaUrl(f.frequency_analysis.frequency_spectrum_url),
          meanMagnitude: f.frequency_analysis.mean_magnitude,
          medianMagnitude: f.frequency_analysis.median_magnitude,
          lowFrequencyEnergy: f.frequency_analysis.low_frequency_energy,
          midFrequencyEnergy: f.frequency_analysis.mid_frequency_energy,
          highFrequencyEnergy: f.frequency_analysis.high_frequency_energy,
          highFrequencyRatio: f.frequency_analysis.high_frequency_ratio,
          spectralCentroid: f.frequency_analysis.spectral_centroid,
          totalEnergy: f.frequency_analysis.total_energy,
          findings: f.frequency_analysis.findings || []
        } : undefined,
        frequencySpectrumUrl: resolveMediaUrl(f.frequency_spectrum_url || f.frequency_analysis?.frequency_spectrum_url)
      };
    });

  const timelineData = rawFrameResults.map((f: any) => {
    const fakeP = !f.face_detected
      ? 0.0
      : (typeof f.fake_probability === 'number'
          ? f.fake_probability
          : (f.prediction === 'fake' ? (f.probability ?? 0.0) : Math.max(0, +(1.0 - (f.probability ?? 1.0)).toFixed(4))));
    const isSuspicious = f.face_detected && (f.prediction === 'fake' || fakeP >= 0.5);

    return {
      timestamp: formatSecondsToMMSS(f.timestamp),
      seconds: f.timestamp,
      probability: fakeP,
      isSuspicious: isSuspicious
    };
  });

  const rawMediaUrl = data.media_url || `/uploads/${data.analysis_id}_${data.filename}`;
  const backendMediaUrl = resolveMediaUrl(rawMediaUrl);
  const mediaUrl = backendMediaUrl || fallbackPreviewUrl || '';

  const firstFaceFrame = rawFrameResults.find((f: any) => f.face_detected && f.bounding_box);
  const videoPrimaryBox = firstFaceFrame && firstFaceFrame.bounding_box ? {
    left: +((firstFaceFrame.bounding_box.x / imgW) * 100).toFixed(2),
    top: +((firstFaceFrame.bounding_box.y / imgH) * 100).toFixed(2),
    width: +((firstFaceFrame.bounding_box.width / imgW) * 100).toFixed(2),
    height: +((firstFaceFrame.bounding_box.height / imgH) * 100).toFixed(2),
    label: `Frame #${firstFaceFrame.frame_number}: ${firstFaceFrame.prediction?.toUpperCase()} (${((firstFaceFrame.probability ?? 0) * 100).toFixed(1)}%)`,
    prediction: firstFaceFrame.prediction,
    probability: firstFaceFrame.probability
  } : undefined;

  const primaryBox = faceBoxes.length > 0 ? faceBoxes[0] : videoPrimaryBox;

  const gradcamOverlay = resolveMediaUrl(data.gradcam_overlay_url || data.detection?.gradcam_overlay_url || '');
  const gradcamHeatmap = resolveMediaUrl(data.gradcam_heatmap_url || data.detection?.gradcam_heatmap_url || '');
  const originalFaceCrop = resolveMediaUrl(data.original_face_crop_url || data.detection?.original_face_crop_url || '');
  const gradcamAvailable = data.gradcam_available ?? data.detection?.gradcam_available ?? (!!gradcamOverlay);

  const rawMeta = data.metadata_analysis || data.detection?.metadata_analysis;
  const metadataAnalysis = rawMeta ? {
    available: rawMeta.available ?? true,
    metadataPresent: rawMeta.metadata_present ?? false,
    metadataFieldsCount: rawMeta.metadata_fields_count ?? 0,
    cameraMake: rawMeta.camera_make,
    cameraModel: rawMeta.camera_model,
    software: rawMeta.software,
    editingSoftwareDetected: rawMeta.editing_software_detected ?? false,
    gpsPresent: rawMeta.gps_present ?? false,
    creationTimestamp: rawMeta.creation_timestamp,
    modificationTimestamp: rawMeta.modification_timestamp,
    rawMetadata: rawMeta.raw_metadata || {},
    findings: rawMeta.findings || []
  } : undefined;

  const rawFreq = data.frequency_analysis || data.detection?.frequency_analysis;
  const frequencyAnalysis = rawFreq ? {
    available: rawFreq.available ?? true,
    analysisRegion: rawFreq.analysis_region ?? 'face_region',
    frequencySpectrumUrl: resolveMediaUrl(rawFreq.frequency_spectrum_url),
    meanMagnitude: rawFreq.mean_magnitude,
    medianMagnitude: rawFreq.median_magnitude,
    lowFrequencyEnergy: rawFreq.low_frequency_energy,
    midFrequencyEnergy: rawFreq.mid_frequency_energy,
    highFrequencyEnergy: rawFreq.high_frequency_energy,
    highFrequencyRatio: rawFreq.high_frequency_ratio,
    spectralCentroid: rawFreq.spectral_centroid,
    totalEnergy: rawFreq.total_energy,
    findings: rawFreq.findings || []
  } : undefined;

  const rawTemporal = data.temporal_analysis || data.detection?.temporal_analysis;

  return {
    id: data.analysis_id,
    filename: data.filename,
    fileSize: data.file_size_human || 'Unknown Size',
    mediaType: data.media_type,
    mediaUrl: mediaUrl,
    createdAt: data.created_at || new Date().toLocaleString(),
    status: 'completed',
    prediction: predictionStr,
    riskLevel: riskLevel,
    confidenceScore: confidenceScore,
    model: isModelLoaded ? modelName : 'Model Not Loaded',
    duration: data.metadata?.duration_formatted || (data.duration ? formatSecondsToMMSS(data.duration) : undefined),
    framesAnalyzed: data.frames_analyzed ?? data.metadata?.frame_count ?? (rawFrameResults.length > 0 ? rawFrameResults.length : 1),
    facesDetected: facesCount,
    suspiciousFrames: suspiciousFrames,
    indicators: [],
    timelineData: timelineData,
    originalFrameUrl: mediaUrl,
    heatmapUrl: gradcamOverlay,
    gradcamAvailable: gradcamAvailable,
    gradcamOverlayUrl: gradcamOverlay,
    gradcamHeatmapUrl: gradcamHeatmap,
    originalFaceCropUrl: originalFaceCrop,
    comparisonFrameUrl: mediaUrl,
    faceBoundingBox: primaryBox,
    faceBoundingBoxes: faceBoxes.length > 0 ? faceBoxes : (primaryBox ? [primaryBox] : []),
    selectedFaceId: data.selected_face_id || data.detection?.selected_face_id || (faceBoxes.length > 0 ? faceBoxes[0].faceId : undefined),
    analysisMode: (data.analysis_mode || data.detection?.analysis_mode || 'primary_face') as 'primary_face' | 'all_faces',
    primaryFaceResult: data.primary_face_result || data.detection?.primary_face_result ? {
      faceId: (data.primary_face_result || data.detection?.primary_face_result).face_id,
      faceIndex: (data.primary_face_result || data.detection?.primary_face_result).face_index,
      prediction: (data.primary_face_result || data.detection?.primary_face_result).prediction,
      probability: (data.primary_face_result || data.detection?.primary_face_result).probability,
      realProbability: (data.primary_face_result || data.detection?.primary_face_result).real_probability,
      fakeProbability: (data.primary_face_result || data.detection?.primary_face_result).fake_probability,
      rawLogits: (data.primary_face_result || data.detection?.primary_face_result).raw_logits,
      area: (data.primary_face_result || data.detection?.primary_face_result).area,
      relativeArea: (data.primary_face_result || data.detection?.primary_face_result).relative_area,
      marginNote: (data.primary_face_result || data.detection?.primary_face_result).margin_note,
      isLowMargin: (data.primary_face_result || data.detection?.primary_face_result).is_low_margin,
      taskScopeNote: (data.primary_face_result || data.detection?.primary_face_result).task_scope_note
    } : undefined,
    allFacesResult: data.all_faces_result || data.detection?.all_faces_result ? {
      faceCount: (data.all_faces_result || data.detection?.all_faces_result).face_count,
      prediction: (data.all_faces_result || data.detection?.all_faces_result).prediction,
      probability: (data.all_faces_result || data.detection?.all_faces_result).probability,
      meanFakeProbability: (data.all_faces_result || data.detection?.all_faces_result).mean_fake_probability,
      meanRealProbability: (data.all_faces_result || data.detection?.all_faces_result).mean_real_probability,
      aggregationRule: (data.all_faces_result || data.detection?.all_faces_result).aggregation_rule,
      marginNote: (data.all_faces_result || data.detection?.all_faces_result).margin_note,
      isLowMargin: (data.all_faces_result || data.detection?.all_faces_result).is_low_margin,
      limitationDisclosure: (data.all_faces_result || data.detection?.all_faces_result).limitation_disclosure
    } : undefined,
    isLowMargin: data.is_low_margin ?? data.detection?.is_low_margin ?? false,
    marginNote: data.margin_note || data.detection?.margin_note,
    taskScopeNote: data.task_scope_note || data.detection?.task_scope_note,
    rawProbability: effectiveProb,
    realProbability: typeof data.real_probability === 'number' ? data.real_probability : (typeof data.detection?.real_probability === 'number' ? data.detection.real_probability : undefined),
    fakeProbability: typeof data.fake_probability === 'number' ? data.fake_probability : (typeof data.detection?.fake_probability === 'number' ? data.detection.fake_probability : undefined),
    metadataAnalysis: metadataAnalysis,
    frequencyAnalysis: frequencyAnalysis,
    temporalAnalysis: rawTemporal || undefined
  };
}

export const forensicApiService = {
  // Stage evidence file and generate local preview
  async uploadMedia(file: File): Promise<UploadResponse> {
    const previewUrl = URL.createObjectURL(file);
    const mediaType = file.type.startsWith('video/') ? 'video' : 'image';
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(1) + ' MB';

    return {
      fileId: `DT-PRE-${Date.now().toString().slice(-6)}`,
      filename: file.name,
      fileSize: fileSizeMB,
      mediaType,
      previewUrl,
      rawFile: file,
    };
  },

  // Start multi-step analysis pipeline and dispatch to FastAPI
  async startAnalysis(
    fileData: UploadResponse,
    _modelId: string = 'xception-cnn',
    onProgress?: (stepIndex: number, stepName: string, percent: number) => void
  ): Promise<Analysis> {
    const steps = [
      'Uploading Evidence',
      'Preparing Media',
      'Detecting Faces',
      'Extracting Frames',
      'Running Detection Model',
      'Analyzing Evidence',
      'Preparing Results'
    ];

    // Simulate progress updates for the forensic steps
    for (let i = 0; i < 4; i++) {
      if (onProgress) {
        onProgress(i, steps[i], Math.round(((i + 1) / steps.length) * 100));
      }
      await new Promise((resolve) => setTimeout(resolve, 200));
    }

    if (USE_LIVE_BACKEND && fileData.rawFile) {
      try {
        const formData = new FormData();
        formData.append('file', fileData.rawFile);

        const res = await fetch(`${API_BASE_URL}/analyze`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
          throw new Error(errData.detail || `Upload failed with status ${res.status}`);
        }

        const data = await res.json();

        // Complete the remaining step indicators
        for (let i = 4; i < steps.length; i++) {
          if (onProgress) {
            onProgress(i, steps[i], Math.round(((i + 1) / steps.length) * 100));
          }
          await new Promise((resolve) => setTimeout(resolve, 100));
        }

        const newAnalysis = transformBackendAnalysis(data, fileData.previewUrl);
        newAnalysis.rawFile = fileData.rawFile;
        sessionAnalyses = [newAnalysis, ...sessionAnalyses];
        return newAnalysis;
      } catch (err: any) {
        throw new Error(err.message || `Failed to connect to backend at ${API_BASE_URL}`);
      }
    }

    // Fallback if no rawFile is available
    const newAnalysis: Analysis = {
      id: fileData.fileId,
      filename: fileData.filename,
      fileSize: fileData.fileSize,
      mediaType: fileData.mediaType,
      mediaUrl: fileData.previewUrl,
      createdAt: new Date().toLocaleString(),
      status: 'completed',
      prediction: 'INCONCLUSIVE',
      riskLevel: 'Inconclusive',
      confidenceScore: 0,
      model: 'Model Not Loaded',
      duration: fileData.mediaType === 'video' ? '00:01' : undefined,
      framesAnalyzed: 1,
      facesDetected: 0,
      suspiciousFrames: [],
      indicators: [],
      originalFrameUrl: fileData.previewUrl,
      heatmapUrl: '',
    };

    sessionAnalyses = [newAnalysis, ...sessionAnalyses];
    return newAnalysis;
  },

  // Get specific analysis by ID
  async getAnalysis(id: string): Promise<Analysis> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/analysis/${id}`);
        if (res.ok) {
          const data = await res.json();
          return transformBackendAnalysis(data);
        }
      } catch (e) {}
    }
    const found = sessionAnalyses.find((a) => a.id === id);
    if (found) return found;
    throw new Error(`Case ${id} not found`);
  },

  // Get all recent analysis cases
  async getAnalysisHistory(): Promise<Analysis[]> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/history`);
        if (res.ok) {
          const historyItems = await res.json();
          if (Array.isArray(historyItems) && historyItems.length > 0) {
            return historyItems.map((data: any) => transformBackendAnalysis(data));
          }
        }
      } catch (e) {
        // Fallback to session
      }
    }
    return [...sessionAnalyses];
  },

  // Get forensic models
  async getModels(): Promise<ModelInfo[]> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/models`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error('Failed to fetch models from backend:', e);
      }
    }
    return [];
  },

  // Get forensic reports
  async getReports(): Promise<ForensicReport[]> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/reports`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error('Failed to fetch reports from backend:', e);
      }
    }
    return [...sessionReports];
  },

  // Stage 8G: Get full detailed report by ID or case ID
  async getDetailedReport(reportId: string): Promise<DetailedForensicReportData | null> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/reports/${reportId}`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error(`Failed to fetch detailed report ${reportId}:`, e);
      }
    }
    return null;
  },

  // Stage 8G: Get printable report HTML URL
  getPrintableReportUrl(reportId: string): string {
    return `${API_BASE_URL}/reports/${reportId}/printable`;
  },

  // Stage 8D: Get all saved benchmark runs
  async getBenchmarkRuns(): Promise<BenchmarkRunSummaryItem[]> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/benchmark/runs`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error('Failed to fetch benchmark runs:', e);
      }
    }
    return [];
  },

  // Stage 8D: Get details of a specific benchmark run
  async getBenchmarkRun(runId: string): Promise<BenchmarkRunDetailData | null> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/benchmark/runs/${runId}`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error(`Failed to fetch benchmark run ${runId}:`, e);
      }
    }
    return null;
  },

  // Stage 8D: Get sample manifest template
  async getBenchmarkTemplate(): Promise<BenchmarkTemplate | null> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/benchmark/template`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error('Failed to fetch benchmark template:', e);
      }
    }
    return null;
  },

  // Stage 8D: Trigger a benchmark run on a manifest path or content
  async runBenchmark(manifestPath?: string, manifestContent?: any, threshold?: number): Promise<BenchmarkRunDetailData> {
    const res = await fetch(`${API_BASE_URL}/benchmark/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        manifest_path: manifestPath,
        manifest_content: manifestContent,
        threshold: threshold ?? 0.5
      })
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errorData.detail || `Benchmark run failed with status ${res.status}`);
    }

    return await res.json();
  },

  // Stage 8E: Cross-Validation API
  async getCrossValidationModels(): Promise<ModelRegistryItem[]> {
    if (USE_LIVE_BACKEND) {
      try {
        const res = await fetch(`${API_BASE_URL}/cross-validation/models`);
        if (res.ok) return await res.json();
      } catch (e) {
        console.error('Failed to fetch cross-validation models:', e);
      }
    }
    return [];
  },

  async crossValidateImage(options: { file?: File; caseId?: string; filePath?: string; comparisonModelIds?: string[] }): Promise<ImageCrossValidationData> {
    if (options.file) {
      const formData = new FormData();
      formData.append('file', options.file);
      if (options.comparisonModelIds && options.comparisonModelIds.length > 0) {
        formData.append('comparison_model_ids', JSON.stringify(options.comparisonModelIds));
      }
      const res = await fetch(`${API_BASE_URL}/cross-validation/image-upload`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Image cross-validation failed: ${res.status}`);
      }
      return await res.json();
    } else {
      const res = await fetch(`${API_BASE_URL}/cross-validation/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: options.caseId,
          file_path: options.filePath,
          comparison_model_ids: options.comparisonModelIds
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Image cross-validation failed: ${res.status}`);
      }
      return await res.json();
    }
  },

  async crossValidateVideo(options: { file?: File; caseId?: string; filePath?: string; comparisonModelIds?: string[]; maxFrames?: number }): Promise<VideoCrossValidationData> {
    if (options.file) {
      const formData = new FormData();
      formData.append('file', options.file);
      if (options.comparisonModelIds && options.comparisonModelIds.length > 0) {
        formData.append('comparison_model_ids', JSON.stringify(options.comparisonModelIds));
      }
      if (options.maxFrames) {
        formData.append('max_frames', options.maxFrames.toString());
      }
      const res = await fetch(`${API_BASE_URL}/cross-validation/video-upload`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Video cross-validation failed: ${res.status}`);
      }
      return await res.json();
    } else {
      const res = await fetch(`${API_BASE_URL}/cross-validation/video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: options.caseId,
          file_path: options.filePath,
          comparison_model_ids: options.comparisonModelIds,
          max_frames: options.maxFrames
        })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `Video cross-validation failed: ${res.status}`);
      }
      return await res.json();
    }
  },

  async runMultiModelBenchmark(options: { manifestPath?: string; manifestContent?: any; modelIds?: string[]; threshold?: number }): Promise<MultiModelBenchmarkComparisonData> {
    const res = await fetch(`${API_BASE_URL}/cross-validation/benchmark-comparison`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        manifest_path: options.manifestPath,
        manifest_content: options.manifestContent,
        model_ids: options.modelIds,
        threshold: options.threshold ?? 0.5
      })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Multi-model benchmark failed: ${res.status}`);
    }
    return await res.json();
  }
};
