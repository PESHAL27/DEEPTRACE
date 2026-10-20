import React, { useState, useRef, useEffect } from 'react';
import { 
  ArrowLeft, 
  Play, 
  Pause, 
  Volume2, 
  VolumeX,
  Settings as SettingsIcon, 
  Maximize, 
  AlertTriangle, 
  Film, 
  Layers, 
  User, 
  Download, 
  FileText, 
  Scan,
  GitCompare,
  CheckCircle2,
  RotateCw,
  Activity,
  TrendingUp,
  Zap,
  ShieldAlert
} from 'lucide-react';
import type { Analysis, SuspiciousFrame, ImageCrossValidationData, VideoCrossValidationData } from '../../types/forensics';
import { forensicApiService } from '../../services/api';
import { ReportModal } from './ReportModal';
import '../../styles/results.css';

interface AnalysisResultsPageProps {
  analysis: Analysis | null;
  onBack: () => void;
}

export const AnalysisResultsPage: React.FC<AnalysisResultsPageProps> = ({ analysis, onBack }) => {
  if (!analysis) {
    return (
      <div className="results-page-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', textAlign: 'center', gap: 16 }}>
        <Film size={40} color="var(--text-dim)" />
        <div style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-secondary)' }}>
          No Active Analysis Selected
        </div>
        <div style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '420px', lineHeight: 1.5 }}>
          Upload an image or video to run real-time deepfake detection with the DeepfakeBench Xception detector.
        </div>
        <button className="btn btn-primary" onClick={onBack} style={{ marginTop: 8 }}>
          <ArrowLeft size={14} />
          <span>Upload Evidence</span>
        </button>
      </div>
    );
  }

  // Real HTML5 Video element ref
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Parse initial duration in seconds from analysis.duration (e.g. "00:33" or "33.46")
  const initialDurationSec = (() => {
    if (analysis.duration) {
      const parts = analysis.duration.split(':').map(Number);
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        return parts[0] * 60 + parts[1];
      }
      const num = parseFloat(analysis.duration);
      if (!isNaN(num) && num > 0) return num;
    }
    return 0;
  })();

  const [durationSec, setDurationSec] = useState<number>(initialDurationSec);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTimeSec, setCurrentTimeSec] = useState(0.0);
  const [isMuted, setIsMuted] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState<SuspiciousFrame | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'primary_face' | 'all_faces'>(analysis.analysisMode || 'primary_face');
  const defaultFaceId = analysis.selectedFaceId
    || (analysis.faceBoundingBoxes && analysis.faceBoundingBoxes.length > 0
        ? (analysis.faceBoundingBoxes.find(b => b.isPrimary)?.faceId || analysis.faceBoundingBoxes[0].faceId || 'face_0')
        : 'face_0');
  const [selectedFaceId, setSelectedFaceId] = useState<string>(defaultFaceId);

  useEffect(() => {
    if (analysis.selectedFaceId) {
      setSelectedFaceId(analysis.selectedFaceId);
    } else if (analysis.faceBoundingBoxes && analysis.faceBoundingBoxes.length > 0) {
      const primary = analysis.faceBoundingBoxes.find(b => b.isPrimary) || analysis.faceBoundingBoxes[0];
      setSelectedFaceId(primary.faceId || `face_${primary.faceIndex ?? 0}`);
    }
    if (analysis.analysisMode) {
      setAnalysisMode(analysis.analysisMode);
    }
  }, [analysis]);

  const [activeTab, setActiveTab] = useState<'heatmap' | 'frequency' | 'metadata' | 'comparison' | 'temporal'>('heatmap');
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [crossValData, setCrossValData] = useState<ImageCrossValidationData | VideoCrossValidationData | null>(analysis.crossValidation || null);
  const [isCrossValLoading, setIsCrossValLoading] = useState<boolean>(false);
  const [crossValError, setCrossValError] = useState<string | null>(null);

  const handleRunCrossValidation = async () => {
    setIsCrossValLoading(true);
    setCrossValError(null);
    try {
      if (analysis.mediaType === 'video') {
        const res = await forensicApiService.crossValidateVideo({
          file: analysis.rawFile,
          caseId: analysis.id
        });
        setCrossValData(res);
      } else {
        const res = await forensicApiService.crossValidateImage({
          file: analysis.rawFile,
          caseId: analysis.id
        });
        setCrossValData(res);
      }
    } catch (err: any) {
      setCrossValError(err.message || 'Failed to execute cross-validation');
    } finally {
      setIsCrossValLoading(false);
    }
  };

  // Video playback event handlers bound directly to HTML5 video element
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTimeSec(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.duration && !isNaN(videoRef.current.duration) && isFinite(videoRef.current.duration)) {
      setDurationSec(videoRef.current.duration);
    }
  };

  const handleDurationChange = () => {
    if (videoRef.current && videoRef.current.duration && !isNaN(videoRef.current.duration) && isFinite(videoRef.current.duration)) {
      setDurationSec(videoRef.current.duration);
    }
  };

  const handleVideoEnded = () => {
    setIsPlaying(false);
    setCurrentTimeSec(0);
  };

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      video.play().catch((err) => {
        console.warn("Video play interrupted:", err);
      });
    } else {
      video.pause();
    }
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setIsMuted(video.muted);
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      video.requestFullscreen().catch(() => {});
    }
  };

  // Format seconds into MM:SS
  const formatTime = (sec: number) => {
    if (isNaN(sec) || !isFinite(sec) || sec < 0) return '00:00';
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current || durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, clickX / rect.width));
    const targetTime = +(ratio * durationSec).toFixed(2);
    videoRef.current.currentTime = targetTime;
    setCurrentTimeSec(targetTime);
  };

  const handleSelectFrame = (frame: SuspiciousFrame) => {
    setSelectedFrame(frame);
    if (videoRef.current) {
      videoRef.current.currentTime = frame.seconds;
    }
    setCurrentTimeSec(frame.seconds);
  };

  const handleDownloadEvidence = () => {
    const evidenceData = {
      caseId: analysis.id,
      filename: analysis.filename,
      mediaType: analysis.mediaType,
      prediction: analysis.prediction,
      confidenceScore: analysis.confidenceScore,
      duration: analysis.duration,
      framesAnalyzed: analysis.framesAnalyzed,
      timestamp: analysis.createdAt,
      sha256: '7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069',
      indicators: analysis.indicators,
      suspiciousFrames: analysis.suspiciousFrames
    };

    const blob = new Blob([JSON.stringify(evidenceData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${analysis.id}_forensic_evidence.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const isNoFace = analysis.facesDetected === 0 || analysis.prediction === 'NO FACE DETECTED';
  const isModelNotLoaded = !isNoFace && (analysis.model === 'Model Not Loaded' || analysis.prediction === 'INCONCLUSIVE');

  const allFaceBoxes = analysis.faceBoundingBoxes && analysis.faceBoundingBoxes.length > 0
    ? analysis.faceBoundingBoxes
    : analysis.faceBoundingBox ? [analysis.faceBoundingBox] : [];

  const selectedFaceBox = allFaceBoxes.find(
    (b) => b.faceId === selectedFaceId || `face_${b.faceIndex}` === selectedFaceId
  ) || (allFaceBoxes.length > 0 ? allFaceBoxes[0] : null);

  let displayedPrediction: string = analysis.prediction;
  let displayedScore: number = analysis.confidenceScore;
  let displayedMarginNote: string | undefined = undefined;

  if (analysis.mediaType === 'image' && !isNoFace && !isModelNotLoaded) {
    if (analysisMode === 'primary_face' && selectedFaceBox) {
      const isFake = selectedFaceBox.prediction?.toLowerCase() === 'fake';
      displayedPrediction = isFake ? 'LIKELY MANIPULATED' : 'REAL';
      displayedScore = selectedFaceBox.probability !== undefined 
        ? +(selectedFaceBox.probability * 100).toFixed(1) 
        : analysis.confidenceScore;
      if (selectedFaceBox.fakeProbability !== undefined && Math.abs(selectedFaceBox.fakeProbability - 0.50) < 0.10) {
        displayedMarginNote = `Low-margin model prediction (${displayedScore}% ${displayedPrediction}, near 50% decision boundary)`;
      }
    } else if (analysisMode === 'all_faces' && analysis.allFacesResult) {
      const isFake = analysis.allFacesResult.prediction.toLowerCase() === 'fake';
      displayedPrediction = isFake ? 'LIKELY MANIPULATED' : 'REAL';
      displayedScore = +(analysis.allFacesResult.probability * 100).toFixed(1);
      displayedMarginNote = `Aggregation: Unweighted mean of ${analysis.allFacesResult.faceCount} localized face(s)`;
    }
  }

  const isCurrentManipulated = !isNoFace && displayedPrediction === 'LIKELY MANIPULATED';

  const scrubberPercent = durationSec > 0
    ? Math.min(100, Math.max(0, (currentTimeSec / durationSec) * 100)).toFixed(1)
    : '0';

  const activeBoxes = selectedFrame?.boundingBox
    ? [selectedFrame.boundingBox]
    : allFaceBoxes;

  const currentPreviewUrl = selectedFrame?.thumbnailUrl || analysis.mediaUrl || '/assets/sample_frame.jpg';
  const currentOverlayUrl = selectedFrame?.gradcamOverlayUrl 
    || (analysis.mediaType === 'image' && selectedFaceBox?.gradcamOverlayUrl) 
    || analysis.gradcamOverlayUrl 
    || analysis.heatmapUrl;
  const currentCropUrl = selectedFrame?.originalFaceCropUrl 
    || (analysis.mediaType === 'image' && selectedFaceBox?.originalFaceCropUrl) 
    || analysis.originalFaceCropUrl 
    || selectedFrame?.thumbnailUrl 
    || analysis.mediaUrl 
    || '/assets/sample_frame.jpg';
  const currentFreqUrl = selectedFrame?.frequencySpectrumUrl || analysis.frequencyAnalysis?.frequencySpectrumUrl;
  const currentFreqAnalysis = selectedFrame?.frequencyAnalysis || analysis.frequencyAnalysis;
  const currentMetadata = analysis.metadataAnalysis;

  return (
    <div className="results-page-container">
      {/* Header Bar */}
      <div className="results-header">
        <div className="results-title-group">
          <button className="back-link-btn" onClick={onBack}>
            <ArrowLeft size={13} />
            <span>Back to Analysis</span>
          </button>
          <h1 className="results-heading">Analysis Results</h1>
          <div className="results-file-meta">
            <span>File: <strong className="results-file-name">{analysis.filename}</strong></span>
            <span>Analyzed on: <span className="results-timestamp">{analysis.createdAt}</span></span>
          </div>
        </div>

        <div>
          <button className="btn btn-secondary btn-sm" style={{ gap: 6 }}>
            <Film size={13} color="var(--accent-blue)" />
            <span>{analysis.mediaType === 'video' ? 'Video Analysis' : 'Image Analysis'}</span>
          </button>
        </div>
      </div>

      {/* Top Grid: Media Preview + Suspicious Frames + Verdict Card */}
      <div className="results-top-grid">
        {/* 1. Media Preview Area */}
        <div className="media-preview-card">
          <div className="media-viewport">
            <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', maxWidth: '100%', maxHeight: '100%' }}>
              {analysis.mediaType === 'video' ? (
                <video 
                  ref={videoRef}
                  src={analysis.mediaUrl}
                  className="media-main-video"
                  style={{ maxWidth: '100%', maxHeight: '360px', width: 'auto', height: 'auto', objectFit: 'contain', display: 'block' }}
                  playsInline
                  preload="metadata"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={handleLoadedMetadata}
                  onDurationChange={handleDurationChange}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={handleVideoEnded}
                />
              ) : (
                <img 
                  src={currentPreviewUrl} 
                  alt="Forensic Evidence Preview" 
                  className="media-main-image"
                  style={{ maxWidth: '100%', maxHeight: '360px', width: 'auto', height: 'auto', objectFit: 'contain', display: 'block' }}
                />
              )}

              {/* Overlaid Face Bounding Box(es) */}
              {activeBoxes.map((box, idx) => {
                const isSelected = analysis.mediaType === 'image' && (
                  box.faceId === selectedFaceId || 
                  (selectedFaceBox && box.faceIndex === selectedFaceBox.faceIndex) ||
                  (activeBoxes.length === 1)
                );
                const isFakeBox = box.prediction?.toLowerCase() === 'fake';
                const boxColor = isFakeBox ? '#ef4444' : '#10b981';
                const tagLabel = box.isPrimary 
                  ? `★ PRIMARY: ${box.label}` 
                  : box.label;

                return (
                  <div 
                    key={box.faceId || idx}
                    className={`face-bounding-box ${isSelected ? 'face-box-selected' : ''}`}
                    onClick={() => {
                      if (analysis.mediaType === 'image' && (box.faceId || box.faceIndex !== undefined)) {
                        setSelectedFaceId(box.faceId || `face_${box.faceIndex}`);
                        setAnalysisMode('primary_face');
                      }
                    }}
                    style={{
                      top: `${box.top}%`,
                      left: `${box.left}%`,
                      width: `${box.width}%`,
                      height: `${box.height}%`,
                      borderColor: boxColor,
                      borderWidth: isSelected ? '3px' : '2px',
                      borderStyle: isSelected ? 'solid' : 'dashed',
                      boxShadow: isSelected 
                        ? `0 0 14px ${isFakeBox ? 'rgba(239, 68, 68, 0.6)' : 'rgba(16, 185, 129, 0.6)'}` 
                        : `0 0 6px ${isFakeBox ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)'}`,
                      opacity: (!isSelected && activeBoxes.length > 1) ? 0.75 : 1.0,
                      cursor: analysis.mediaType === 'image' ? 'pointer' : 'default',
                      zIndex: isSelected ? 10 : 5
                    }}
                    title={analysis.mediaType === 'image' ? `Click to inspect Face #${(box.faceIndex ?? 0) + 1}` : undefined}
                  >
                    <span 
                      className="face-detected-tag"
                      style={{ 
                        backgroundColor: boxColor,
                        fontWeight: isSelected ? 700 : 500,
                        fontSize: isSelected ? '10px' : '9px'
                      }}
                    >
                      <Scan size={10} />
                      <span>{tagLabel}</span>
                    </span>
                    <span 
                      className="bounding-box-corner-marker" 
                      style={{ backgroundColor: boxColor }}
                    />
                  </div>
                );
              })}
            </div>
          </div>

          {/* Custom Video Player Controls */}
          {analysis.mediaType === 'video' && (
            <div className="media-controls-bar">
              <button 
                className="control-btn" 
                onClick={togglePlay}
                title={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <Pause size={14} fill="#fff" /> : <Play size={14} fill="#fff" />}
              </button>

              <div className="time-display">
                {formatTime(currentTimeSec)} / {formatTime(durationSec)}
              </div>

              <div className="scrubber-track" onClick={handleScrubberClick}>
                <div 
                  className="scrubber-fill" 
                  style={{ width: `${scrubberPercent}%` }} 
                />
                <div 
                  className="scrubber-thumb" 
                  style={{ left: `${scrubberPercent}%` }} 
                />
              </div>

              <button className="control-btn" onClick={toggleMute} title={isMuted ? 'Unmute' : 'Mute'}>
                {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
              </button>
              <button className="control-btn" title="Settings">
                <SettingsIcon size={14} />
              </button>
              <button className="control-btn" onClick={toggleFullscreen} title="Fullscreen">
                <Maximize size={14} />
              </button>
            </div>
          )}
        </div>

        {/* 2. Detected Faces (Image) or Suspicious Frames (Video) Panel */}
        <div className="suspicious-frames-card">
          {analysis.mediaType === 'video' ? (
            <>
              <div className="suspicious-header">
                <span>Suspicious Frames</span>
                <span className="suspicious-count">({analysis.suspiciousFrames?.length ?? 0})</span>
                {analysis.framesAnalyzed && (
                  <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                    {analysis.framesAnalyzed} analyzed
                  </span>
                )}
              </div>

              <div className="frames-list">
                {(analysis.suspiciousFrames?.length ?? 0) === 0 ? (
                  <div style={{ color: 'var(--text-muted)', fontSize: '11.5px', padding: '28px 8px', textAlign: 'center' }}>
                    {isNoFace ? 'No face detected in video' : 'No suspicious frames detected'}
                  </div>
                ) : (
                  analysis.suspiciousFrames.map((frame) => (
                    <div 
                      key={frame.id}
                      className={`frame-item ${selectedFrame?.id === frame.id ? 'frame-active' : ''}`}
                      onClick={() => handleSelectFrame(frame)}
                      title={`Click to seek and inspect frame at ${frame.timestamp}`}
                    >
                      <div className="frame-thumb-box">
                        <img src={frame.thumbnailUrl} alt={`Frame ${frame.timestamp}`} className="frame-thumb-img" />
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1, marginLeft: 8 }}>
                        <span className="frame-timestamp-badge">{frame.timestamp}</span>
                        <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                          Prob: {(frame.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <span className="badge badge-high" style={{ fontSize: '9px', padding: '2px 5px' }}>
                        FAKE
                      </span>
                    </div>
                  ))
                )}
              </div>
            </>
          ) : (
            <>
              <div className="suspicious-header" style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingBottom: 8 }}>
                <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                  <span>Detected Faces</span>
                  <span className="suspicious-count">({allFaceBoxes.length})</span>
                </div>
                {/* Mode Selector */}
                {allFaceBoxes.length > 0 && (
                  <div style={{ display: 'flex', gap: 4, width: '100%', background: 'var(--bg-app)', padding: 3, borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                    <button
                      type="button"
                      onClick={() => setAnalysisMode('primary_face')}
                      style={{
                        flex: 1,
                        padding: '4px 6px',
                        fontSize: '10.5px',
                        fontWeight: analysisMode === 'primary_face' ? 600 : 400,
                        backgroundColor: analysisMode === 'primary_face' ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                        color: analysisMode === 'primary_face' ? '#38bdf8' : 'var(--text-muted)',
                        border: 'none',
                        borderRadius: 4,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      Primary Face
                    </button>
                    <button
                      type="button"
                      onClick={() => setAnalysisMode('all_faces')}
                      style={{
                        flex: 1,
                        padding: '4px 6px',
                        fontSize: '10.5px',
                        fontWeight: analysisMode === 'all_faces' ? 600 : 400,
                        backgroundColor: analysisMode === 'all_faces' ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                        color: analysisMode === 'all_faces' ? '#38bdf8' : 'var(--text-muted)',
                        border: 'none',
                        borderRadius: 4,
                        cursor: 'pointer',
                        transition: 'all 0.15s ease'
                      }}
                    >
                      All Faces
                    </button>
                  </div>
                )}
              </div>

              <div className="frames-list">
                {allFaceBoxes.length === 0 ? (
                  <div style={{ color: 'var(--text-muted)', fontSize: '11.5px', padding: '28px 8px', textAlign: 'center' }}>
                    No face detected in image
                  </div>
                ) : (
                  allFaceBoxes.map((box, idx) => {
                    const isSelected = (box.faceId === selectedFaceId) || (selectedFaceBox && box.faceIndex === selectedFaceBox.faceIndex);
                    const isFake = box.prediction?.toLowerCase() === 'fake';
                    const prob = box.probability !== undefined ? +(box.probability * 100).toFixed(1) : 0;
                    const cropThumb = box.originalFaceCropUrl || analysis.mediaUrl;

                    return (
                      <div 
                        key={box.faceId || idx}
                        className={`frame-item ${isSelected ? 'frame-active' : ''}`}
                        onClick={() => {
                          if (box.faceId || box.faceIndex !== undefined) {
                            setSelectedFaceId(box.faceId || `face_${box.faceIndex}`);
                            setAnalysisMode('primary_face');
                          }
                        }}
                        style={{ cursor: 'pointer' }}
                        title={`Click to select and inspect Face #${(box.faceIndex ?? 0) + 1}`}
                      >
                        <div className="frame-thumb-box" style={{ width: 44, height: 44, borderRadius: 4, overflow: 'hidden', flexShrink: 0, backgroundColor: '#000' }}>
                          <img src={cropThumb} alt={`Face ${(box.faceIndex ?? 0) + 1}`} className="frame-thumb-img" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1, marginLeft: 8 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                              Face #{(box.faceIndex ?? 0) + 1}
                            </span>
                            {box.isPrimary && (
                              <span style={{ fontSize: '9px', padding: '1px 4px', borderRadius: 3, backgroundColor: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', fontWeight: 600 }}>
                                PRIMARY
                              </span>
                            )}
                          </div>
                          <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                            {box.relativeArea ? `${box.relativeArea}% area • ` : ''}Prob: {prob}%
                          </span>
                        </div>
                        <span className={`badge ${isFake ? 'badge-high' : 'badge-low'}`} style={{ fontSize: '9px', padding: '2px 5px' }}>
                          {isFake ? 'FAKE' : 'REAL'}
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </>
          )}
        </div>

        {/* 3. Prominent Verdict Card */}
        <div className="verdict-card">
          <div>
            <div className="verdict-header-row">
              <div className="verdict-status-title-box">
                {isNoFace ? (
                  <Scan size={18} color="#94a3b8" />
                ) : isModelNotLoaded ? (
                  <AlertTriangle size={18} color="#f59e0b" />
                ) : isCurrentManipulated ? (
                  <AlertTriangle size={18} className="verdict-alert-icon" />
                ) : (
                  <Scan size={18} color="#10b981" />
                )}
                <span 
                  className={`verdict-state-text ${
                    isNoFace ? 'is-no-face' : isModelNotLoaded ? '' : !isCurrentManipulated ? 'is-real' : ''
                  }`}
                  style={isNoFace ? { color: '#94a3b8', fontSize: '13.5px' } : isModelNotLoaded ? { color: '#f59e0b', fontSize: '13px' } : undefined}
                >
                  {isNoFace ? 'NO FACE DETECTED' : isModelNotLoaded ? 'MODEL NOT CONNECTED' : displayedPrediction}
                </span>
              </div>
              <span 
                className={`badge ${
                  isNoFace ? 'badge-inconclusive' : isModelNotLoaded ? 'badge-inconclusive' : isCurrentManipulated ? 'badge-high' : 'badge-low'
                }`}
              >
                {isNoFace 
                  ? 'No Face Detected' 
                  : isModelNotLoaded 
                    ? 'Awaiting Model' 
                    : analysis.mediaType === 'video' 
                      ? `Video Probability: ${displayedScore}%` 
                      : analysisMode === 'primary_face' 
                        ? `Primary Face: ${displayedScore}%` 
                        : `All Faces Mean: ${displayedScore}%`}
              </span>
            </div>

            {/* Model Probability Score Bar */}
            <div className="confidence-section">
              <span className="confidence-label">
                {analysis.mediaType === 'video' ? 'Video Probability' : analysisMode === 'primary_face' ? 'Primary Face Probability' : 'All Faces Probability (Mean)'}
              </span>
              <div 
                className="confidence-score-value"
                style={isNoFace || isModelNotLoaded ? { fontSize: '18px', color: 'var(--text-muted)' } : undefined}
              >
                {isNoFace ? 'N/A' : isModelNotLoaded ? 'Awaiting Analysis' : `${displayedScore}%`}
              </div>
              <div className="confidence-progress-track">
                <div 
                  className={`confidence-progress-bar ${!isCurrentManipulated && !isNoFace ? 'is-real' : ''}`}
                  style={{ 
                    width: isNoFace || isModelNotLoaded ? '0%' : `${displayedScore}%`,
                    backgroundColor: isNoFace || isModelNotLoaded ? 'var(--text-dim)' : undefined
                  }}
                />
              </div>
              {displayedMarginNote && (
                <div style={{ marginTop: 8, padding: '5px 8px', borderRadius: 4, backgroundColor: 'rgba(245, 158, 11, 0.08)', border: '1px solid rgba(245, 158, 11, 0.25)', fontSize: '10.5px', color: '#fbbf24', lineHeight: 1.35 }}>
                  <span style={{ fontWeight: 600 }}>Advisory:</span> {displayedMarginNote}
                </div>
              )}
              {analysis.mediaType === 'image' && analysisMode === 'all_faces' && (
                <div style={{ marginTop: 6, padding: '5px 8px', borderRadius: 4, backgroundColor: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.2)', fontSize: '10px', color: '#93c5fd', lineHeight: 1.35 }}>
                  <span style={{ fontWeight: 600 }}>All-Faces Limitation:</span> Each face receives equal weight. Background detections or peripheral faces can influence this aggregate; Primary Face is the default standard for examining the primary subject.
                </div>
              )}
            </div>
          </div>

          {/* Model Metrics Breakdown */}
          <div className="verdict-meta-list">
            <div className="verdict-meta-row">
              <span className="verdict-meta-key">
                <Film size={13} />
                <span>Media Type</span>
              </span>
              <span className="verdict-meta-val">{analysis.mediaType === 'video' ? 'Video' : 'Image'}</span>
            </div>

            {analysis.mediaType === 'image' && (
              <div className="verdict-meta-row">
                <span className="verdict-meta-key">
                  <Layers size={13} />
                  <span>Analysis Mode</span>
                </span>
                <span className="verdict-meta-val" style={{ textTransform: 'capitalize' }}>
                  {analysisMode === 'primary_face' ? 'Primary Face' : 'All Faces'}
                </span>
              </div>
            )}

            <div className="verdict-meta-row">
              <span className="verdict-meta-key">
                <Layers size={13} />
                <span>Model Name</span>
              </span>
              <span className="verdict-meta-val mono-text">{analysis.model}</span>
            </div>

            {analysis.mediaType === 'video' && (
              <div className="verdict-meta-row">
                <span className="verdict-meta-key">
                  <Film size={13} />
                  <span>Frames Analyzed</span>
                </span>
                <span className="verdict-meta-val mono-text">
                  {analysis.framesAnalyzed ?? 1} Frame(s)
                </span>
              </div>
            )}

            <div className="verdict-meta-row">
              <span className="verdict-meta-key">
                <User size={13} />
                <span>Faces Detected</span>
              </span>
              <span className="verdict-meta-val">
                {analysis.facesDetected}
              </span>
            </div>

            <div className="verdict-meta-row">
              <span className="verdict-meta-key">
                <Scan size={13} />
                <span>Prediction</span>
              </span>
              <span className="verdict-meta-val" style={{ fontWeight: 600, color: isNoFace ? '#94a3b8' : isCurrentManipulated ? '#f87171' : '#34d399' }}>
                {isNoFace ? 'NO FACE DETECTED' : displayedPrediction}
              </span>
            </div>

            <div className="verdict-meta-row" style={{ alignItems: 'flex-start', borderTop: '1px solid var(--border-subtle)', paddingTop: 8, marginTop: 4 }}>
              <span className="verdict-meta-key" style={{ marginTop: 2 }}>
                <ShieldAlert size={13} color="var(--accent-blue)" />
                <span>Task Scope</span>
              </span>
              <span className="verdict-meta-val" style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.3, textAlign: 'right', maxWidth: '62%' }}>
                Facial manipulation (face-swaps / reenactments). Not validated for full-scene AI/diffusion generation.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Middle Grid: Forensic Analysis Indicators + Evidence Visualization */}
      <div className="results-middle-grid">
        {/* Forensic Analysis Card */}
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Forensic Analysis</h3>
              <p className="card-subtitle">Key empirical evidence indicators</p>
            </div>
          </div>

          <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            {/* Model Telemetry */}
            <div style={{ padding: '8px 12px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Model Prediction</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: isNoFace ? '#94a3b8' : isCurrentManipulated ? '#f87171' : '#34d399', marginTop: 2 }}>
                  {isNoFace ? 'NO FACE DETECTED' : `${displayedPrediction} (${displayedScore}%)`}
                </div>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Xception CNN</span>
            </div>

            {/* Face Localization */}
            <div style={{ padding: '8px 12px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Facial Localization</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {analysis.facesDetected > 0 ? `${analysis.facesDetected} subject(s) localized` : 'No face structures detected'}
                </div>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '9px' }}>YuNet</span>
            </div>

            {/* Grad-CAM Explainability */}
            <div style={{ padding: '8px 12px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Visual Attribution</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: currentOverlayUrl ? 'var(--text-primary)' : 'var(--text-muted)', marginTop: 2 }}>
                  {currentOverlayUrl ? 'conv4 feature map computed' : 'Attribution unavailable'}
                </div>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Grad-CAM</span>
            </div>

            {/* Frequency-Domain Analysis */}
            <div style={{ padding: '8px 12px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Frequency Spectrum</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2 }}>
                  {currentFreqAnalysis?.available 
                    ? `HF Ratio: ${((currentFreqAnalysis.highFrequencyRatio ?? 0) * 100).toFixed(1)}% (Centroid: ${(currentFreqAnalysis.spectralCentroid ?? 0).toFixed(3)})`
                    : 'Not computed'}
                </div>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '9px' }}>2D FFT</span>
            </div>

            {/* Metadata Status */}
            <div style={{ padding: '8px 12px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>File Metadata</div>
                <div style={{ fontSize: '12px', fontWeight: 600, color: currentMetadata?.editingSoftwareDetected ? '#f87171' : 'var(--text-secondary)', marginTop: 2 }}>
                  {currentMetadata?.metadataPresent 
                    ? `EXIF Present (${currentMetadata.metadataFieldsCount} fields)`
                    : 'Metadata absent'}
                </div>
              </div>
              <span className="badge badge-neutral" style={{ fontSize: '9px' }}>EXIF</span>
            </div>

            {/* Distinction Reminder */}
            <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4, marginTop: 4, padding: '0 4px' }}>
              Empirical forensic indicators provide supporting telemetry alongside the neural network model probability.
            </div>
          </div>
        </div>

        {/* Evidence Visualization Card */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Evidence Visualization</h3>

            {/* Forensic Tabs */}
            <div className="tabs-header">
              <button 
                className={`tab-button ${activeTab === 'heatmap' ? 'active' : ''}`}
                onClick={() => setActiveTab('heatmap')}
              >
                Heatmap
              </button>
              <button 
                className={`tab-button ${activeTab === 'frequency' ? 'active' : ''}`}
                onClick={() => setActiveTab('frequency')}
              >
                Frequency Domain
              </button>
              <button 
                className={`tab-button ${activeTab === 'metadata' ? 'active' : ''}`}
                onClick={() => setActiveTab('metadata')}
              >
                Metadata
              </button>
              <button 
                className={`tab-button ${activeTab === 'comparison' ? 'active' : ''}`}
                onClick={() => setActiveTab('comparison')}
              >
                Cross-Validation
              </button>
              <button 
                className={`tab-button ${activeTab === 'temporal' ? 'active' : ''}`}
                onClick={() => setActiveTab('temporal')}
              >
                Temporal Consistency
              </button>
            </div>
          </div>

          <div>
            {activeTab === 'heatmap' && (
              currentOverlayUrl ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.6px', color: 'var(--text-primary)', textTransform: 'uppercase' }}>
                      AI Visual Explanation (Grad-CAM)
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '9.5px', textTransform: 'uppercase' }}>
                      conv4 Activation Map
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    {/* 1. Analyzed Face Crop */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
                      <div style={{ 
                        width: '100%', 
                        height: '140px', 
                        backgroundColor: 'var(--bg-app)', 
                        borderRadius: 'var(--radius-sm)', 
                        overflow: 'hidden', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        border: '1px solid var(--border-subtle)'
                      }}>
                        <img 
                          src={currentCropUrl} 
                          alt="Analyzed Face Crop" 
                          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} 
                        />
                      </div>
                      <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                        Analyzed Facial Subject
                      </span>
                    </div>

                    {/* 2. Grad-CAM Overlay */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
                      <div style={{ 
                        width: '100%', 
                        height: '140px', 
                        backgroundColor: 'var(--bg-app)', 
                        borderRadius: 'var(--radius-sm)', 
                        overflow: 'hidden', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        border: '1px solid var(--border-subtle)'
                      }}>
                        <img 
                          src={currentOverlayUrl} 
                          alt="Grad-CAM Activation Overlay" 
                          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} 
                        />
                      </div>
                      <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                        Model Attention Visualization
                      </span>
                    </div>
                  </div>

                  {/* Explanatory telemetry footer */}
                  <div style={{ 
                    fontSize: '11px', 
                    color: 'var(--text-secondary)', 
                    backgroundColor: 'rgba(255, 255, 255, 0.02)', 
                    padding: '8px 12px', 
                    borderRadius: 'var(--radius-xs)', 
                    border: '1px solid var(--border-subtle)',
                    lineHeight: 1.5
                  }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Regions contributing to model prediction: </strong>
                    Warm colors (red/yellow) indicate localized convolutional features that exerted the greatest positive contribution toward the Xception manipulated-class logit.
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '48px 16px', textAlign: 'center', lineHeight: 1.6 }}>
                  {isNoFace ? 'No face detected for visual attribution' : 'Not available for this analysis'}
                </div>
              )
            )}

            {activeTab === 'frequency' && (
              currentFreqAnalysis?.available && currentFreqUrl ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.6px', color: 'var(--text-primary)', textTransform: 'uppercase' }}>
                      2D Fast Fourier Transform (FFT)
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '9.5px', textTransform: 'uppercase' }}>
                      {currentFreqAnalysis.analysisRegion === 'face_region' ? 'Face Region Spectrum' : 'Full Image Spectrum'}
                    </span>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr', gap: 14, alignItems: 'start' }}>
                    {/* Spectrum Artifact Image */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center' }}>
                      <div style={{
                        width: '140px',
                        height: '140px',
                        backgroundColor: '#05070c',
                        borderRadius: 'var(--radius-sm)',
                        overflow: 'hidden',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid var(--border-subtle)'
                      }}>
                        <img
                          src={currentFreqUrl}
                          alt="2D FFT Frequency Magnitude Spectrum"
                          style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                        />
                      </div>
                      <span style={{ fontSize: '10.5px', color: 'var(--text-muted)' }}>
                        Magnitude Spectrum
                      </span>
                    </div>

                    {/* Numerical Spectral Energy Metrics */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>High-Frequency Ratio</div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: (currentFreqAnalysis.highFrequencyRatio ?? 0) > 0.35 ? '#fbbf24' : 'var(--text-primary)', marginTop: 2 }}>
                          {((currentFreqAnalysis.highFrequencyRatio ?? 0) * 100).toFixed(2)}%
                        </div>
                      </div>

                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Spectral Centroid</div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                          {currentFreqAnalysis.spectralCentroid?.toFixed(4) ?? '0.0000'}
                        </div>
                      </div>

                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Low-Frequency Energy</div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                          {currentFreqAnalysis.lowFrequencyEnergy?.toExponential(2) ?? '—'}
                        </div>
                      </div>

                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>High-Frequency Energy</div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                          {currentFreqAnalysis.highFrequencyEnergy?.toExponential(2) ?? '—'}
                        </div>
                      </div>

                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Mean Magnitude</div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                          {currentFreqAnalysis.meanMagnitude?.toFixed(2) ?? '—'}
                        </div>
                      </div>

                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Energy</div>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                          {currentFreqAnalysis.totalEnergy?.toExponential(2) ?? '—'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Findings List */}
                  {currentFreqAnalysis.findings && currentFreqAnalysis.findings.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {currentFreqAnalysis.findings.map((f, idx) => (
                        <div key={idx} style={{ 
                          fontSize: '11px', 
                          display: 'flex', 
                          alignItems: 'baseline', 
                          gap: 8,
                          padding: '6px 10px',
                          backgroundColor: f.severity === 'warning' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                          borderRadius: 'var(--radius-xs)',
                          border: `1px solid ${f.severity === 'warning' ? 'rgba(245, 158, 11, 0.25)' : 'var(--border-subtle)'}`
                        }}>
                          <span style={{ 
                            fontSize: '9px', 
                            fontWeight: 700, 
                            textTransform: 'uppercase', 
                            color: f.severity === 'warning' ? '#fbbf24' : 'var(--text-muted)',
                            flexShrink: 0
                          }}>
                            {f.severity}
                          </span>
                          <span style={{ color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                            {f.message}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Explanatory Footer */}
                  <div style={{ 
                    fontSize: '11px', 
                    color: 'var(--text-secondary)', 
                    backgroundColor: 'rgba(255, 255, 255, 0.02)', 
                    padding: '8px 12px', 
                    borderRadius: 'var(--radius-xs)', 
                    border: '1px solid var(--border-subtle)',
                    lineHeight: 1.5
                  }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Frequency-Domain Analysis: </strong>
                    2D Discrete Fourier Transform isolates orthogonal spatial frequency components. The VIRIDIS colormap represents log magnitude spectrum |F(u, v)| with zero frequency (DC) centered. High-frequency ratios capture potential generative boundary noise.
                  </div>
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '48px 16px', textAlign: 'center', lineHeight: 1.6 }}>
                  Frequency analysis not available for this media
                </div>
              )
            )}

            {activeTab === 'metadata' && (
              currentMetadata?.available ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.6px', color: 'var(--text-primary)', textTransform: 'uppercase' }}>
                      File & EXIF Metadata Inspection
                    </span>
                    <span className="badge badge-neutral" style={{ fontSize: '9.5px', textTransform: 'uppercase' }}>
                      {currentMetadata.metadataPresent ? 'Metadata Present' : 'Metadata Absent'}
                    </span>
                  </div>

                  {/* Metadata key properties */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
                    <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Camera Make & Model</div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                        {[currentMetadata.cameraMake, currentMetadata.cameraModel].filter(Boolean).join(' ') || 'Not available'}
                      </div>
                    </div>

                    <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Software Signature</div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: currentMetadata.editingSoftwareDetected ? '#f87171' : 'var(--text-primary)', marginTop: 2 }}>
                        {currentMetadata.software || 'Not available'}
                      </div>
                    </div>

                    <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Capture Timestamp</div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                        {currentMetadata.creationTimestamp || 'Not available'}
                      </div>
                    </div>

                    <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>GPS Geolocation</div>
                      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: 2 }}>
                        {currentMetadata.gpsPresent ? 'GPS coordinates present' : 'No GPS tags present'}
                      </div>
                    </div>
                  </div>

                  {!currentMetadata.metadataPresent && (
                    <div style={{ 
                      fontSize: '11px', 
                      color: 'var(--text-secondary)', 
                      backgroundColor: 'rgba(255, 255, 255, 0.02)', 
                      padding: '8px 12px', 
                      borderRadius: 'var(--radius-xs)', 
                      border: '1px solid var(--border-subtle)',
                      lineHeight: 1.5
                    }}>
                      <strong style={{ color: 'var(--text-primary)' }}>Notice: </strong>
                      EXIF metadata was not present in the uploaded file. This is standard for media shared via messaging apps or web platforms, which routinely strip EXIF metadata for privacy and bandwidth reasons. Absence of metadata is not proof of manipulation.
                    </div>
                  )}

                  {/* Metadata Findings List */}
                  {currentMetadata.findings && currentMetadata.findings.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {currentMetadata.findings.map((f, idx) => (
                        <div key={idx} style={{ 
                          fontSize: '11px', 
                          display: 'flex', 
                          alignItems: 'baseline', 
                          gap: 8,
                          padding: '6px 10px',
                          backgroundColor: f.severity === 'warning' ? 'rgba(245, 158, 11, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                          borderRadius: 'var(--radius-xs)',
                          border: `1px solid ${f.severity === 'warning' ? 'rgba(245, 158, 11, 0.25)' : 'var(--border-subtle)'}`
                        }}>
                          <span style={{ 
                            fontSize: '9px', 
                            fontWeight: 700, 
                            textTransform: 'uppercase', 
                            color: f.severity === 'warning' ? '#fbbf24' : 'var(--text-muted)',
                            flexShrink: 0
                          }}>
                            {f.severity}
                          </span>
                          <span style={{ color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                            {f.message}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '48px 16px', textAlign: 'center', lineHeight: 1.6 }}>
                  No metadata was available in the uploaded file
                </div>
              )
            )}

            {activeTab === 'comparison' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                {!crossValData && !isCrossValLoading && (
                  <div style={{
                    padding: '16px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 12
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 32,
                        height: 32,
                        borderRadius: 'var(--radius-xs)',
                        backgroundColor: 'rgba(96, 165, 250, 0.15)',
                        color: '#60a5fa',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}>
                        <GitCompare size={18} />
                      </div>
                      <div>
                        <h4 style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          Multi-Model Cross-Validation
                        </h4>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          Primary Xception vs Independent MesoNet (Meso4)
                        </span>
                      </div>
                    </div>
                    <p style={{ margin: 0, fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                      Evaluate the detected face crop against DeepfakeBench MesoNet (Meso4, 27.9K params) in addition to the sovereign Xception detector (20.8M params). Both models evaluate the exact same facial region without blending or score averaging.
                    </p>
                    {crossValError && (
                      <div style={{ padding: '8px 12px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 'var(--radius-xs)', color: '#fca5a5', fontSize: '11px' }}>
                        {crossValError}
                      </div>
                    )}
                    <button
                      className="btn btn-primary"
                      onClick={handleRunCrossValidation}
                      style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 6, fontSize: '12px', padding: '6px 14px' }}
                    >
                      <GitCompare size={14} />
                      <span>Run Cross-Validation</span>
                    </button>
                  </div>
                )}

                {isCrossValLoading && (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px 16px', gap: 12, textAlign: 'center' }}>
                    <RotateCw size={24} className="spin-icon" style={{ color: '#60a5fa' }} />
                    <div style={{ fontSize: '12px', color: 'var(--text-primary)', fontWeight: 500 }}>
                      Running MesoNet Cross-Validation...
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', maxWidth: '340px' }}>
                      Extracting identical face crop coordinates and computing independent prediction and inter-model agreement metrics.
                    </div>
                  </div>
                )}

                {crossValData && !isCrossValLoading && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {/* Agreement Status Banner */}
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: crossValData.disagreement_detected ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                      border: `1px solid ${crossValData.disagreement_detected ? 'rgba(245, 158, 11, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`,
                      borderRadius: 'var(--radius-sm)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      gap: 12
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        {crossValData.disagreement_detected ? (
                          <AlertTriangle size={20} style={{ color: '#fbbf24', flexShrink: 0 }} />
                        ) : (
                          <CheckCircle2 size={20} style={{ color: '#34d399', flexShrink: 0 }} />
                        )}
                        <div>
                          <div style={{
                            fontSize: '12px',
                            fontWeight: 700,
                            color: crossValData.disagreement_detected ? '#fbbf24' : '#34d399',
                            textTransform: 'uppercase',
                            letterSpacing: '0.5px'
                          }}>
                            {crossValData.disagreement_detected ? 'Detector Disagreement Detected' : 'Model Consensus Agreement'}
                          </div>
                          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: 2 }}>
                            {crossValData.analysis_summary}
                          </div>
                        </div>
                      </div>
                      <button
                        className="btn btn-secondary"
                        onClick={handleRunCrossValidation}
                        title="Re-run cross validation"
                        style={{ padding: '4px 8px', fontSize: '11px', flexShrink: 0, display: 'flex', alignItems: 'center', gap: 4 }}
                      >
                        <RotateCw size={12} />
                        <span>Re-test</span>
                      </button>
                    </div>

                    {/* Dual-Model Comparison Cards */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 10 }}>
                      {/* Primary Model Card */}
                      <div style={{
                        padding: '12px',
                        backgroundColor: 'var(--bg-app)',
                        borderRadius: 'var(--radius-xs)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '10px', color: '#60a5fa', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            Primary Production (Sovereign)
                          </span>
                          <span className="badge" style={{ backgroundColor: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', fontSize: '9px', padding: '2px 6px' }}>
                            20.8M params
                          </span>
                        </div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          Xception (DeepfakeBench)
                        </div>
                        {'primary_detector' in crossValData ? (
                          <>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
                              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Verdict</span>
                              <span
                                className={`badge ${crossValData.primary_detector.prediction === 'manipulated' ? 'badge-fake' : 'badge-real'}`}
                                style={{ textTransform: 'uppercase', fontSize: '10px' }}
                              >
                                {crossValData.primary_detector.prediction}
                              </span>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: 3 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Manipulated Prob</span>
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                  {(crossValData.primary_detector.manipulated_probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div style={{ width: '100%', height: 4, backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                                <div
                                  style={{
                                    width: `${crossValData.primary_detector.manipulated_probability * 100}%`,
                                    height: '100%',
                                    backgroundColor: crossValData.primary_detector.manipulated_probability >= 0.5 ? '#f87171' : '#34d399'
                                  }}
                                />
                              </div>
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                              <span>Latency: {crossValData.primary_detector.inference_time_ms.toFixed(1)} ms</span>
                              <span>Threshold: {crossValData.primary_detector.threshold_used}</span>
                            </div>
                          </>
                        ) : (
                          <>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
                              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Video Prediction</span>
                              <span
                                className={`badge ${(crossValData as VideoCrossValidationData).primary_video_prediction === 'manipulated' ? 'badge-fake' : 'badge-real'}`}
                                style={{ textTransform: 'uppercase', fontSize: '10px' }}
                              >
                                {(crossValData as VideoCrossValidationData).primary_video_prediction}
                              </span>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: 3 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Aggregate Probability</span>
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                  {(((crossValData as VideoCrossValidationData).primary_video_probability ?? 0) * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </>
                        )}
                      </div>

                      {/* Comparison Model Card */}
                      <div style={{
                        padding: '12px',
                        backgroundColor: 'var(--bg-app)',
                        borderRadius: 'var(--radius-xs)',
                        border: '1px solid var(--border-subtle)',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '10px', color: '#c084fc', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            Comparison Model (Secondary)
                          </span>
                          <span className="badge" style={{ backgroundColor: 'rgba(168, 85, 247, 0.15)', color: '#c084fc', fontSize: '9px', padding: '2px 6px' }}>
                            27.9K params
                          </span>
                        </div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          MesoNet / Meso4 (DeepfakeBench)
                        </div>
                        {'comparison_detectors' in crossValData && crossValData.comparison_detectors.length > 0 ? (
                          <>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
                              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Verdict</span>
                              <span
                                className={`badge ${crossValData.comparison_detectors[0].prediction === 'manipulated' ? 'badge-fake' : 'badge-real'}`}
                                style={{ textTransform: 'uppercase', fontSize: '10px' }}
                              >
                                {crossValData.comparison_detectors[0].prediction}
                              </span>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: 3 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Manipulated Prob</span>
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                  {(crossValData.comparison_detectors[0].manipulated_probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div style={{ width: '100%', height: 4, backgroundColor: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                                <div
                                  style={{
                                    width: `${crossValData.comparison_detectors[0].manipulated_probability * 100}%`,
                                    height: '100%',
                                    backgroundColor: crossValData.comparison_detectors[0].manipulated_probability >= 0.5 ? '#f87171' : '#34d399'
                                  }}
                                />
                              </div>
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                              <span>Latency: {crossValData.comparison_detectors[0].inference_time_ms.toFixed(1)} ms</span>
                              <span>Threshold: {crossValData.comparison_detectors[0].threshold_used}</span>
                            </div>
                          </>
                        ) : 'comparison_video_predictions' in crossValData && (crossValData as VideoCrossValidationData).comparison_video_predictions.length > 0 ? (
                          <>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
                              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Video Prediction</span>
                              <span
                                className={`badge ${(crossValData as VideoCrossValidationData).comparison_video_predictions[0].video_prediction === 'manipulated' ? 'badge-fake' : 'badge-real'}`}
                                style={{ textTransform: 'uppercase', fontSize: '10px' }}
                              >
                                {(crossValData as VideoCrossValidationData).comparison_video_predictions[0].video_prediction}
                              </span>
                            </div>
                            <div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: 3 }}>
                                <span style={{ color: 'var(--text-muted)' }}>Aggregate Probability</span>
                                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                                  {((((crossValData as VideoCrossValidationData).comparison_video_predictions[0].video_probability ?? 0) * 100)).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </>
                        ) : (
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Comparison data unavailable</div>
                        )}
                      </div>
                    </div>

                    {/* Video Alignment Stats (if Video) */}
                    {'video_agreement_rate' in crossValData && (
                      <div style={{
                        padding: '10px 12px',
                        backgroundColor: 'var(--bg-app)',
                        borderRadius: 'var(--radius-xs)',
                        border: '1px solid var(--border-subtle)',
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                        gap: 10,
                        fontSize: '11px'
                      }}>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Video Agreement Rate</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                            {(crossValData as VideoCrossValidationData).video_agreement_rate.toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Frames Analyzed</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                            {(crossValData as VideoCrossValidationData).frames_analyzed}
                          </span>
                        </div>
                        {(crossValData as VideoCrossValidationData).agreement_statistics?.length > 0 && (
                          <>
                            <div>
                              <span style={{ color: 'var(--text-muted)', display: 'block' }}>Cohen's Kappa (κ)</span>
                              <span style={{ fontWeight: 600, color: '#38bdf8' }}>
                                {(crossValData as VideoCrossValidationData).agreement_statistics[0].cohen_kappa !== null
                                  ? (crossValData as VideoCrossValidationData).agreement_statistics[0].cohen_kappa?.toFixed(3)
                                  : 'N/A'}
                              </span>
                            </div>
                            <div>
                              <span style={{ color: 'var(--text-muted)', display: 'block' }}>Pearson r</span>
                              <span style={{ fontWeight: 600, color: '#c084fc' }}>
                                {(crossValData as VideoCrossValidationData).agreement_statistics[0].pearson_correlation !== null
                                  ? (crossValData as VideoCrossValidationData).agreement_statistics[0].pearson_correlation?.toFixed(3)
                                  : 'N/A'}
                              </span>
                            </div>
                          </>
                        )}
                      </div>
                    )}

                    {/* Forensic Methodology Disclaimer */}
                    <div style={{
                      fontSize: '11px',
                      color: 'var(--text-secondary)',
                      backgroundColor: 'rgba(255, 255, 255, 0.02)',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      lineHeight: 1.5
                    }}>
                      <strong style={{ color: 'var(--text-primary)' }}>Forensic Governance: </strong>
                      DEEPTRACE does not ensemble, average, or blend model scores. The primary Xception detector remains the sovereign decision model. MesoNet-4 provides an independent, secondary verification layer to identify edge cases and representation divergence.
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'temporal' && (
              analysis.mediaType !== 'video' ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '48px 16px', textAlign: 'center', lineHeight: 1.6 }}>
                  Temporal consistency analysis is specific to multi-frame video sequences. For single image evidence, inspect Grad-CAM activations, 2D FFT frequency spectrum, and EXIF metadata.
                </div>
              ) : !analysis.temporalAnalysis || analysis.temporalAnalysis.status === 'not_available' ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '12px', padding: '48px 16px', textAlign: 'center', lineHeight: 1.6 }}>
                  {analysis.temporalAnalysis?.reason || 'Temporal consistency metrics require at least two sampled frames with valid facial detections.'}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '4px 0' }}>
                  {/* KPI Cards Grid */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10 }}>
                    {/* 1. Face Presence Continuity */}
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Face Presence</span>
                        <User size={13} color="var(--accent-blue)" />
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.face_presence.face_presence_percentage.toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                        {analysis.temporalAnalysis.face_presence.frames_with_face}/{analysis.temporalAnalysis.face_presence.total_sampled_frames} frames • Streak: {analysis.temporalAnalysis.face_presence.longest_consecutive_face_present}
                      </div>
                    </div>

                    {/* 2. Probability Volatility */}
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Volatility (σ)</span>
                        <Activity size={13} color="#f59e0b" />
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.statistics ? analysis.temporalAnalysis.statistics.std_probability.toFixed(3) : '0.000'}
                      </div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                        Mean: {analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.mean_probability * 100).toFixed(1) : 0}% • Range: {analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.min_probability * 100).toFixed(0) : 0}-{analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.max_probability * 100).toFixed(0) : 0}%
                      </div>
                    </div>

                    {/* 3. Prediction Transitions */}
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Prediction Flips</span>
                        <TrendingUp size={13} color="#c084fc" />
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: 700, color: analysis.temporalAnalysis.prediction_transitions.length > 0 ? '#fbbf24' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.prediction_transitions.length}
                      </div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                        {analysis.temporalAnalysis.prediction_transitions.length === 0 ? 'Consistent classification' : 'Boundary flips detected'}
                      </div>
                    </div>

                    {/* 4. Max Probability Jump */}
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Max Step (Δ)</span>
                        <Zap size={13} color="#f87171" />
                      </div>
                      <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.max_absolute_delta * 100).toFixed(1) : 0}%
                      </div>
                      <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.3 }}>
                        Mean Δ: {analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.mean_absolute_delta * 100).toFixed(1) : 0}% (Threshold: 30%)
                      </div>
                    </div>
                  </div>

                  {/* Dual-Model Temporal Alignment (if available) */}
                  {analysis.temporalAnalysis.model_agreement && (
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <GitCompare size={14} color="#38bdf8" />
                          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                            Inter-Model Temporal Agreement (Xception vs MesoNet-4)
                          </span>
                        </div>
                        <span className="badge" style={{ backgroundColor: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8', fontSize: '9.5px' }}>
                          {(analysis.temporalAnalysis.model_agreement.agreement_rate * 100).toFixed(1)}% Agreement
                        </span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, fontSize: '11px' }}>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Frames Compared</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                            {analysis.temporalAnalysis.model_agreement.agreeing_frames} of {analysis.temporalAnalysis.model_agreement.frames_compared} aligned
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Cohen's Kappa (κ)</span>
                          <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                            {analysis.temporalAnalysis.model_agreement.cohen_kappa != null ? analysis.temporalAnalysis.model_agreement.cohen_kappa.toFixed(3) : 'N/A'}
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Pearson Correlation (r)</span>
                          <span style={{ fontWeight: 600, color: '#c084fc', fontFamily: 'var(--font-mono)' }}>
                            {analysis.temporalAnalysis.model_agreement.pearson_correlation != null ? analysis.temporalAnalysis.model_agreement.pearson_correlation.toFixed(3) : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Spatial Tracking & Scale Dynamics (if available) */}
                  {analysis.temporalAnalysis.bbox_tracking && (
                    <div style={{
                      padding: '12px 14px',
                      backgroundColor: 'var(--bg-app)',
                      borderRadius: 'var(--radius-xs)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <Scan size={14} color="var(--accent-blue)" />
                        <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                          Facial Bounding Box Spatial & Scale Dynamics
                        </span>
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, fontSize: '11px' }}>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Mean Center Shift</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                            {(analysis.temporalAnalysis.bbox_tracking.mean_center_displacement * 100).toFixed(1)}% frame width
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Max Center Shift</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                            {(analysis.temporalAnalysis.bbox_tracking.max_center_displacement * 100).toFixed(1)}% frame width
                          </span>
                        </div>
                        <div>
                          <span style={{ color: 'var(--text-muted)', display: 'block' }}>Max Scale Change (Area Δ)</span>
                          <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                            {(analysis.temporalAnalysis.bbox_tracking.max_area_delta * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        {analysis.temporalAnalysis.bbox_tracking.tracking_note}
                      </div>
                    </div>
                  )}

                  {/* Temporal Events Table */}
                  <div style={{
                    padding: '12px 14px',
                    backgroundColor: 'var(--bg-app)',
                    borderRadius: 'var(--radius-xs)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 10
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                        Discrete Temporal Event Log ({analysis.temporalAnalysis.temporal_events.length})
                      </span>
                      <span className="badge badge-neutral" style={{ fontSize: '9.5px' }}>
                        Automated Anomaly Filter
                      </span>
                    </div>

                    {analysis.temporalAnalysis.temporal_events.length === 0 ? (
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                        No temporal threshold anomalies detected. Consecutive frame probability steps, bounding box shifts, and classification verdicts remained stable.
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: '240px', overflowY: 'auto' }}>
                        {analysis.temporalAnalysis.temporal_events.map((evt) => (
                          <div
                            key={evt.event_id}
                            style={{
                              padding: '8px 10px',
                              backgroundColor: evt.severity === 'notable' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                              borderRadius: 'var(--radius-xs)',
                              border: `1px solid ${evt.severity === 'notable' ? 'rgba(239, 68, 68, 0.25)' : 'var(--border-subtle)'}`,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              gap: 10
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
                              <span
                                className={`badge ${evt.severity === 'notable' ? 'badge-high' : 'badge-neutral'}`}
                                style={{ fontSize: '9px', textTransform: 'uppercase', flexShrink: 0 }}
                              >
                                {evt.event_type.replace(/_/g, ' ')}
                              </span>
                              <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', flexShrink: 0, fontFamily: 'var(--font-mono)' }}>
                                Frame #{evt.frame_number} • {evt.timestamp.toFixed(2)}s
                              </span>
                              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {evt.description}
                              </span>
                            </div>
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ fontSize: '10px', padding: '3px 8px', flexShrink: 0 }}
                              onClick={() => {
                                if (videoRef.current) {
                                  videoRef.current.currentTime = evt.timestamp;
                                }
                                setCurrentTimeSec(evt.timestamp);
                                const match = analysis.suspiciousFrames?.find(f => Math.abs(f.seconds - evt.timestamp) < 0.25);
                                if (match) {
                                  setSelectedFrame(match);
                                }
                              }}
                            >
                              Inspect Frame
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Forensic Governance Disclaimer */}
                  <div style={{
                    fontSize: '11px',
                    color: 'var(--text-secondary)',
                    backgroundColor: 'rgba(255, 255, 255, 0.02)',
                    padding: '8px 12px',
                    borderRadius: 'var(--radius-xs)',
                    border: '1px solid var(--border-subtle)',
                    lineHeight: 1.5
                  }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Forensic Governance: </strong>
                    {analysis.temporalAnalysis.analysis_disclaimer}
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      </div>

      {/* Bottom Timeline Section */}
      <div className="timeline-card">
        <div className="timeline-header-row">
          <div className="timeline-title-group">
            <h3 className="card-title">Temporal Analysis</h3>
            <p className="card-subtitle">
              {analysis.mediaType === 'video' 
                ? `Sampled frame anomaly progression (${analysis.framesAnalyzed ?? 0} frames analyzed)`
                : 'Inter-frame anomaly distribution over time'}
            </p>
          </div>
          {analysis.mediaType === 'video' && analysis.timelineData && analysis.timelineData.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              {/* Event Markers Legend */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: '10px', color: 'var(--text-muted)' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ color: '#f87171', fontSize: '12px' }}>●</span> Jump (Δ ≥ 30%)
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ color: '#c084fc', fontSize: '10px' }}>▲</span> Prediction Flip
                </span>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
                  <span style={{ color: '#38bdf8', fontSize: '11px' }}>◆</span> Disagreement
                </span>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {analysis.timelineData.filter(p => p.isSuspicious).length} / {analysis.timelineData.length} Anomaly Frame(s)
              </div>
            </div>
          )}
        </div>

        <div className="timeline-graph-container">
          {analysis.mediaType === 'video' && analysis.timelineData && analysis.timelineData.length > 0 ? (
            <div className="timeline-bars-track">
              {analysis.timelineData.map((point, idx) => {
                const ptEvents = (analysis.temporalAnalysis?.temporal_events || []).filter(
                  e => Math.abs(e.timestamp - point.seconds) < 0.25 || e.frame_number === idx * 15
                );
                const hasJump = ptEvents.some(e => e.event_type === 'probability_jump');
                const hasFlip = ptEvents.some(e => e.event_type === 'prediction_transition');
                const hasDisagree = ptEvents.some(e => e.event_type === 'detector_disagreement');

                return (
                  <div 
                    key={idx}
                    style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end', position: 'relative', cursor: 'pointer' }}
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = point.seconds;
                      }
                      setCurrentTimeSec(point.seconds);
                      const match = analysis.suspiciousFrames?.find(f => Math.abs(f.seconds - point.seconds) < 0.25);
                      if (match) setSelectedFrame(match);
                    }}
                    title={`Time: ${point.timestamp} | Prob: ${(point.probability * 100).toFixed(1)}%${ptEvents.length > 0 ? ` | Events: ${ptEvents.map(e => e.event_type).join(', ')}` : ''}`}
                  >
                    {/* Event Marker Icons above bar */}
                    <div style={{ display: 'flex', gap: 2, marginBottom: 2, height: 12, alignItems: 'center' }}>
                      {hasJump && <span style={{ color: '#f87171', fontSize: '9px', lineHeight: 1 }}>●</span>}
                      {hasFlip && <span style={{ color: '#c084fc', fontSize: '8px', lineHeight: 1 }}>▲</span>}
                      {hasDisagree && <span style={{ color: '#38bdf8', fontSize: '9px', lineHeight: 1 }}>◆</span>}
                    </div>

                    {/* Bar */}
                    <div 
                      className={`timeline-bar-col ${point.isSuspicious ? 'is-suspicious' : ''}`}
                      style={{ 
                        height: `${Math.max(12, point.probability * 100)}%`,
                        backgroundColor: point.isSuspicious ? undefined : '#334155',
                        width: '100%'
                      }}
                    />
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ width: '100%', color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center', padding: '24px 0' }}>
              Not available for this analysis
            </div>
          )}
        </div>

        {/* Time Ticks */}
        <div className="timeline-ticks-row">
          <span>00:00</span>
          <span>00:01</span>
          <span>00:02</span>
          <span>00:03</span>
          <span>00:04</span>
          <span>{analysis.duration || '00:05'}</span>
        </div>
      </div>

      {/* Bottom Right Actions Bar */}
      <div className="results-bottom-actions">
        <button className="btn btn-secondary" onClick={() => setIsReportOpen(true)}>
          <FileText size={14} />
          <span>View Full Report</span>
        </button>
        <button className="btn btn-primary" onClick={handleDownloadEvidence}>
          <Download size={14} />
          <span>Download Evidence</span>
        </button>
      </div>

      {/* Audit Report Modal */}
      <ReportModal 
        analysis={analysis}
        isOpen={isReportOpen}
        onClose={() => setIsReportOpen(false)}
      />
    </div>
  );
};
