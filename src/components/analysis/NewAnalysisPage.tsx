import React, { useState, useRef } from 'react';
import { 
  Upload, 
  Film, 
  Trash2, 
  Play, 
  CheckCircle2, 
  Loader2, 
  AlertCircle 
} from 'lucide-react';
import type { Analysis } from '../../types/forensics';
import { forensicApiService } from '../../services/api';
import type { UploadResponse } from '../../services/api';
import '../../styles/analysis.css';

interface NewAnalysisPageProps {
  onAnalysisComplete: (analysis: Analysis) => void;
  onCancel: () => void;
}

const SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'mp4', 'mov'];

const PIPELINE_STEPS = [
  'Uploading Evidence',
  'Preparing Media',
  'Detecting Faces',
  'Extracting Frames',
  'Running Detection Model',
  'Analyzing Evidence',
  'Preparing Results'
];

export const NewAnalysisPage: React.FC<NewAnalysisPageProps> = ({
  onAnalysisComplete,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileData, setFileData] = useState<UploadResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Analysis Progress States
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [progressPercent, setProgressPercent] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndSetFile = async (file: File) => {
    setErrorMsg(null);
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!ext || !SUPPORTED_EXTENSIONS.includes(ext)) {
      setErrorMsg(`Unsupported file format .${ext}. Supported: JPG, JPEG, PNG, WEBP, MP4, MOV.`);
      return;
    }

    try {
      setSelectedFile(file);
      const res = await forensicApiService.uploadMedia(file);
      setFileData(res);
    } catch (err: any) {
      setErrorMsg(err?.message || 'Failed to process media file. Please try again.');
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setFileData(null);
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleStartAnalysis = async () => {
    if (!fileData) return;
    setIsAnalyzing(true);
    setCurrentStepIndex(0);
    setProgressPercent(5);

    try {
      const result = await forensicApiService.startAnalysis(
        fileData,
        'xception-cnn',
        (stepIndex, _stepName, percent) => {
          setCurrentStepIndex(stepIndex);
          setProgressPercent(percent);
        }
      );

      // Brief pause to display completion state
      setTimeout(() => {
        setIsAnalyzing(false);
        onAnalysisComplete(result);
      }, 500);
    } catch (err: any) {
      setIsAnalyzing(false);
      setErrorMsg(err?.message || 'Analysis failed. Check model connectivity.');
    }
  };

  return (
    <div className="new-analysis-container">
      <div className="page-header-box">
        <h1 className="page-title">New Analysis</h1>
        <p className="page-subtitle">Upload media for AI-powered forensic analysis.</p>
      </div>

      {errorMsg && (
        <div className="card" style={{ borderColor: 'var(--status-red)', display: 'flex', alignItems: 'center', gap: 10 }}>
          <AlertCircle size={16} color="var(--status-red)" />
          <span style={{ color: 'var(--status-red)', fontSize: '12.5px' }}>{errorMsg}</span>
        </div>
      )}

      {!isAnalyzing ? (
        <>
          {/* Main Upload Dropzone */}
          {!selectedFile ? (
            <div
              className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                id="media-file-input"
                data-testid="media-file-input"
                type="file"
                style={{ display: 'none' }}
                accept=".jpg,.jpeg,.png,.webp,.mp4,.mov"
                onChange={handleFileChange}
              />
              <div className="upload-icon-circle">
                <Upload size={24} />
              </div>
              <div className="upload-title">DROP EVIDENCE HERE</div>
              <div className="upload-or">or</div>
              <button
                type="button"
                id="choose-file-button"
                className="btn btn-secondary btn-sm"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                Choose File
              </button>

              <div className="upload-formats-box">
                <span className="format-pill">JPG</span>
                <span className="format-pill">JPEG</span>
                <span className="format-pill">PNG</span>
                <span className="format-pill">WEBP</span>
                <span className="format-pill">MP4</span>
                <span className="format-pill">MOV</span>
              </div>

              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center', gap: 8, fontSize: '11px', color: 'var(--text-muted)' }}>
                <span>Test Fixtures:</span>
                <button
                  type="button"
                  id="btn-sample-real"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/known_real.jpg');
                    const blob = await res.blob();
                    const file = new File([blob], 'known_real.jpg', { type: 'image/jpeg' });
                    validateAndSetFile(file);
                  }}
                >
                  Authentic Sample
                </button>
                <button
                  type="button"
                  id="btn-sample-fake"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/known_fake.jpg');
                    const blob = await res.blob();
                    const file = new File([blob], 'known_fake.jpg', { type: 'image/jpeg' });
                    validateAndSetFile(file);
                  }}
                >
                  Deepfake Sample
                </button>
                <button
                  type="button"
                  id="btn-sample-noface"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/no_face.jpg');
                    const blob = await res.blob();
                    const file = new File([blob], 'no_face.jpg', { type: 'image/jpeg' });
                    validateAndSetFile(file);
                  }}
                >
                  Zero-Face Sample
                </button>
                <button
                  type="button"
                  id="btn-sample-fake-video"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px', color: '#f87171' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/test_fake_video.mp4');
                    const blob = await res.blob();
                    const file = new File([blob], 'test_fake_video.mp4', { type: 'video/mp4' });
                    validateAndSetFile(file);
                  }}
                >
                  Deepfake Video
                </button>
                <button
                  type="button"
                  id="btn-sample-real-video"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px', color: '#34d399' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/test_real_video.mp4');
                    const blob = await res.blob();
                    const file = new File([blob], 'test_real_video.mp4', { type: 'video/mp4' });
                    validateAndSetFile(file);
                  }}
                >
                  Authentic Video
                </button>
                <button
                  type="button"
                  id="btn-sample-noface-video"
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '3px 8px', fontSize: '10.5px' }}
                  onClick={async (e) => {
                    e.stopPropagation();
                    const res = await fetch('/assets/test_video.mp4');
                    const blob = await res.blob();
                    const file = new File([blob], 'test_video.mp4', { type: 'video/mp4' });
                    validateAndSetFile(file);
                  }}
                >
                  Zero-Face Video
                </button>
              </div>
            </div>
          ) : (
            /* Selected File Details & Verification Configuration */
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
              <div className="selected-file-card">
                <div className="file-preview-left">
                  <div className="file-thumb-preview">
                    {fileData?.mediaType === 'image' ? (
                      <img
                        src={fileData.previewUrl}
                        alt="Preview"
                        className="file-thumb-img"
                      />
                    ) : (
                      <Film size={22} color="#60a5fa" />
                    )}
                  </div>
                  <div className="file-meta-box">
                    <span className="file-meta-name">{fileData?.filename}</span>
                    <div className="file-meta-details">
                      <span className="badge badge-neutral">{fileData?.mediaType}</span>
                      <span className="cell-mono">{fileData?.fileSize}</span>
                    </div>
                  </div>
                </div>

                <button
                  className="btn btn-secondary btn-sm"
                  onClick={handleRemoveFile}
                  title="Remove selected file"
                >
                  <Trash2 size={14} />
                  <span>Remove</span>
                </button>
              </div>

              {/* Model Selection Config */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <label style={{ fontSize: '11.5px', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                    DETECTION MODEL
                  </label>
                  <select
                    style={{
                      width: '100%',
                      backgroundColor: 'var(--bg-input)',
                      border: '1px solid var(--border-card)',
                      color: 'var(--text-primary)',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '12.5px',
                      fontFamily: 'var(--font-sans)',
                    }}
                    defaultValue="xception-cnn"
                  >
                    <option value="xception-cnn">Xception CNN (Facial Forgery & Blending) [Ready]</option>
                    <option value="vit-base" disabled>Vision Transformer ViT-B/16 (Planned)</option>
                    <option value="temporal-lstm" disabled>Temporal Consistency Model (Planned)</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '11.5px', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
                    ANALYSIS PROFILE
                  </label>
                  <select
                    style={{
                      width: '100%',
                      backgroundColor: 'var(--bg-input)',
                      border: '1px solid var(--border-card)',
                      color: 'var(--text-primary)',
                      padding: '8px 12px',
                      borderRadius: 'var(--radius-sm)',
                      fontSize: '12.5px',
                      fontFamily: 'var(--font-sans)',
                    }}
                    defaultValue="full"
                  >
                    <option value="full">Deep Multi-Scale + Grad-CAM Attribution</option>
                    <option value="fast">Fast Verification (Keyframes Only)</option>
                  </select>
                </div>
              </div>

              {/* Action */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, marginTop: 8 }}>
                <button className="btn btn-secondary" onClick={handleRemoveFile}>
                  Cancel
                </button>
                <button className="btn btn-primary btn-lg" onClick={handleStartAnalysis}>
                  <Play size={15} fill="#fff" />
                  <span>BEGIN ANALYSIS</span>
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        /* Professional Forensic Analysis Progress Interface (Section 11) */
        <div className="analysis-pipeline-card">
          <div className="pipeline-header">
            <div className="pipeline-title">
              <Loader2 size={18} className="animate-spin" color="var(--accent-blue)" />
              <span>Running Forensic Pipeline</span>
            </div>
            <span className="pipeline-status-badge">
              Step {currentStepIndex + 1} of {PIPELINE_STEPS.length} ({progressPercent}%)
            </span>
          </div>

          <div className="pipeline-progress-bar-track">
            <div 
              className="pipeline-progress-bar-fill" 
              style={{ width: `${progressPercent}%` }} 
            />
          </div>

          <div className="pipeline-steps-list">
            {PIPELINE_STEPS.map((step, idx) => {
              const isCompleted = idx < currentStepIndex;
              const isRunning = idx === currentStepIndex;

              return (
                <div 
                  key={step} 
                  className={`pipeline-step-row ${
                    isRunning ? 'step-running' : isCompleted ? 'step-completed' : ''
                  }`}
                >
                  <div className="step-label-group">
                    <div 
                      className={`step-indicator-circle ${
                        isCompleted ? 'completed' : isRunning ? 'running' : 'waiting'
                      }`}
                    >
                      {isCompleted ? (
                        <CheckCircle2 size={13} strokeWidth={3} />
                      ) : isRunning ? (
                        <Loader2 size={11} className="animate-spin" />
                      ) : (
                        idx + 1
                      )}
                    </div>
                    <span className="step-name">{step}</span>
                  </div>

                  <span className="step-state-text">
                    {isCompleted ? 'Completed' : isRunning ? 'In progress...' : 'Pending'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
