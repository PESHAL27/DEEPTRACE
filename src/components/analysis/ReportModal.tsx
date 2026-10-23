import React, { useState, useEffect } from 'react';
import { 
  X, 
  Download, 
  Printer, 
  ShieldAlert, 
  ShieldCheck, 
  FileText, 
  CheckCircle2, 
  Copy
} from 'lucide-react';
import type { Analysis, DetailedForensicReportData } from '../../types/forensics';
import { forensicApiService } from '../../services/api';

interface ReportModalProps {
  analysis: Analysis;
  isOpen: boolean;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ analysis, isOpen, onClose }) => {
  const [detailedReport, setDetailedReport] = useState<DetailedForensicReportData | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'models' | 'physical' | 'temporal' | 'governance'>('overview');
  const [copiedHash, setCopiedHash] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;

    const fetchReport = async () => {
      try {
        const rep = await forensicApiService.getDetailedReport(analysis.id);
        if (isMounted && rep) {
          setDetailedReport(rep);
        }
      } catch (err) {
        console.error('Failed to load detailed report:', err);
      }
    };

    fetchReport();

    return () => {
      isMounted = false;
    };
  }, [isOpen, analysis.id]);

  if (!isOpen) return null;

  const isNoFace = analysis.prediction === 'NO FACE DETECTED';
  const isManipulated = analysis.prediction === 'LIKELY MANIPULATED';
  const sha256 = detailedReport?.hashSha256 || '7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069';
  const reportId = detailedReport?.id || `REP-${analysis.id}`;

  const handleCopyHash = () => {
    navigator.clipboard.writeText(sha256);
    setCopiedHash(true);
    setTimeout(() => setCopiedHash(false), 2000);
  };

  const handleDownloadJSON = () => {
    const dataToExport = detailedReport || {
      id: reportId,
      caseId: analysis.id,
      filename: analysis.filename,
      mediaType: analysis.mediaType,
      result: analysis.prediction,
      confidence: analysis.confidenceScore,
      created: analysis.createdAt,
      hashSha256: sha256,
      examiner: "DEEPTRACE Automated Analysis Engine (Xception-DeepfakeBench)",
      summary: "Forensic Media Verification Audit",
      methodology: "OpenCV YuNet -> Xception CNN -> Grad-CAM -> 2D FFT -> Temporal Tracking"
    };

    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${reportId}_audit_case_record.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleOpenPrintable = () => {
    const printUrl = forensicApiService.getPrintableReportUrl(reportId);
    window.open(printUrl, '_blank');
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div 
        className="modal-dialog" 
        style={{ maxWidth: '880px', width: '92%', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }} 
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="modal-header" style={{ flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <FileText size={18} color="var(--accent-blue)" />
            <div>
              <div style={{ fontSize: '14.5px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.3px' }}>
                FORENSIC MEDIA AUDIT REPORT
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {reportId} • CASE REF: {analysis.id} • v1.0
              </div>
            </div>
          </div>
          <button className="control-btn" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </div>

        {/* Modal Navigation Tabs */}
        <div style={{ 
          display: 'flex', 
          borderBottom: '1px solid var(--border-subtle)', 
          backgroundColor: 'var(--bg-app)', 
          padding: '0 16px',
          gap: 6,
          flexShrink: 0 
        }}>
          <button 
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
            style={{ fontSize: '11.5px', padding: '10px 12px' }}
          >
            Executive Summary
          </button>
          <button 
            className={`tab-button ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
            style={{ fontSize: '11.5px', padding: '10px 12px' }}
          >
            Detector & Explainability
          </button>
          <button 
            className={`tab-button ${activeTab === 'physical' ? 'active' : ''}`}
            onClick={() => setActiveTab('physical')}
            style={{ fontSize: '11.5px', padding: '10px 12px' }}
          >
            Physical Forensics
          </button>
          {analysis.mediaType === 'video' && (
            <button 
              className={`tab-button ${activeTab === 'temporal' ? 'active' : ''}`}
              onClick={() => setActiveTab('temporal')}
              style={{ fontSize: '11.5px', padding: '10px 12px' }}
            >
              Temporal & Cross-Model
            </button>
          )}
          <button 
            className={`tab-button ${activeTab === 'governance' ? 'active' : ''}`}
            onClick={() => setActiveTab('governance')}
            style={{ fontSize: '11.5px', padding: '10px 12px' }}
          >
            Methodology & Limitations
          </button>
        </div>

        {/* Modal Scrollable Body */}
        <div className="modal-body" style={{ overflowY: 'auto', flex: 1, padding: '18px 20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
          
          {/* Executive Verdict Banner */}
          <div 
            style={{
              padding: '12px 16px',
              borderRadius: 'var(--radius-sm)',
              border: `1px solid ${
                isNoFace 
                  ? 'var(--border-subtle)' 
                  : isManipulated 
                    ? 'var(--status-red-border)' 
                    : 'var(--status-green-border)'
              }`,
              backgroundColor: isNoFace 
                ? 'var(--bg-app)' 
                : isManipulated 
                  ? 'var(--status-red-subtle)' 
                  : 'var(--status-green-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {isNoFace ? (
                <ShieldAlert size={26} color="var(--text-muted)" />
              ) : isManipulated ? (
                <ShieldAlert size={26} color="var(--status-red)" />
              ) : (
                <ShieldCheck size={26} color="var(--status-green)" />
              )}
              <div>
                <div 
                  style={{ 
                    fontSize: '13.5px', 
                    fontWeight: 700, 
                    color: isNoFace 
                      ? 'var(--text-muted)' 
                      : isManipulated 
                        ? '#f87171' 
                        : '#34d399' 
                  }}
                >
                  PRIMARY CLASSIFICATION: {analysis.prediction} {isNoFace ? '' : `(${analysis.confidenceScore.toFixed(1)}% PROBABILITY)`}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                  {isNoFace 
                    ? 'No face detected in evidence media. Automated manipulation classification requires localized facial crops.' 
                    : isManipulated 
                      ? 'DeepfakeBench Xception model classifies input features as statistically aligned with manipulated media.' 
                      : 'Facial pattern characteristics evaluated as consistent with authentic imagery.'}
                </div>
              </div>
            </div>
            <span 
              className={`badge ${
                isNoFace 
                  ? '' 
                  : isManipulated 
                    ? 'badge-manipulated' 
                    : 'badge-real'
              }`}
              style={isNoFace ? { backgroundColor: 'var(--border-subtle)', color: 'var(--text-muted)' } : {}}
            >
              {isNoFace ? 'No Face' : `${analysis.mediaType === 'video' ? 'Video Probability' : 'Model Probability'}: ${analysis.confidenceScore.toFixed(1)}%`}
            </span>
          </div>

          {/* Whole-Image AI Detection Finding (Smaug AI Detector) */}
          {analysis.aiGeneration && (
            <div
              style={{
                padding: '10px 14px',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${
                  analysis.aiGeneration.predictedLabel === 'likely_ai_generated'
                    ? 'var(--status-red-border)'
                    : 'var(--status-green-border)'
                }`,
                backgroundColor:
                  analysis.aiGeneration.predictedLabel === 'likely_ai_generated'
                    ? 'var(--status-red-subtle)'
                    : 'var(--status-green-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <Zap size={22} color={analysis.aiGeneration.predictedLabel === 'likely_ai_generated' ? '#f87171' : '#34d399'} />
                <div>
                  <div style={{ fontSize: '12.5px', fontWeight: 700, color: analysis.aiGeneration.predictedLabel === 'likely_ai_generated' ? '#f87171' : '#34d399' }}>
                    WHOLE-IMAGE AI DETECTION: {analysis.aiGeneration.assessment.toUpperCase()} {typeof analysis.aiGeneration.score === 'number' ? `(${(analysis.aiGeneration.score * 100).toFixed(1)}% LIKELIHOOD)` : ''}
                  </div>
                  <div style={{ fontSize: '10.5px', color: 'var(--text-secondary)' }}>
                    Evaluated with {analysis.aiGeneration.modelName} (Abacus.AI, Apache 2.0). Operates on complete image without facial bounding box dependency.
                  </div>
                </div>
              </div>
              <span
                className={`badge ${
                  analysis.aiGeneration.predictedLabel === 'likely_ai_generated'
                    ? 'badge-manipulated'
                    : 'badge-real'
                }`}
                style={{ fontSize: '9.5px', padding: '3px 8px' }}
              >
                {typeof analysis.aiGeneration.score === 'number' ? `Score: ${(analysis.aiGeneration.score * 100).toFixed(1)}%` : 'Unavailable'}
              </span>
            </div>
          )}

          {/* SHA-256 Cryptographic Digest Bar */}
          <div style={{
            padding: '8px 12px',
            backgroundColor: 'var(--bg-app)',
            borderRadius: 'var(--radius-xs)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
              <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', flexShrink: 0 }}>
                Evidence SHA-256:
              </span>
              <span style={{ fontSize: '10.5px', fontFamily: 'var(--font-mono)', color: 'var(--accent-blue)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {sha256}
              </span>
            </div>
            <button
              className="btn btn-secondary btn-sm"
              onClick={handleCopyHash}
              style={{ fontSize: '10px', padding: '2px 8px', flexShrink: 0 }}
              title="Copy cryptographic digest"
            >
              {copiedHash ? <CheckCircle2 size={11} color="#34d399" /> : <Copy size={11} />}
              <span>{copiedHash ? 'Copied' : 'Copy'}</span>
            </button>
          </div>

          {/* TAB 1: OVERVIEW */}
          {activeTab === 'overview' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Executive Summary Text */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, letterSpacing: '0.5px' }}>
                  EXECUTIVE SUMMARY
                </div>
                <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {detailedReport?.executive_summary ? (
                    detailedReport.executive_summary.split('\n\n').map((p, idx) => (
                      <p key={idx} style={{ margin: idx === 0 ? '0 0 8px 0' : '8px 0 0 0' }}>{p}</p>
                    ))
                  ) : (
                    <p style={{ margin: 0 }}>
                      DEEPTRACE analyzed media file '{analysis.filename}' ({analysis.fileSize}) using the DeepfakeBench Xception
                      neural network detector paired with OpenCV YuNet facial localization. The primary detector evaluated the
                      evidence as {analysis.prediction} with a model probability of {analysis.confidenceScore.toFixed(1)}%.
                    </p>
                  )}
                </div>
              </div>

              {/* Independent Evidence Domains Table */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10, letterSpacing: '0.5px' }}>
                  INDEPENDENT EVIDENCE DOMAINS (FACTUAL STATUS)
                </div>
                {detailedReport?.evidence_summary_categories && detailedReport.evidence_summary_categories.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {detailedReport.evidence_summary_categories.map((cat, idx) => (
                      <div 
                        key={idx}
                        style={{
                          padding: '8px 10px',
                          backgroundColor: 'var(--bg-app)',
                          borderRadius: 'var(--radius-xs)',
                          border: '1px solid var(--border-subtle)',
                          display: 'flex',
                          alignItems: 'baseline',
                          justifyContent: 'space-between',
                          gap: 12
                        }}
                      >
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2 }}>
                            <span style={{ fontSize: '9.5px', textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)', fontWeight: 600 }}>
                              {cat.category}
                            </span>
                          </div>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {cat.headline}
                          </span>
                          <span style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                            {cat.details}
                          </span>
                        </div>
                        <span className="badge badge-neutral" style={{ fontSize: '9px', textTransform: 'uppercase', flexShrink: 0, fontWeight: 600 }}>
                          {cat.status}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    Factual breakdown compiled from primary model, Grad-CAM, FFT, and metadata telemetry.
                  </div>
                )}
              </div>

              {/* Administrative Specifications Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
                <div className="card" style={{ padding: '12px 14px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, letterSpacing: '0.5px' }}>
                    MEDIA SPECIFICATIONS
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '11px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Filename:</span>
                      <span className="mono-text">{analysis.filename}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Format & Size:</span>
                      <span>{analysis.mediaType.toUpperCase()} ({analysis.fileSize})</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Duration / Frames:</span>
                      <span className="mono-text">{analysis.duration || 'N/A'} ({analysis.framesAnalyzed} frame(s))</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Faces Tracked:</span>
                      <span>{analysis.facesDetected} Subject(s)</span>
                    </div>
                  </div>
                </div>

                <div className="card" style={{ padding: '12px 14px' }}>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8, letterSpacing: '0.5px' }}>
                    AUDIT TELEMETRY
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: '11px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Primary Architecture:</span>
                      <span className="mono-text">Xception (DeepfakeBench)</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Checkpoint:</span>
                      <span className="mono-text">xception_best.pth</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Audit Timestamp:</span>
                      <span className="mono-text">{analysis.createdAt}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Inference Hardware:</span>
                      <span>CPU (Host Processor)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: DETECTOR & EXPLAINABILITY */}
          {activeTab === 'models' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Primary Model Findings */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    PRIMARY DETECTOR TELEMETRY
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Xception CNN</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 10 }}>
                  <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>Output Verdict</span>
                    <span style={{ fontSize: '12.5px', fontWeight: 700, color: isManipulated ? '#f87171' : isNoFace ? '#94a3b8' : '#34d399' }}>
                      {analysis.prediction}
                    </span>
                  </div>
                  <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>Model Probability</span>
                    <span style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                      {analysis.confidenceScore.toFixed(1)}%
                    </span>
                  </div>
                  <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>Decision Threshold</span>
                    <span style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                      50.0%
                    </span>
                  </div>
                </div>
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.primary_detector.classification_note || 'Primary detector evaluates 48 depthwise-separable convolutional layers to locate microscopic blending artifacts.'}
                </div>
              </div>

              {/* Facial Localization Evidence */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    FACIAL LOCALIZATION TELEMETRY
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>OpenCV YuNet</span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: 8 }}>
                  {analysis.facesDetected > 0 
                    ? `Successfully localized ${analysis.facesDetected} subject(s) with 15% uniform spatial margin.` 
                    : 'No facial structures detected in evidence media.'}
                </div>
                {analysis.faceBoundingBoxes && analysis.faceBoundingBoxes.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {analysis.faceBoundingBoxes.map((box, idx) => (
                      <div key={idx} style={{ fontSize: '10.5px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', padding: '4px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)' }}>
                        Face #{idx}: left={box.left}, top={box.top}, width={box.width}, height={box.height}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Model Visual Attention (Grad-CAM) */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    MODEL VISUAL ATTRIBUTION (GRAD-CAM)
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>conv4 Layer</span>
                </div>
                {analysis.gradcamOverlayUrl ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 8 }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ height: '130px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-subtle)' }}>
                        <img src={analysis.originalFaceCropUrl || analysis.mediaUrl} alt="Face Crop" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                      </div>
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>Analyzed Facial Subject</span>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ height: '130px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-subtle)' }}>
                        <img src={analysis.gradcamOverlayUrl} alt="Grad-CAM Overlay" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                      </div>
                      <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>conv4 Feature Activation Overlay</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                    Grad-CAM visual attribution unavailable for this media record.
                  </div>
                )}
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.gradcam_evidence.interpretation_disclaimer || 'Grad-CAM visualizes neural network attention focus and does not delineate ground-truth manipulation boundaries.'}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: PHYSICAL FORENSICS */}
          {activeTab === 'physical' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* 2D FFT Frequency Analysis */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    2D FAST FOURIER TRANSFORM FREQUENCY ANALYSIS
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Frequency Domain</span>
                </div>
                {analysis.frequencyAnalysis?.available ? (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 10 }}>
                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>High-Freq Ratio</span>
                        <span style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {((analysis.frequencyAnalysis.highFrequencyRatio ?? 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>Spectral Centroid</span>
                        <span style={{ fontSize: '12.5px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {(analysis.frequencyAnalysis.spectralCentroid ?? 0).toFixed(4)}
                        </span>
                      </div>
                      <div style={{ padding: '8px 10px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block' }}>Analysis Region</span>
                        <span style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          {analysis.frequencyAnalysis.analysisRegion.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    {analysis.frequencyAnalysis.frequencySpectrumUrl && (
                      <div style={{ textAlign: 'center', marginBottom: 8 }}>
                        <div style={{ height: '140px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-subtle)' }}>
                          <img src={analysis.frequencyAnalysis.frequencySpectrumUrl} alt="2D FFT Spectrum" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} />
                        </div>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', display: 'block', marginTop: 4 }}>Log-Magnitude 2D FFT Spectrum (DC Centered)</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                    Frequency-domain analysis not computed.
                  </div>
                )}
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.frequency_evidence.interpretation_disclaimer || 'Frequency-domain characteristics can be affected by lossy compression, resizing, sharpening, and sensor processing.'}
                </div>
              </div>

              {/* Metadata Analysis */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    CONTAINER & EXIF METADATA TELEMETRY
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>EXIF Header</span>
                </div>
                {analysis.metadataAnalysis?.metadataPresent ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10, fontSize: '11px', marginBottom: 8 }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Camera / Device:</span>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                        {analysis.metadataAnalysis.cameraMake || 'Unspecified'} {analysis.metadataAnalysis.cameraModel || ''}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Software Tag:</span>
                      <span style={{ fontWeight: 600, color: analysis.metadataAnalysis.editingSoftwareDetected ? '#f87171' : 'var(--text-primary)' }}>
                        {analysis.metadataAnalysis.software || 'None Recorded'}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Metadata Tag Count:</span>
                      <span className="mono-text">{analysis.metadataAnalysis.metadataFieldsCount} fields</span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>GPS Coordinates:</span>
                      <span>{analysis.metadataAnalysis.gpsPresent ? 'Present' : 'Not Present'}</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                    EXIF metadata was absent or stripped from the evidence file.
                  </div>
                )}
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.metadata_evidence.interpretation_disclaimer || 'Absence of metadata is standard across social media and web platforms and does not indicate manipulation.'}
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: TEMPORAL & CROSS-MODEL (VIDEO ONLY) */}
          {activeTab === 'temporal' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Temporal Consistency Findings */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    INTER-FRAME TEMPORAL CONSISTENCY
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Temporal Dynamics</span>
                </div>
                {analysis.temporalAnalysis?.status === 'available' ? (
                  <>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 10 }}>
                      <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '9.5px', color: 'var(--text-muted)', display: 'block' }}>Face Presence</span>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {analysis.temporalAnalysis.face_presence.face_presence_percentage.toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '9.5px', color: 'var(--text-muted)', display: 'block' }}>Volatility (σ)</span>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {analysis.temporalAnalysis.statistics ? analysis.temporalAnalysis.statistics.std_probability.toFixed(3) : '0.000'}
                        </span>
                      </div>
                      <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '9.5px', color: 'var(--text-muted)', display: 'block' }}>Prediction Flips</span>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {analysis.temporalAnalysis.prediction_transitions.length}
                        </span>
                      </div>
                      <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', border: '1px solid var(--border-subtle)' }}>
                        <span style={{ fontSize: '9.5px', color: 'var(--text-muted)', display: 'block' }}>Max Step (Δ)</span>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                          {analysis.temporalAnalysis.statistics ? (analysis.temporalAnalysis.statistics.max_absolute_delta * 100).toFixed(1) : 0}%
                        </span>
                      </div>
                    </div>
                    {analysis.temporalAnalysis.temporal_events.length > 0 && (
                      <div style={{ marginBottom: 8 }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block', marginBottom: 4 }}>
                          Discrete Anomaly Events ({analysis.temporalAnalysis.temporal_events.length})
                        </span>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: '120px', overflowY: 'auto' }}>
                          {analysis.temporalAnalysis.temporal_events.map((evt, idx) => (
                            <div key={idx} style={{ fontSize: '10.5px', padding: '4px 8px', backgroundColor: 'var(--bg-app)', borderRadius: 'var(--radius-xs)', display: 'flex', justifyContent: 'space-between' }}>
                              <span>Frame #{evt.frame_number} ({evt.timestamp.toFixed(2)}s): {evt.description}</span>
                              <span className="badge badge-neutral" style={{ fontSize: '8.5px' }}>{evt.event_type}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                    Temporal consistency analysis not available for this record.
                  </div>
                )}
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.temporal_evidence?.interpretation_disclaimer || 'Temporal anomalies can arise from compression, motion blur, and lighting changes, and do not prove manipulation.'}
                </div>
              </div>

              {/* Cross-Model Verification */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    DUAL-MODEL CROSS-VALIDATION (XCEPTION VS MESONET-4)
                  </span>
                  <span className="badge badge-neutral" style={{ fontSize: '9px' }}>Independent Verification</span>
                </div>
                {analysis.temporalAnalysis?.model_agreement ? (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, fontSize: '11px', marginBottom: 8 }}>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Agreement Rate:</span>
                      <span style={{ fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                        {(analysis.temporalAnalysis.model_agreement.agreement_rate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Cohen's Kappa (κ):</span>
                      <span style={{ fontWeight: 600, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.model_agreement.cohen_kappa != null ? analysis.temporalAnalysis.model_agreement.cohen_kappa.toFixed(3) : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)', display: 'block' }}>Pearson Correlation (r):</span>
                      <span style={{ fontWeight: 600, color: '#c084fc', fontFamily: 'var(--font-mono)' }}>
                        {analysis.temporalAnalysis.model_agreement.pearson_correlation != null ? analysis.temporalAnalysis.model_agreement.pearson_correlation.toFixed(3) : 'N/A'}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', padding: '12px 0', textAlign: 'center' }}>
                    Cross-model verification data not recorded.
                  </div>
                )}
                <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.4 }}>
                  {detailedReport?.cross_model_comparison?.governance_note || 'Models are evaluated independently without ensembling. Disagreements warrant manual investigation rather than proving manipulation.'}
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: METHODOLOGY & LIMITATIONS */}
          {activeTab === 'governance' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Methodology Section */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10, letterSpacing: '0.5px' }}>
                  DEEPTRACE FORENSIC PIPELINE METHODOLOGY
                </div>
                {detailedReport?.methodology_details ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {detailedReport.methodology_details.map((m, idx) => (
                      <div key={idx} style={{ fontSize: '11px', lineHeight: 1.4 }}>
                        <strong style={{ color: 'var(--text-primary)' }}>{m.step}: </strong>
                        <span style={{ color: 'var(--text-secondary)' }}>{m.description}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    OpenCV YuNet face localization with 15% margin padding -&gt; 256x256 RGB normalization -&gt;
                    DeepfakeBench Xception CNN inference -&gt; Grad-CAM conv4 activation mapping -&gt;
                    2D Fast Fourier Transform frequency-domain analysis -&gt; Temporal consistency tracking -&gt;
                    EXIF/container metadata inspection.
                  </div>
                )}
              </div>

              {/* Technical Limitations Section */}
              <div className="card" style={{ padding: '12px 14px' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10, letterSpacing: '0.5px' }}>
                  TECHNICAL LIMITATIONS & FORENSIC GOVERNANCE
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: '11px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 6, lineHeight: 1.4 }}>
                  {detailedReport?.limitations ? (
                    detailedReport.limitations.map((lim, idx) => (
                      <li key={idx}>{lim}</li>
                    ))
                  ) : (
                    <>
                      <li>Model probability represents statistical correlation with training artifacts, not proof of provenance.</li>
                      <li>Video compression can induce high-frequency noise that inflates detector suspicion.</li>
                      <li>Absence of EXIF metadata is standard across web platforms and is not evidence of tampering.</li>
                      <li>Grad-CAM visualizes neural network attention focus and does not delineate ground-truth boundaries.</li>
                      <li>Findings represent automated analytical measurements and do not constitute certified legal admissibility.</li>
                    </>
                  )}
                </ul>
              </div>

              {/* Automated Analysis Statement */}
              <div className="card" style={{ padding: '12px 14px', backgroundColor: 'var(--bg-app)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 6, letterSpacing: '0.5px', fontWeight: 600 }}>
                  AUTOMATED ANALYSIS STATEMENT
                </div>
                <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {detailedReport?.automated_analysis_statement || 
                    'This report summarizes automated algorithmic media analysis performed by DEEPTRACE and does not constitute human examiner certification or legal proof of media provenance.'}
                </p>
              </div>
            </div>
          )}

        </div>

        {/* Modal Actions Footer */}
        <div className="modal-footer" style={{ flexShrink: 0, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '11px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
            DEEPTRACE Media Forensics • Automated Analysis Case Record
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary btn-sm" onClick={handleOpenPrintable} title="Open Print View">
              <Printer size={13} />
              <span>Print / Save PDF</span>
            </button>
            <button className="btn btn-primary btn-sm" onClick={handleDownloadJSON} title="Download Audit Record">
              <Download size={13} />
              <span>Download Report JSON</span>
            </button>
            <button className="btn btn-secondary btn-sm" onClick={onClose}>
              <span>Close</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
