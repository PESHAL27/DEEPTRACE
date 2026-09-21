import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  FileCode,
  Play,
  RotateCw,
  Shield,
  Layers,
  Copy,
  Check,
  ChevronDown,
  GitCompare
} from 'lucide-react';
import { forensicApiService } from '../../services/api';
import type {
  BenchmarkRunSummaryItem,
  BenchmarkRunDetailData,
  BenchmarkTemplate,
  MultiModelBenchmarkComparisonData
} from '../../types/forensics';

export const BenchmarksPage: React.FC = () => {
  const [runs, setRuns] = useState<BenchmarkRunSummaryItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [activeRunDetail, setActiveRunDetail] = useState<BenchmarkRunDetailData | null>(null);
  const [template, setTemplate] = useState<BenchmarkTemplate | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [manifestPathInput, setManifestPathInput] = useState<string>('evaluation/manifests/template_manifest.json');
  const [execThreshold, setExecThreshold] = useState<number>(0.5);
  const [execError, setExecError] = useState<string | null>(null);
  const [copiedTemplate, setCopiedTemplate] = useState<boolean>(false);
  const [errorFilter, setErrorFilter] = useState<'all' | 'false_positive' | 'false_negative'>('all');

  // Stage 8E: Multi-model cross-validation benchmark state
  const [activeBenchmarkTab, setActiveBenchmarkTab] = useState<'primary' | 'multi_model'>('primary');
  const [multiModelComparison, setMultiModelComparison] = useState<MultiModelBenchmarkComparisonData | null>(null);
  const [isMultiModelExecuting, setIsMultiModelExecuting] = useState<boolean>(false);
  const [multiModelError, setMultiModelError] = useState<string | null>(null);
  const [multiModelManifestInput, setMultiModelManifestInput] = useState<string>('evaluation/manifests/template_manifest.json');
  const [multiModelThreshold, setMultiModelThreshold] = useState<number>(0.5);

  // Load initial benchmark runs and template
  const loadBenchmarkData = async () => {
    setIsLoading(true);
    try {
      const [runsList, tmpl] = await Promise.all([
        forensicApiService.getBenchmarkRuns(),
        forensicApiService.getBenchmarkTemplate()
      ]);
      setRuns(runsList);
      setTemplate(tmpl);

      if (runsList.length > 0) {
        const firstId = runsList[0].benchmark_run_id;
        setSelectedRunId(firstId);
        const detail = await forensicApiService.getBenchmarkRun(firstId);
        setActiveRunDetail(detail);
      } else {
        setSelectedRunId(null);
        setActiveRunDetail(null);
      }
    } catch (err: any) {
      console.error('Failed to load benchmark data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadBenchmarkData();
  }, []);

  const handleSelectRun = async (runId: string) => {
    setSelectedRunId(runId);
    setIsLoading(true);
    try {
      const detail = await forensicApiService.getBenchmarkRun(runId);
      setActiveRunDetail(detail);
    } catch (err) {
      console.error(`Failed to load benchmark run ${runId}:`, err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExecuteBenchmark = async () => {
    if (!manifestPathInput.trim()) return;
    setIsExecuting(true);
    setExecError(null);
    try {
      const detail = await forensicApiService.runBenchmark(manifestPathInput.trim(), undefined, execThreshold);
      await loadBenchmarkData();
      setSelectedRunId(detail.benchmark_run_id);
      setActiveRunDetail(detail);
    } catch (err: any) {
      setExecError(err.message || 'Failed to execute benchmark run.');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleRunMultiModelBenchmark = async () => {
    if (!multiModelManifestInput.trim()) return;
    setIsMultiModelExecuting(true);
    setMultiModelError(null);
    try {
      const res = await forensicApiService.runMultiModelBenchmark({
        manifestPath: multiModelManifestInput.trim(),
        threshold: multiModelThreshold
      });
      setMultiModelComparison(res);
    } catch (err: any) {
      setMultiModelError(err.message || 'Failed to execute multi-model benchmark comparison.');
    } finally {
      setIsMultiModelExecuting(false);
    }
  };

  const handleCopyTemplate = () => {
    if (!template) return;
    navigator.clipboard.writeText(JSON.stringify(template, null, 2));
    setCopiedTemplate(true);
    setTimeout(() => setCopiedTemplate(false), 2000);
  };

  const filteredErrors = (activeRunDetail?.errors || []).filter(e => {
    if (errorFilter === 'all') return true;
    return e.error_type === errorFilter;
  });

  return (
    <div className="models-page" style={{ paddingBottom: '3rem' }}>
      {/* Page Header */}
      <div className="section-header" style={{ marginBottom: '1.5rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.25rem' }}>
            <BarChart3 size={24} style={{ color: 'var(--accent-primary, #00F0FF)' }} />
            <h1 className="page-title" style={{ margin: 0, fontSize: '1.4rem', fontWeight: 600 }}>
              Detection Reliability & Benchmarking
            </h1>
          </div>
          <p className="page-subtitle" style={{ margin: 0, color: 'var(--text-secondary, #94A3B8)', fontSize: '0.85rem' }}>
            Reproducible supervised evaluation of the DeepfakeBench Xception detector on ground-truth labeled datasets.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            className="action-btn secondary"
            onClick={loadBenchmarkData}
            disabled={isLoading || isExecuting}
            title="Refresh Benchmarks"
            style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 0.85rem', fontSize: '0.8rem' }}
          >
            <RotateCw size={14} className={isLoading ? 'spin-icon' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Benchmark Mode Sub-Navigation Tabs */}
      <div style={{ display: 'flex', gap: '0.6rem', marginBottom: '1.5rem', borderBottom: '1px solid rgba(148, 163, 184, 0.15)', paddingBottom: '0.75rem' }}>
        <button
          className={`action-btn ${activeBenchmarkTab === 'primary' ? 'primary' : 'secondary'}`}
          onClick={() => setActiveBenchmarkTab('primary')}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.5rem 1.1rem', fontSize: '0.85rem' }}
        >
          <BarChart3 size={15} />
          <span>Primary Detector Evaluation (Xception)</span>
        </button>
        <button
          className={`action-btn ${activeBenchmarkTab === 'multi_model' ? 'primary' : 'secondary'}`}
          onClick={() => setActiveBenchmarkTab('multi_model')}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.5rem 1.1rem', fontSize: '0.85rem' }}
        >
          <GitCompare size={15} />
          <span>Multi-Model Cross Validation (Xception vs MesoNet)</span>
        </button>
      </div>

      {activeBenchmarkTab === 'primary' ? (
        <>
          {/* Model & Evaluation Hardware Header Card */}
          <div
            className="glass-card"
        style={{
          padding: '1.25rem 1.5rem',
          marginBottom: '1.5rem',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '1.25rem',
          borderLeft: '4px solid var(--accent-primary, #00F0FF)'
        }}
      >
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Underlying Detector
          </span>
          <div style={{ fontWeight: 600, color: 'var(--text-primary, #F8FAFC)', fontSize: '1rem', marginTop: '0.2rem' }}>
            Xception (DeepfakeBench)
          </div>
          <span style={{ fontSize: '0.75rem', color: '#38BDF8' }}>Official Checkpoint (xception_best.pth)</span>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Face Detection Subsystem
          </span>
          <div style={{ fontWeight: 600, color: 'var(--text-primary, #F8FAFC)', fontSize: '1rem', marginTop: '0.2rem' }}>
            OpenCV YuNet
          </div>
          <span style={{ fontSize: '0.75rem', color: '#10B981' }}>Tight crop (0.2 expansion) • 256x256</span>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Default Threshold (T)
          </span>
          <div style={{ fontWeight: 600, color: 'var(--text-primary, #F8FAFC)', fontSize: '1rem', marginTop: '0.2rem' }}>
            0.50 Decision Boundary
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)' }}>Continuous [0.10, 0.90] Sweep</span>
        </div>
        <div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Evaluation Ground Truth
          </span>
          <div style={{ fontWeight: 600, color: 'var(--text-primary, #F8FAFC)', fontSize: '1rem', marginTop: '0.2rem' }}>
            Strict Supervised Manifest
          </div>
          <span style={{ fontSize: '0.75rem', color: '#F59E0B' }}>Zero synthetic or fabricated labels</span>
        </div>
      </div>

      {/* Benchmark Execution & Run Selection Bar */}
      <div
        className="glass-card"
        style={{
          padding: '1.2rem 1.5rem',
          marginBottom: '1.5rem',
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}
      >
        {/* Run Selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flex: '1 1 320px' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary, #94A3B8)', whiteSpace: 'nowrap' }}>
            Select Run:
          </span>
          <div style={{ position: 'relative', width: '100%', maxWidth: '380px' }}>
            <select
              value={selectedRunId || ''}
              onChange={(e) => handleSelectRun(e.target.value)}
              disabled={runs.length === 0 || isLoading || isExecuting}
              style={{
                width: '100%',
                padding: '0.55rem 2rem 0.55rem 0.85rem',
                backgroundColor: 'rgba(15, 23, 42, 0.8)',
                color: 'var(--text-primary, #F8FAFC)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '6px',
                fontSize: '0.85rem',
                appearance: 'none',
                outline: 'none',
                cursor: runs.length > 0 ? 'pointer' : 'not-allowed'
              }}
            >
              {runs.length === 0 ? (
                <option value="">No benchmark runs available</option>
              ) : (
                runs.map((r) => (
                  <option key={r.benchmark_run_id} value={r.benchmark_run_id}>
                    {r.benchmark_run_id} — {r.dataset_name} ({r.sample_count} samples, Acc: {(r.accuracy * 100).toFixed(1)}%)
                  </option>
                ))
              )}
            </select>
            <ChevronDown
              size={16}
              style={{
                position: 'absolute',
                right: '0.75rem',
                top: '50%',
                transform: 'translateY(-50%)',
                color: 'var(--text-secondary, #94A3B8)',
                pointerEvents: 'none'
              }}
            />
          </div>
        </div>

        {/* Execute New Benchmark Trigger */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: '1 1 420px', justifyContent: 'flex-end' }}>
          <input
            type="text"
            value={manifestPathInput}
            onChange={(e) => setManifestPathInput(e.target.value)}
            placeholder="Path to manifest JSON..."
            disabled={isExecuting}
            style={{
              padding: '0.55rem 0.85rem',
              backgroundColor: 'rgba(15, 23, 42, 0.8)',
              color: 'var(--text-primary, #F8FAFC)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              borderRadius: '6px',
              fontSize: '0.85rem',
              width: '100%',
              maxWidth: '260px'
            }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>T:</span>
            <input
              type="number"
              min="0.1"
              max="0.9"
              step="0.05"
              value={execThreshold}
              onChange={(e) => setExecThreshold(parseFloat(e.target.value) || 0.5)}
              disabled={isExecuting}
              style={{
                width: '64px',
                padding: '0.55rem 0.4rem',
                backgroundColor: 'rgba(15, 23, 42, 0.8)',
                color: 'var(--text-primary, #F8FAFC)',
                border: '1px solid rgba(148, 163, 184, 0.2)',
                borderRadius: '6px',
                fontSize: '0.85rem',
                textAlign: 'center'
              }}
            />
          </div>
          <button
            className="action-btn primary"
            onClick={handleExecuteBenchmark}
            disabled={isExecuting || !manifestPathInput.trim()}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.55rem 1rem',
              fontSize: '0.85rem',
              whiteSpace: 'nowrap',
              backgroundColor: 'var(--accent-primary, #00F0FF)',
              color: '#0B0F19',
              fontWeight: 600
            }}
          >
            <Play size={14} fill={isExecuting ? 'none' : 'currentColor'} className={isExecuting ? 'spin-icon' : ''} />
            <span>{isExecuting ? 'Evaluating...' : 'Run Benchmark'}</span>
          </button>
        </div>
      </div>

      {execError && (
        <div
          style={{
            padding: '0.85rem 1.25rem',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '6px',
            color: '#F87171',
            fontSize: '0.85rem',
            marginBottom: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem'
          }}
        >
          <AlertTriangle size={18} />
          <span>{execError}</span>
        </div>
      )}

      {/* Main Content Area */}
      {runs.length === 0 && !activeRunDetail ? (
        /* Empty State: Evaluation Dataset Not Configured */
        <div
          className="glass-card"
          style={{
            padding: '2.5rem 2rem',
            textAlign: 'center',
            borderRadius: '8px',
            border: '1px dashed rgba(148, 163, 184, 0.3)'
          }}
        >
          <div
            style={{
              width: '56px',
              height: '56px',
              borderRadius: '50%',
              backgroundColor: 'rgba(0, 240, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.2rem auto',
              color: 'var(--accent-primary, #00F0FF)'
            }}
          >
            <FileCode size={28} />
          </div>
          <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-primary, #F8FAFC)' }}>
            Evaluation Dataset Not Configured
          </h2>
          <p
            style={{
              maxWidth: '620px',
              margin: '0 auto 1.5rem auto',
              color: 'var(--text-secondary, #94A3B8)',
              fontSize: '0.9rem',
              lineHeight: 1.5
            }}
          >
            No evaluated benchmark runs are currently recorded. DEEPTRACE adheres to rigorous forensic ethics: accuracy,
            F1, ROC-AUC, and error rates are never fabricated. To generate verified metrics, provide a dataset manifest
            following the official specification below.
          </p>

          {/* Manifest Specification Card */}
          <div
            style={{
              textAlign: 'left',
              maxWidth: '780px',
              margin: '0 auto',
              backgroundColor: 'rgba(10, 15, 29, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.15)',
              borderRadius: '8px',
              overflow: 'hidden'
            }}
          >
            <div
              style={{
                padding: '0.75rem 1rem',
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: '#38BDF8' }}>
                <FileCode size={14} />
                <span>backend/evaluation/manifests/template_manifest.json</span>
              </div>
              <button
                onClick={handleCopyTemplate}
                style={{
                  background: 'none',
                  border: 'none',
                  color: copiedTemplate ? '#10B981' : 'var(--text-secondary, #94A3B8)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  fontSize: '0.75rem'
                }}
              >
                {copiedTemplate ? <Check size={14} /> : <Copy size={14} />}
                <span>{copiedTemplate ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>
            <pre
              style={{
                padding: '1rem',
                margin: 0,
                fontSize: '0.75rem',
                fontFamily: 'monospace',
                color: '#E2E8F0',
                overflowX: 'auto',
                maxHeight: '280px'
              }}
            >
              {template ? JSON.stringify(template, null, 2) : 'Loading manifest specification template...'}
            </pre>
          </div>
        </div>
      ) : activeRunDetail ? (
        /* Populated Benchmark Detail View */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
          {/* Executive Metrics Cards Grid */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Shield size={16} style={{ color: 'var(--accent-primary, #00F0FF)' }} />
                <h3 style={{ fontSize: '1rem', fontWeight: 600, margin: 0, color: 'var(--text-primary, #F8FAFC)' }}>
                  Executive Performance Metrics ({activeRunDetail.dataset_name})
                </h3>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)' }}>
                Evaluated on {activeRunDetail.timestamp} ({activeRunDetail.sample_count} samples in {activeRunDetail.total_runtime_seconds}s)
              </span>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                gap: '1rem'
              }}
            >
              {/* Accuracy */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #00F0FF' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  Accuracy
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {(activeRunDetail.metrics.accuracy * 100).toFixed(1)}%
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Overall correctness</span>
              </div>

              {/* F1 Score */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #38BDF8' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  F1-Score
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {activeRunDetail.metrics.f1_score.toFixed(3)}
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Harmonic mean P & R</span>
              </div>

              {/* Precision */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #10B981' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  Precision (PPV)
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {(activeRunDetail.metrics.precision * 100).toFixed(1)}%
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>TP / (TP + FP)</span>
              </div>

              {/* Recall / Sensitivity */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #A855F7' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  Recall (Sensitivity)
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {(activeRunDetail.metrics.recall * 100).toFixed(1)}%
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Manipulated caught</span>
              </div>

              {/* Specificity */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #6366F1' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  Specificity (TNR)
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {(activeRunDetail.metrics.specificity * 100).toFixed(1)}%
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Authentic protected</span>
              </div>

              {/* ROC-AUC */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #F59E0B' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  ROC-AUC
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {activeRunDetail.metrics.roc_auc !== null
                    ? activeRunDetail.metrics.roc_auc.toFixed(3)
                    : 'N/A'}
                </div>
                <span style={{ fontSize: '0.75rem', color: activeRunDetail.metrics.roc_auc !== null ? '#10B981' : '#F59E0B' }}>
                  {activeRunDetail.metrics.roc_auc !== null ? 'Threshold-independent' : 'Single-class dataset'}
                </span>
              </div>

              {/* Latency */}
              <div className="glass-card" style={{ padding: '1rem', borderTop: '3px solid #EC4899' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)', textTransform: 'uppercase' }}>
                  Avg Latency
                </span>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                  {activeRunDetail.avg_inference_time_ms.toFixed(1)} ms
                </div>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>Per sample inference</span>
              </div>
            </div>
          </div>

          {/* Confusion Matrix & Subgroup Analysis Row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
            {/* 2x2 Confusion Matrix */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                  Confusion Matrix (T = {activeRunDetail.threshold.toFixed(2)})
                </h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)' }}>
                  Total Evaluated: {activeRunDetail.sample_count}
                </span>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '100px 1fr 1fr',
                  gridTemplateRows: 'auto 1fr 1fr',
                  gap: '6px',
                  textAlign: 'center',
                  fontSize: '0.8rem'
                }}
              >
                {/* Header labels */}
                <div style={{ padding: '6px' }}></div>
                <div style={{ padding: '6px', fontWeight: 600, color: '#38BDF8', backgroundColor: 'rgba(56, 189, 248, 0.1)', borderRadius: '4px' }}>
                  Pred Authentic
                </div>
                <div style={{ padding: '6px', fontWeight: 600, color: '#F87171', backgroundColor: 'rgba(248, 113, 113, 0.1)', borderRadius: '4px' }}>
                  Pred Manipulated
                </div>

                {/* Actual Authentic Row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, color: '#38BDF8' }}>
                  Act. Authentic
                </div>
                {/* True Negative */}
                <div
                  style={{
                    padding: '1.2rem 0.5rem',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    borderRadius: '6px'
                  }}
                >
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#10B981' }}>
                    {activeRunDetail.confusion_matrix.tn}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#6EE7B7', fontWeight: 500 }}>
                    True Negatives (TN)
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94A3B8', marginTop: '2px' }}>
                    {(activeRunDetail.confusion_matrix.tn_rate * 100).toFixed(1)}% rate
                  </div>
                </div>
                {/* False Positive */}
                <div
                  style={{
                    padding: '1.2rem 0.5rem',
                    backgroundColor: 'rgba(239, 68, 68, 0.15)',
                    border: '1px solid rgba(239, 68, 68, 0.3)',
                    borderRadius: '6px'
                  }}
                >
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#EF4444' }}>
                    {activeRunDetail.confusion_matrix.fp}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#FCA5A5', fontWeight: 500 }}>
                    False Positives (FP)
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94A3B8', marginTop: '2px' }}>
                    {(activeRunDetail.confusion_matrix.fp_rate * 100).toFixed(1)}% rate
                  </div>
                </div>

                {/* Actual Manipulated Row */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 600, color: '#F87171' }}>
                  Act. Manipulated
                </div>
                {/* False Negative */}
                <div
                  style={{
                    padding: '1.2rem 0.5rem',
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    border: '1px solid rgba(245, 158, 11, 0.3)',
                    borderRadius: '6px'
                  }}
                >
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#F59E0B' }}>
                    {activeRunDetail.confusion_matrix.fn}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#FCD34D', fontWeight: 500 }}>
                    False Negatives (FN)
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94A3B8', marginTop: '2px' }}>
                    {(activeRunDetail.confusion_matrix.fn_rate * 100).toFixed(1)}% rate
                  </div>
                </div>
                {/* True Positive */}
                <div
                  style={{
                    padding: '1.2rem 0.5rem',
                    backgroundColor: 'rgba(0, 240, 255, 0.15)',
                    border: '1px solid rgba(0, 240, 255, 0.3)',
                    borderRadius: '6px'
                  }}
                >
                  <div style={{ fontSize: '1.4rem', fontWeight: 700, color: '#00F0FF' }}>
                    {activeRunDetail.confusion_matrix.tp}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#67E8F9', fontWeight: 500 }}>
                    True Positives (TP)
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94A3B8', marginTop: '2px' }}>
                    {(activeRunDetail.confusion_matrix.tp_rate * 100).toFixed(1)}% rate
                  </div>
                </div>
              </div>
            </div>

            {/* Subgroup Analysis Table */}
            <div className="glass-card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                  Subgroup Reliability Breakdown
                </h4>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)' }}>
                  Category Performance
                </span>
              </div>

              {activeRunDetail.subgroup_metrics.length === 0 ? (
                <p style={{ color: '#94A3B8', fontSize: '0.85rem' }}>No subgroup metadata available in manifest.</p>
              ) : (
                <div style={{ overflowX: 'auto', maxHeight: '240px' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.2)', color: '#94A3B8' }}>
                        <th style={{ padding: '6px 8px' }}>Category</th>
                        <th style={{ padding: '6px 8px' }}>Value</th>
                        <th style={{ padding: '6px 8px' }}>Samples</th>
                        <th style={{ padding: '6px 8px' }}>Accuracy</th>
                        <th style={{ padding: '6px 8px' }}>F1</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeRunDetail.subgroup_metrics.map((sg, idx) => (
                        <tr key={idx} style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.08)' }}>
                          <td style={{ padding: '6px 8px', color: '#38BDF8', textTransform: 'capitalize' }}>
                            {sg.category.replace('_', ' ')}
                          </td>
                          <td style={{ padding: '6px 8px', color: '#F8FAFC', fontWeight: 500 }}>
                            {sg.subgroup_value}
                          </td>
                          <td style={{ padding: '6px 8px', color: '#94A3B8' }}>{sg.total_samples}</td>
                          <td style={{ padding: '6px 8px', color: sg.accuracy >= 0.8 ? '#10B981' : '#F59E0B' }}>
                            {(sg.accuracy * 100).toFixed(1)}%
                          </td>
                          <td style={{ padding: '6px 8px', color: '#E2E8F0' }}>{sg.f1.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          {/* Threshold Sweep Table (0.10 to 0.90) */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Layers size={16} style={{ color: 'var(--accent-primary, #00F0FF)' }} />
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                  Decision Threshold Sweep (0.10 to 0.90)
                </h4>
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary, #94A3B8)' }}>
                Highlight indicates default operating threshold (0.50)
              </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'center' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.2)', color: '#94A3B8' }}>
                    <th style={{ padding: '8px' }}>Threshold (T)</th>
                    <th style={{ padding: '8px' }}>Accuracy</th>
                    <th style={{ padding: '8px' }}>Precision</th>
                    <th style={{ padding: '8px' }}>Recall</th>
                    <th style={{ padding: '8px' }}>Specificity</th>
                    <th style={{ padding: '8px' }}>F1-Score</th>
                    <th style={{ padding: '8px' }}>FPR</th>
                    <th style={{ padding: '8px' }}>FNR</th>
                    <th style={{ padding: '8px' }}>TP</th>
                    <th style={{ padding: '8px' }}>FP</th>
                    <th style={{ padding: '8px' }}>TN</th>
                    <th style={{ padding: '8px' }}>FN</th>
                  </tr>
                </thead>
                <tbody>
                  {activeRunDetail.threshold_sweep.map((row) => {
                    const isDefault = Math.abs(row.threshold - 0.50) < 0.01;
                    return (
                      <tr
                        key={row.threshold}
                        style={{
                          borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
                          backgroundColor: isDefault ? 'rgba(0, 240, 255, 0.1)' : 'transparent',
                          fontWeight: isDefault ? 600 : 400
                        }}
                      >
                        <td style={{ padding: '8px', color: isDefault ? '#00F0FF' : '#F8FAFC' }}>
                          {row.threshold.toFixed(2)} {isDefault && '★'}
                        </td>
                        <td style={{ padding: '8px', color: row.accuracy >= 0.8 ? '#10B981' : '#F8FAFC' }}>
                          {(row.accuracy * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px', color: '#E2E8F0' }}>{(row.precision * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px', color: '#E2E8F0' }}>{(row.recall * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px', color: '#E2E8F0' }}>{(row.specificity * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px', color: isDefault ? '#38BDF8' : '#E2E8F0' }}>{row.f1.toFixed(3)}</td>
                        <td style={{ padding: '8px', color: row.fpr > 0.2 ? '#EF4444' : '#94A3B8' }}>
                          {(row.fpr * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px', color: row.fnr > 0.2 ? '#F59E0B' : '#94A3B8' }}>
                          {(row.fnr * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px', color: '#10B981' }}>{row.tp}</td>
                        <td style={{ padding: '8px', color: '#EF4444' }}>{row.fp}</td>
                        <td style={{ padding: '8px', color: '#10B981' }}>{row.tn}</td>
                        <td style={{ padding: '8px', color: '#F59E0B' }}>{row.fn}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Error Analysis Section (False Positives & False Negatives) */}
          <div className="glass-card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertTriangle size={16} style={{ color: '#F59E0B' }} />
                <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                  Supervised Error Analysis ({activeRunDetail.errors.length} misclassified sample(s))
                </h4>
              </div>

              {/* Filter Pills */}
              <div style={{ display: 'flex', gap: '0.4rem' }}>
                <button
                  className={`action-btn ${errorFilter === 'all' ? 'primary' : 'secondary'}`}
                  onClick={() => setErrorFilter('all')}
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
                >
                  All ({activeRunDetail.errors.length})
                </button>
                <button
                  className={`action-btn ${errorFilter === 'false_positive' ? 'primary' : 'secondary'}`}
                  onClick={() => setErrorFilter('false_positive')}
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem', color: errorFilter === 'false_positive' ? '#0B0F19' : '#FCA5A5' }}
                >
                  False Positives ({activeRunDetail.confusion_matrix.fp})
                </button>
                <button
                  className={`action-btn ${errorFilter === 'false_negative' ? 'primary' : 'secondary'}`}
                  onClick={() => setErrorFilter('false_negative')}
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem', color: errorFilter === 'false_negative' ? '#0B0F19' : '#FCD34D' }}
                >
                  False Negatives ({activeRunDetail.confusion_matrix.fn})
                </button>
              </div>
            </div>

            {filteredErrors.length === 0 ? (
              <div style={{ padding: '1.5rem', textAlign: 'center', color: '#94A3B8', fontSize: '0.85rem' }}>
                <CheckCircle2 size={24} style={{ color: '#10B981', margin: '0 auto 0.5rem auto' }} />
                <span>Zero classification errors detected for the selected filter!</span>
              </div>
            ) : (
              <div style={{ overflowX: 'auto', maxHeight: '320px' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.2)', color: '#94A3B8' }}>
                      <th style={{ padding: '8px' }}>Sample ID</th>
                      <th style={{ padding: '8px' }}>Filename</th>
                      <th style={{ padding: '8px' }}>Error Type</th>
                      <th style={{ padding: '8px' }}>Ground Truth</th>
                      <th style={{ padding: '8px' }}>Predicted</th>
                      <th style={{ padding: '8px' }}>Manipulated Prob</th>
                      <th style={{ padding: '8px' }}>Resolution</th>
                      <th style={{ padding: '8px' }}>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredErrors.map((err) => (
                      <tr key={err.sample_id} style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.08)' }}>
                        <td style={{ padding: '8px', color: '#38BDF8', fontFamily: 'monospace' }}>{err.sample_id}</td>
                        <td style={{ padding: '8px', color: '#F8FAFC' }}>{err.filename}</td>
                        <td style={{ padding: '8px' }}>
                          <span
                            style={{
                              padding: '2px 8px',
                              borderRadius: '4px',
                              fontSize: '0.7rem',
                              fontWeight: 600,
                              backgroundColor: err.error_type === 'false_positive' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                              color: err.error_type === 'false_positive' ? '#FCA5A5' : '#FCD34D'
                            }}
                          >
                            {err.error_type.replace('_', ' ').toUpperCase()}
                          </span>
                        </td>
                        <td style={{ padding: '8px', color: '#94A3B8', textTransform: 'capitalize' }}>{err.ground_truth}</td>
                        <td style={{ padding: '8px', color: '#E2E8F0', textTransform: 'capitalize' }}>{err.predicted_class}</td>
                        <td style={{ padding: '8px', color: '#F8FAFC', fontWeight: 600 }}>
                          {(err.manipulated_probability * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px', color: '#94A3B8' }}>{err.resolution || 'N/A'}</td>
                        <td style={{ padding: '8px', color: '#94A3B8' }}>{err.source || 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      ) : null}
        </>
      ) : (
        /* Stage 8E: Multi-Model Cross Validation Tab */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Multi-Model Overview Card */}
          <div
            className="glass-card"
            style={{
              padding: '1.25rem 1.5rem',
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
              gap: '1.25rem',
              borderLeft: '4px solid #a855f7'
            }}
          >
            <div>
              <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Primary Production Detector
              </span>
              <div style={{ fontWeight: 600, color: '#F8FAFC', fontSize: '1rem', marginTop: '0.2rem' }}>
                Xception (DeepfakeBench)
              </div>
              <span style={{ fontSize: '0.75rem', color: '#38BDF8' }}>Official Checkpoint (20.8M parameters)</span>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Independent Comparison Model
              </span>
              <div style={{ fontWeight: 600, color: '#F8FAFC', fontSize: '1rem', marginTop: '0.2rem' }}>
                MesoNet-4 (DeepfakeBench)
              </div>
              <span style={{ fontSize: '0.75rem', color: '#C084FC' }}>Official Checkpoint (27.9K parameters)</span>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Evaluation Alignment
              </span>
              <div style={{ fontWeight: 600, color: '#F8FAFC', fontSize: '1rem', marginTop: '0.2rem' }}>
                Identical Supervised Manifest
              </div>
              <span style={{ fontSize: '0.75rem', color: '#10B981' }}>Exact same face crops & bounding boxes</span>
            </div>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Statistical Metrics
              </span>
              <div style={{ fontWeight: 600, color: '#F8FAFC', fontSize: '1rem', marginTop: '0.2rem' }}>
                Agreement & Correlation
              </div>
              <span style={{ fontSize: '0.75rem', color: '#F59E0B' }}>Cohen's κ • Pearson r • Spearman ρ</span>
            </div>
          </div>

          {/* Multi-Model Benchmark Execution Card */}
          <div
            className="glass-card"
            style={{
              padding: '1.2rem 1.5rem',
              display: 'flex',
              flexWrap: 'wrap',
              gap: '1rem',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}
          >
            <div>
              <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                Run Comparative Evaluation
              </h3>
              <p style={{ margin: '2px 0 0 0', fontSize: '0.8rem', color: '#94A3B8' }}>
                Execute primary Xception and MesoNet-4 side-by-side on the specified ground-truth manifest.
              </p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flex: '1 1 420px', justifyContent: 'flex-end' }}>
              <input
                type="text"
                value={multiModelManifestInput}
                onChange={(e) => setMultiModelManifestInput(e.target.value)}
                placeholder="Path to manifest JSON..."
                disabled={isMultiModelExecuting}
                style={{
                  padding: '0.55rem 0.85rem',
                  backgroundColor: 'rgba(15, 23, 42, 0.8)',
                  color: '#F8FAFC',
                  border: '1px solid rgba(148, 163, 184, 0.2)',
                  borderRadius: '6px',
                  fontSize: '0.85rem',
                  width: '100%',
                  maxWidth: '260px'
                }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>T:</span>
                <input
                  type="number"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={multiModelThreshold}
                  onChange={(e) => setMultiModelThreshold(parseFloat(e.target.value) || 0.5)}
                  disabled={isMultiModelExecuting}
                  style={{
                    width: '64px',
                    padding: '0.55rem 0.4rem',
                    backgroundColor: 'rgba(15, 23, 42, 0.8)',
                    color: '#F8FAFC',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: '6px',
                    fontSize: '0.85rem',
                    textAlign: 'center'
                  }}
                />
              </div>
              <button
                className="action-btn primary"
                onClick={handleRunMultiModelBenchmark}
                disabled={isMultiModelExecuting || !multiModelManifestInput.trim()}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  padding: '0.55rem 1rem',
                  fontSize: '0.85rem',
                  whiteSpace: 'nowrap',
                  backgroundColor: '#a855f7',
                  color: '#FFFFFF',
                  fontWeight: 600
                }}
              >
                {isMultiModelExecuting ? (
                  <>
                    <RotateCw size={14} className="spin-icon" />
                    <span>Evaluating Models...</span>
                  </>
                ) : (
                  <>
                    <Play size={14} fill="#FFFFFF" />
                    <span>Run Evaluation</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {multiModelError && (
            <div
              style={{
                padding: '0.85rem 1.25rem',
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                borderRadius: '8px',
                color: '#FCA5A5',
                fontSize: '0.85rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem'
              }}
            >
              <AlertTriangle size={18} style={{ color: '#EF4444', flexShrink: 0 }} />
              <span>{multiModelError}</span>
            </div>
          )}

          {!multiModelComparison && !isMultiModelExecuting && (
            <div
              className="glass-card"
              style={{
                padding: '3rem 2rem',
                textAlign: 'center',
                color: '#94A3B8'
              }}
            >
              <GitCompare size={36} style={{ color: '#a855f7', margin: '0 auto 1rem auto' }} />
              <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#F8FAFC', fontWeight: 600 }}>
                No Multi-Model Benchmark Executed Yet
              </h3>
              <p style={{ margin: '0.5rem auto 1.5rem auto', maxWidth: '520px', fontSize: '0.85rem', lineHeight: 1.6 }}>
                Click "Run Evaluation" above to evaluate both DeepfakeBench Xception and MesoNet-4 side-by-side across the standard evaluation manifest to view comparative accuracy, F1-scores, Cohen's Kappa, and disagreement analysis.
              </p>
              <button
                className="action-btn primary"
                onClick={handleRunMultiModelBenchmark}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.6rem 1.25rem',
                  fontSize: '0.85rem',
                  backgroundColor: '#a855f7',
                  color: '#FFFFFF',
                  fontWeight: 600
                }}
              >
                <Play size={14} fill="#FFFFFF" />
                <span>Execute Multi-Model Benchmark</span>
              </button>
            </div>
          )}

          {multiModelComparison && (
            <>
              {/* Summary Metric Stats Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '1rem' }}>
                <div className="glass-card" style={{ padding: '1rem 1.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Samples Evaluated</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#F8FAFC', marginTop: '0.2rem' }}>
                    {multiModelComparison.sample_count}
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#38BDF8' }}>{multiModelComparison.dataset_name}</span>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Inter-Model Agreement</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#10B981', marginTop: '0.2rem' }}>
                    {multiModelComparison.agreement_statistics
                      ? `${(multiModelComparison.agreement_statistics.agreement_rate * 100).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#10B981' }}>
                    {multiModelComparison.agreement_statistics
                      ? `${multiModelComparison.agreement_statistics.agreement_count} / ${multiModelComparison.agreement_statistics.frames_compared} concordant`
                      : ''}
                  </span>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Disagreements Detected</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: (multiModelComparison.agreement_statistics?.disagreement_count ?? 0) > 0 ? '#F59E0B' : '#10B981', marginTop: '0.2rem' }}>
                    {multiModelComparison.agreement_statistics?.disagreement_count ?? 0}
                  </div>
                  <span style={{ fontSize: '0.75rem', color: (multiModelComparison.agreement_statistics?.disagreement_count ?? 0) > 0 ? '#F59E0B' : '#10B981' }}>
                    {(multiModelComparison.agreement_statistics?.disagreement_count ?? 0) > 0 ? 'Divergent representations' : 'Full model consensus'}
                  </span>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Cohen's Kappa (κ)</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#A855F7', marginTop: '0.2rem' }}>
                    {multiModelComparison.agreement_statistics?.cohen_kappa !== null && multiModelComparison.agreement_statistics?.cohen_kappa !== undefined
                      ? multiModelComparison.agreement_statistics.cohen_kappa.toFixed(3)
                      : 'N/A'}
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#A855F7' }}>Inter-rater reliability</span>
                </div>

                <div className="glass-card" style={{ padding: '1rem 1.25rem' }}>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase' }}>Pearson r</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#38BDF8', marginTop: '0.2rem' }}>
                    {multiModelComparison.agreement_statistics?.pearson_correlation !== null && multiModelComparison.agreement_statistics?.pearson_correlation !== undefined
                      ? multiModelComparison.agreement_statistics.pearson_correlation.toFixed(3)
                      : 'N/A'}
                  </div>
                  <span style={{ fontSize: '0.75rem', color: '#38BDF8' }}>Score correlation</span>
                </div>
              </div>

              {/* Comparative Performance Table */}
              <div className="glass-card" style={{ padding: '1.25rem 1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                    Comparative Model Evaluation Metrics (Evaluated on Identical Dataset)
                  </h4>
                  <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                    Decision Boundary: T = {multiModelComparison.evaluated_models[0]?.threshold ?? 0.5}
                  </span>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.2)', color: '#94A3B8' }}>
                        <th style={{ padding: '10px' }}>Model</th>
                        <th style={{ padding: '10px' }}>Role</th>
                        <th style={{ padding: '10px' }}>Threshold (T)</th>
                        <th style={{ padding: '10px' }}>Accuracy</th>
                        <th style={{ padding: '10px' }}>Precision</th>
                        <th style={{ padding: '10px' }}>Recall</th>
                        <th style={{ padding: '10px' }}>Specificity</th>
                        <th style={{ padding: '10px' }}>F1 Score</th>
                        <th style={{ padding: '10px' }}>FPR / FNR</th>
                        <th style={{ padding: '10px' }}>ROC-AUC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {multiModelComparison.evaluated_models.map((m) => {
                        const isPrimary = m.role === 'primary';
                        return (
                          <tr key={m.model_id} style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.08)' }}>
                            <td style={{ padding: '10px', color: '#F8FAFC', fontWeight: 600 }}>
                              {m.display_name}
                            </td>
                            <td style={{ padding: '10px' }}>
                              <span
                                style={{
                                  padding: '2px 8px',
                                  borderRadius: '4px',
                                  fontSize: '0.7rem',
                                  fontWeight: 600,
                                  backgroundColor: isPrimary ? 'rgba(59, 130, 246, 0.2)' : 'rgba(168, 85, 247, 0.2)',
                                  color: isPrimary ? '#60A5FA' : '#C084FC',
                                  textTransform: 'uppercase'
                                }}
                              >
                                {isPrimary ? 'Primary Production' : 'Comparison Model'}
                              </span>
                            </td>
                            <td style={{ padding: '10px', color: '#94A3B8', fontFamily: 'monospace' }}>
                              {m.threshold.toFixed(2)}
                            </td>
                            <td style={{ padding: '10px', color: '#F8FAFC', fontWeight: 600 }}>
                              {(m.accuracy * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: '#E2E8F0' }}>
                              {(m.precision * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: '#E2E8F0' }}>
                              {(m.recall * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: '#E2E8F0' }}>
                              {(m.specificity * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: '#10B981', fontWeight: 600 }}>
                              {(m.f1_score * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: '#94A3B8', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                              {(m.false_positive_rate * 100).toFixed(1)}% / {(m.false_negative_rate * 100).toFixed(1)}%
                            </td>
                            <td style={{ padding: '10px', color: (m.roc_auc !== null && m.roc_auc !== undefined) ? '#38BDF8' : '#94A3B8' }}>
                              {(m.roc_auc !== null && m.roc_auc !== undefined) ? (m.roc_auc * 100).toFixed(1) + '%' : 'N/A'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Forensic Governance & Scientific Rigor Card */}
              <div
                className="glass-card"
                style={{
                  padding: '1.25rem 1.5rem',
                  borderLeft: '4px solid #10b981',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Shield size={18} style={{ color: '#10B981' }} />
                  <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 600, color: '#F8FAFC' }}>
                    Forensic Validation Governance & Citations
                  </h4>
                </div>
                <p style={{ margin: 0, fontSize: '0.85rem', color: '#94A3B8', lineHeight: 1.6 }}>
                  <strong>Strict Non-Ensemble Rule: </strong>
                  DEEPTRACE deliberately prohibits automated ensemble blending, probability averaging, or majority voting. Cross-validation serves as an independent secondary audit to uncover feature blind spots and representation divergence. Primary Xception classification remains sovereign.
                </p>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', fontSize: '0.8rem', color: '#64748B', paddingTop: '0.5rem', borderTop: '1px solid rgba(148, 163, 184, 0.1)' }}>
                  <span><strong>DeepfakeBench:</strong> NeurIPS 2023 Datasets & Benchmarks Track</span>
                  <span><strong>Xception:</strong> Chollet, CVPR 2017</span>
                  <span><strong>MesoNet:</strong> Afchar et al., IEEE WIFS 2018</span>
                  <span><strong>Documentation:</strong> backend/evaluation/MODEL_SOURCES.md</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};
