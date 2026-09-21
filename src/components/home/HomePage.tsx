import React from 'react';
import { 
  Upload, 
  PlayCircle, 
  Image as ImageIcon, 
  Film, 
  FileCheck2, 
  ArrowRight,
  ExternalLink 
} from 'lucide-react';
import type { Analysis } from '../../types/forensics';
import '../../styles/home.css';

interface HomePageProps {
  recentAnalyses: Analysis[];
  onNavigateToUpload: () => void;
  onNavigateToDemo: () => void;
  onSelectAnalysis: (analysis: Analysis) => void;
  onNavigateToHistory: () => void;
  onNavigateToReports: () => void;
}

export const HomePage: React.FC<HomePageProps> = ({
  recentAnalyses,
  onNavigateToUpload,
  onNavigateToDemo,
  onSelectAnalysis,
  onNavigateToHistory,
  onNavigateToReports,
}) => {
  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge">DIGITAL MEDIA FORENSICS</div>

          <h1 className="hero-title">
            Detect Manipulated Media.
            <span className="hero-title-accent">Reveal the Truth.</span>
          </h1>

          <p className="hero-description">
            DeepTrace uses advanced AI and forensic analysis to detect deepfakes in images and videos. 
            Make media verification simple, accurate and reliable.
          </p>

          <div className="hero-actions">
            <button className="btn btn-primary btn-lg" onClick={onNavigateToUpload}>
              <Upload size={16} />
              <span>Upload Evidence</span>
            </button>
            <button className="btn btn-secondary btn-lg" onClick={onNavigateToDemo}>
              <PlayCircle size={16} />
              <span>View Demo</span>
            </button>
          </div>

          <div className="hero-metrics">
            <div className="metric-item">
              <span className="metric-value">99%+</span>
              <span className="metric-label">Detection Accuracy</span>
              <span className="metric-sub">Benchmark Evaluated</span>
            </div>
            <div className="metric-item">
              <span className="metric-value">Multiple Models</span>
              <span className="metric-label">CNN + ViT + Temporal</span>
              <span className="metric-sub">Ensemble Architecture</span>
            </div>
            <div className="metric-item">
              <span className="metric-value">Image & Video</span>
              <span className="metric-label">All Major Formats</span>
              <span className="metric-sub">JPG, PNG, MP4, MOV</span>
            </div>
          </div>
        </div>

        {/* Hero Visual Area with Forensic HUD */}
        <div className="hero-visual-card">
          <img 
            src="/assets/hero_forensic_face.jpg" 
            alt="AI Forensic Media Inspection" 
            className="hero-visual-img"
          />

          {/* Forensic HUD Viewfinder Brackets */}
          <div className="hud-corner hud-top-left" />
          <div className="hud-corner hud-top-right" />
          <div className="hud-corner hud-bottom-left" />
          <div className="hud-corner hud-bottom-right" />

          {/* Forensic Real vs Fake Overlay Tag */}
          <div className="hud-tag">
            <span className="hud-tag-title">AI Forensics</span>
            <span className="hud-tag-subtitle">Real vs Fake</span>
          </div>
        </div>
      </section>

      {/* Quick Action Cards */}
      <section className="quick-actions-grid">
        <div className="quick-card" onClick={onNavigateToUpload}>
          <div className="quick-card-icon">
            <ImageIcon size={18} />
          </div>
          <div className="quick-card-title">Image Analysis</div>
          <div className="quick-card-desc">
            Detect potential manipulation, warping, and generative splicing in still images.
          </div>
          <div className="quick-card-link">
            <span>Analyze Image</span>
            <ArrowRight size={13} />
          </div>
        </div>

        <div className="quick-card" onClick={onNavigateToUpload}>
          <div className="quick-card-icon">
            <Film size={18} />
          </div>
          <div className="quick-card-title">Video Analysis</div>
          <div className="quick-card-desc">
            Analyze videos frame by frame with temporal consistency and facial geometry checks.
          </div>
          <div className="quick-card-link">
            <span>Analyze Video</span>
            <ArrowRight size={13} />
          </div>
        </div>

        <div className="quick-card" onClick={onNavigateToReports}>
          <div className="quick-card-icon">
            <FileCheck2 size={18} />
          </div>
          <div className="quick-card-title">Forensic Report</div>
          <div className="quick-card-desc">
            Generate detailed evidentiary analysis reports with visual model attribution.
          </div>
          <div className="quick-card-link">
            <span>View Reports</span>
            <ArrowRight size={13} />
          </div>
        </div>
      </section>

      {/* Recent Analysis Data Table */}
      <section className="recent-section">
        <div className="section-header">
          <h2 className="section-title">Recent Analysis</h2>
          <span className="section-link" onClick={onNavigateToHistory}>
            View All History →
          </span>
        </div>

        <div className="forensic-table-container">
          <table className="forensic-table">
            <thead>
              <tr>
                <th>File Name</th>
                <th>Type</th>
                <th>Result</th>
                <th>Probability</th>
                <th>Analyzed On</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {recentAnalyses.map((item) => (
                <tr key={item.id}>
                  <td className="cell-primary cell-mono">{item.filename}</td>
                  <td>
                    <span className="badge badge-neutral">{item.mediaType}</span>
                  </td>
                  <td>
                    <span 
                      className={`badge ${
                        item.prediction === 'LIKELY MANIPULATED'
                          ? 'badge-manipulated'
                          : item.prediction === 'REAL'
                          ? 'badge-real'
                          : 'badge-inconclusive'
                      }`}
                    >
                      {item.prediction}
                    </span>
                  </td>
                  <td className="cell-mono">
                    <span 
                      style={{ 
                        color: item.prediction === 'LIKELY MANIPULATED' ? '#f87171' : '#34d399',
                        fontWeight: 600
                      }}
                    >
                      {item.confidenceScore.toFixed(1)}%
                    </span>
                  </td>
                  <td className="cell-mono">{item.createdAt}</td>
                  <td>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => onSelectAnalysis(item)}
                    >
                      <span>View Analysis</span>
                      <ExternalLink size={12} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
