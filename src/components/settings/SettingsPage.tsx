import React, { useState } from 'react';
import { Check } from 'lucide-react';
import { API_BASE_URL } from '../../services/api';

export const SettingsPage: React.FC = () => {
  const [theme, setTheme] = useState('dark-navy');
  const [confidenceThreshold, setConfidenceThreshold] = useState(85);
  const [frameSampleRate, setFrameSampleRate] = useState(15);
  const [apiUrl, setApiUrl] = useState(API_BASE_URL);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSave = () => {
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, maxWidth: '820px' }}>
      <div className="page-header-box">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Configure application environment, inference thresholds, and backend API integration.</p>
      </div>

      {savedSuccess && (
        <div className="card" style={{ borderColor: 'var(--status-green)', backgroundColor: 'var(--status-green-subtle)', display: 'flex', alignItems: 'center', gap: 10, padding: '10px 14px' }}>
          <Check size={16} color="var(--status-green)" />
          <span style={{ color: 'var(--status-green)', fontSize: '12.5px', fontWeight: 500 }}>Preferences saved successfully.</span>
        </div>
      )}

      {/* 1. Appearance */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Appearance</h3>
            <p className="card-subtitle">Interface styling and forensic display themes</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 16 }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '12.5px' }}>
            <input 
              type="radio" 
              name="theme" 
              value="dark-navy" 
              checked={theme === 'dark-navy'} 
              onChange={() => setTheme('dark-navy')}
            />
            <span>Forensic Dark Navy (Standard)</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '12.5px' }}>
            <input 
              type="radio" 
              name="theme" 
              value="high-contrast" 
              checked={theme === 'high-contrast'} 
              onChange={() => setTheme('high-contrast')}
            />
            <span>High-Contrast Studio</span>
          </label>
        </div>
      </div>

      {/* 2. Analysis Preferences */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Analysis Preferences</h3>
            <p className="card-subtitle">Confidence thresholds, sampling frequency, and detection heuristics</p>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '12px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Manipulation Alert Threshold</span>
              <span className="mono-text" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{confidenceThreshold}%</span>
            </div>
            <input 
              type="range" 
              min="50" 
              max="99" 
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
            <span style={{ fontSize: '10.5px', color: 'var(--text-dim)', display: 'block', marginTop: 4 }}>
              Scores above this value trigger a "LIKELY MANIPULATED" flag.
            </span>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '12px' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Video Frame Sampling Rate</span>
              <span className="mono-text" style={{ color: 'var(--accent-blue)', fontWeight: 600 }}>{frameSampleRate} FPS</span>
            </div>
            <input 
              type="range" 
              min="5" 
              max="30" 
              step="5"
              value={frameSampleRate}
              onChange={(e) => setFrameSampleRate(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--accent-blue)' }}
            />
            <span style={{ fontSize: '10.5px', color: 'var(--text-dim)', display: 'block', marginTop: 4 }}>
              Extracts and feeds {frameSampleRate} frames per second through the detection pipeline.
            </span>
          </div>
        </div>
      </div>

      {/* 3. Backend & Model Preferences */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Backend API Integration</h3>
            <p className="card-subtitle">FastAPI ML inference server connection URL</p>
          </div>
        </div>

        <div>
          <label style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: 6 }}>
            ENDPOINT URL
          </label>
          <input 
            type="text"
            className="header-search-input"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            style={{
              width: '100%',
              backgroundColor: 'var(--bg-input)',
              border: '1px solid var(--border-card)',
              borderRadius: 'var(--radius-sm)',
              padding: '8px 12px',
              fontFamily: 'var(--font-mono)'
            }}
          />
          <span style={{ fontSize: '10.5px', color: 'var(--text-dim)', display: 'block', marginTop: 6 }}>
            Default local development server: http://127.0.0.1:8000/api/v1
          </span>
        </div>
      </div>

      {/* 4. About */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">About DEEPTRACE</h3>
            <p className="card-subtitle">Product specification and version information</p>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, fontSize: '11.5px' }}>
          <div>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Release Version</span>
            <span className="mono-text">v1.0.0-forensic-alpha</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Primary Engine</span>
            <span>Xception CNN + Grad-CAM</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-muted)', display: 'block' }}>Compliance</span>
            <span>Evidentiary Chain Verification</span>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn btn-primary" onClick={handleSave}>
          Save Preferences
        </button>
      </div>
    </div>
  );
};
