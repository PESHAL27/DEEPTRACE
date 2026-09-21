import React from 'react';
import { Cpu } from 'lucide-react';
import type { ModelInfo } from '../../types/forensics';

interface ModelsPageProps {
  models: ModelInfo[];
}

export const ModelsPage: React.FC<ModelsPageProps> = ({ models }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header-box">
        <h1 className="page-title">Detection Models</h1>
        <p className="page-subtitle">
          Neural network architectures and deepfake classification models deployed in the DEEPTRACE pipeline.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 16 }}>
        {models.map((model) => {
          const isReady = model.status === 'Ready';

          return (
            <div 
              key={model.id} 
              className="card" 
              style={{ 
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'space-between',
                gap: 14,
                borderColor: isReady ? 'var(--border-card)' : 'var(--border-subtle)'
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <div 
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: 'var(--radius-sm)',
                        backgroundColor: isReady ? 'var(--accent-blue-subtle)' : 'rgba(255, 255, 255, 0.04)',
                        color: isReady ? '#60a5fa' : 'var(--text-muted)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                    >
                      <Cpu size={16} />
                    </div>
                    <div>
                      <h3 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        {model.name}
                      </h3>
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                        {model.category}
                      </span>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    {model.role === 'primary' && (
                      <span 
                        className="badge" 
                        style={{ 
                          backgroundColor: 'rgba(59, 130, 246, 0.15)', 
                          color: '#60a5fa', 
                          border: '1px solid rgba(59, 130, 246, 0.3)', 
                          textTransform: 'uppercase', 
                          fontSize: '10px' 
                        }}
                      >
                        Primary Production
                      </span>
                    )}
                    {model.role === 'comparison' && (
                      <span 
                        className="badge" 
                        style={{ 
                          backgroundColor: 'rgba(168, 85, 247, 0.15)', 
                          color: '#c084fc', 
                          border: '1px solid rgba(168, 85, 247, 0.3)', 
                          textTransform: 'uppercase', 
                          fontSize: '10px' 
                        }}
                      >
                        Comparison Model
                      </span>
                    )}
                    <span 
                      className={`badge ${isReady ? 'badge-real' : 'badge-neutral'}`}
                      style={{ textTransform: 'uppercase', fontSize: '10px' }}
                    >
                      {isReady ? 'Ready' : 'Planned'}
                    </span>
                  </div>
                </div>

                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: '8px 0' }}>
                  {model.description}
                </p>
              </div>

              <div 
                style={{ 
                  borderTop: '1px solid var(--border-subtle)', 
                  paddingTop: 12, 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr', 
                  gap: 10,
                  fontSize: '11px' 
                }}
              >
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Parameters</span>
                  <span className="mono-text" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                    {model.parameters}
                  </span>
                </div>
                <div>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Resolution</span>
                  <span className="mono-text" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                    {model.inputResolution}
                  </span>
                </div>
                {model.device && (
                  <div>
                    <span style={{ color: 'var(--text-muted)', display: 'block' }}>Active Device</span>
                    <span className="mono-text" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                      {model.device}
                    </span>
                  </div>
                )}
                {model.checkpoint && (
                  <div>
                    <span style={{ color: 'var(--text-muted)', display: 'block' }}>Checkpoint</span>
                    <span className="mono-text" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                      {model.checkpoint}
                    </span>
                  </div>
                )}
                <div style={{ gridColumn: 'span 2' }}>
                  <span style={{ color: 'var(--text-muted)', display: 'block' }}>Target Task</span>
                  <span style={{ color: '#60a5fa' }}>{model.purpose}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
