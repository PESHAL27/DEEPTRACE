import React, { useState } from 'react';
import { Search, ExternalLink, Film, Image as ImageIcon } from 'lucide-react';
import type { Analysis } from '../../types/forensics';

interface HistoryPageProps {
  analyses: Analysis[];
  onSelectAnalysis: (analysis: Analysis) => void;
}

export const HistoryPage: React.FC<HistoryPageProps> = ({ analyses, onSelectAnalysis }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState<'all' | 'image' | 'video' | 'manipulated' | 'real'>('all');

  const filtered = analyses.filter((item) => {
    // Search query match
    const query = searchTerm.toLowerCase();
    const matchesSearch = 
      item.filename.toLowerCase().includes(query) || 
      item.id.toLowerCase().includes(query);
    if (!matchesSearch) return false;

    // Filter match
    if (activeFilter === 'image') return item.mediaType === 'image';
    if (activeFilter === 'video') return item.mediaType === 'video';
    if (activeFilter === 'manipulated') return item.prediction === 'LIKELY MANIPULATED';
    if (activeFilter === 'real') return item.prediction === 'REAL';
    return true;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Page Header */}
      <div className="page-header-box">
        <h1 className="page-title">Analysis History</h1>
        <p className="page-subtitle">Historical case records and verified evidence audits.</p>
      </div>

      {/* Search & Filter Toolbar */}
      <div 
        style={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between', 
          gap: 16, 
          flexWrap: 'wrap' 
        }}
      >
        <div className="header-search" style={{ maxWidth: '360px' }}>
          <Search size={14} className="header-search-icon" />
          <input
            type="text"
            className="header-search-input"
            placeholder="Search by filename or case ID..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {/* Filter Pills */}
        <div className="tabs-header">
          <button 
            className={`tab-button ${activeFilter === 'all' ? 'active' : ''}`}
            onClick={() => setActiveFilter('all')}
          >
            All ({analyses.length})
          </button>
          <button 
            className={`tab-button ${activeFilter === 'video' ? 'active' : ''}`}
            onClick={() => setActiveFilter('video')}
          >
            Videos
          </button>
          <button 
            className={`tab-button ${activeFilter === 'image' ? 'active' : ''}`}
            onClick={() => setActiveFilter('image')}
          >
            Images
          </button>
          <button 
            className={`tab-button ${activeFilter === 'manipulated' ? 'active' : ''}`}
            onClick={() => setActiveFilter('manipulated')}
          >
            Likely Manipulated
          </button>
          <button 
            className={`tab-button ${activeFilter === 'real' ? 'active' : ''}`}
            onClick={() => setActiveFilter('real')}
          >
            Real
          </button>
        </div>
      </div>

      {/* History Data Table */}
      <div className="forensic-table-container">
        <table className="forensic-table">
          <thead>
            <tr>
              <th>Evidence Media</th>
              <th>Case ID</th>
              <th>Type</th>
              <th>Analyzed On</th>
              <th>Verdict</th>
              <th>Confidence</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length > 0 ? (
              filtered.map((item) => (
                <tr key={item.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div 
                        style={{ 
                          width: 36, 
                          height: 26, 
                          backgroundColor: '#03060c', 
                          borderRadius: 2, 
                          overflow: 'hidden',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          border: '1px solid var(--border-subtle)'
                        }}
                      >
                        {item.originalFrameUrl ? (
                          <img src={item.originalFrameUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        ) : item.mediaType === 'video' ? (
                          <Film size={14} color="#60a5fa" />
                        ) : (
                          <ImageIcon size={14} color="#60a5fa" />
                        )}
                      </div>
                      <span className="cell-primary cell-mono">{item.filename}</span>
                    </div>
                  </td>
                  <td className="cell-mono" style={{ color: 'var(--text-muted)' }}>{item.id}</td>
                  <td>
                    <span className="badge badge-neutral">{item.mediaType}</span>
                  </td>
                  <td className="cell-mono">{item.createdAt}</td>
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
              ))
            ) : (
              <tr>
                <td colSpan={7} style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-muted)' }}>
                  No forensic cases found matching your criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
