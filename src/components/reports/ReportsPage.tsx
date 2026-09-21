import React, { useState } from 'react';
import { FileText, Download, Eye } from 'lucide-react';
import type { ForensicReport, Analysis } from '../../types/forensics';
import { ReportModal } from '../analysis/ReportModal';
import { forensicApiService } from '../../services/api';

interface ReportsPageProps {
  reports: ForensicReport[];
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ reports }) => {
  const [selectedReport, setSelectedReport] = useState<ForensicReport | null>(null);
  const [activeAnalysis, setActiveAnalysis] = useState<Analysis | null>(null);

  const handleViewReport = async (report: ForensicReport) => {
    setSelectedReport(report);
    try {
      const realAnalysis = await forensicApiService.getAnalysis(report.caseId);
      setActiveAnalysis(realAnalysis);
    } catch (err) {
      console.error('Failed to load analysis for report:', err);
      setActiveAnalysis({
        id: report.caseId,
        filename: report.filename,
        fileSize: 'Forensic File',
        mediaType: report.mediaType,
        mediaUrl: `/uploads/${report.caseId}_${report.filename}`,
        createdAt: report.created,
        status: 'completed',
        prediction: report.result,
        riskLevel: report.result === 'LIKELY MANIPULATED' ? 'High Risk' : report.result === 'REAL' ? 'Low Risk' : 'Inconclusive',
        confidenceScore: report.confidence,
        model: 'Xception (DeepfakeBench)',
        facesDetected: 1,
        suspiciousFrames: [],
        indicators: [],
        originalFrameUrl: `/uploads/${report.caseId}_${report.filename}`,
        heatmapUrl: '',
      });
    }
  };

  const handleDownloadReport = (report: ForensicReport) => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${report.id}_audit_report.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header-box">
        <h1 className="page-title">Forensic Reports</h1>
        <p className="page-subtitle">Official media verification audits, SHA-256 integrity hashes, and evidentiary findings.</p>
      </div>

      <div className="forensic-table-container">
        <table className="forensic-table">
          <thead>
            <tr>
              <th>Report ID</th>
              <th>Case ID</th>
              <th>Media</th>
              <th>Result</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {reports.length > 0 ? (
              reports.map((rep) => (
                <tr key={rep.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <FileText size={14} color="var(--accent-blue)" />
                      <span className="cell-primary cell-mono">{rep.id}</span>
                    </div>
                  </td>
                  <td className="cell-mono" style={{ color: 'var(--text-muted)' }}>{rep.caseId}</td>
                  <td className="cell-mono">{rep.filename}</td>
                  <td>
                    <span 
                      className={`badge ${
                        rep.result === 'LIKELY MANIPULATED'
                          ? 'badge-manipulated'
                          : rep.result === 'REAL'
                          ? 'badge-real'
                          : 'badge-inconclusive'
                      }`}
                    >
                      {rep.result}
                    </span>
                  </td>
                  <td className="cell-mono">{rep.created}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <button 
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleViewReport(rep)}
                        title="View Report"
                      >
                        <Eye size={12} />
                        <span>View</span>
                      </button>
                      <button 
                        className="btn btn-secondary btn-sm"
                        onClick={() => handleDownloadReport(rep)}
                        title="Download Report"
                      >
                        <Download size={12} />
                        <span>Download</span>
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '48px 16px', color: 'var(--text-muted)' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
                    <FileText size={28} color="var(--text-dim)" />
                    <div style={{ fontSize: '13.5px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                      No Forensic Reports Available
                    </div>
                    <div style={{ fontSize: '11.5px', color: 'var(--text-muted)', maxWidth: '420px', lineHeight: 1.5 }}>
                      Forensic verification reports are automatically compiled when images or videos are analyzed by the DEEPTRACE forensic engine.
                    </div>
                  </div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {activeAnalysis && (
        <ReportModal 
          analysis={activeAnalysis}
          isOpen={!!selectedReport}
          onClose={() => {
            setSelectedReport(null);
            setActiveAnalysis(null);
          }}
        />
      )}
    </div>
  );
};
