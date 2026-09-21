/**
 * DEEPTRACE - Mock Data Deprecation Notice
 * 
 * In Stage 8A, all forensic analyses, reports, and models are served live
 * by the FastAPI backend (http://127.0.0.1:8000/api):
 * - GET /api/models: Live telemetry from loaded Xception detector
 * - GET /api/reports: Case reports generated from actual completed analyses
 * - GET /api/history: Stored historical cases
 * - POST /api/analyze: DeepfakeBench Xception + YuNet inference
 * 
 * Fabricated mock datasets (SAMPLE_VIDEO_ANALYSIS, FORENSIC_REPORTS, RECENT_ANALYSES)
 * have been decommissioned to maintain 100% forensic data integrity.
 */

import type { Analysis, ModelInfo, ForensicReport } from '../types/forensics';

export const RECENT_ANALYSES: Analysis[] = [];
export const DETECTION_MODELS: ModelInfo[] = [];
export const FORENSIC_REPORTS: ForensicReport[] = [];
