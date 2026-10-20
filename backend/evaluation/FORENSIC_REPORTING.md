# DEEPTRACE Media Forensics — Professional Forensic Reporting Standard (Stage 8G)

## 1. Executive Summary & Purpose

In digital media forensics, technical detection models (such as deep learning classifiers) provide statistical probabilities, but real-world forensic investigations require comprehensive, verifiable, and neutral documentation. 

The **DEEPTRACE Professional Forensic Reporting Engine** synthesizes multi-layered forensic evidence collected across all analytical pipelines—primary deep learning detection, spatial facial tracking, visual attribution, frequency-domain analysis, container metadata extraction, temporal dynamics, and cross-model comparison—into an immutable, reproducible, and standards-compliant forensic case report.

Every report generated adheres to strict forensic standards:
- **Zero Hallucination / Zero Fabrication**: No synthetic "confidence scores", artificial heatmaps, or invented metadata.
- **Cryptographic Chain of Custody**: Direct computation of real SHA-256 cryptographic digests from uploaded media files.
- **Forensic Neutrality**: Objective, evidence-based phrasing adhering strictly to scientific reporting practices (no claims of "100% fake", "absolute proof", or legal admissibility).
- **Auditability & Immutability**: Persistent JSON report snapshots saved to disk (`uploads/reports/REP-{case_id}.json`) to maintain identical state across server restarts.
- **Security & Privacy**: Zero leakage of host environment filesystem paths or internal server configuration.

---

## 2. Report Architecture & Structure

Each forensic report conforms to `DetailedForensicReport` schema, structured into 15 cohesive forensic sections:

```
+-----------------------------------------------------------------------+
|                       DEEPTRACE FORENSIC REPORT                       |
+-----------------------------------------------------------------------+
|  1. Header & Case Identification (Case ID, SHA-256, Timestamps)       |
|  2. Analysis Environment (Platform, Engine, Runtime Device, Specs)     |
|  3. Primary Detector Findings (Xception Architecture, Verdict, Prob)   |
|  4. Face Detection Evidence (YuNet, Coordinates, Multi-Face Context)  |
|  5. Visual Attribution Evidence (Grad-CAM Layer, Region Attribution)  |
|  6. Frequency-Domain Evidence (2D FFT, Spectral Characteristics)       |
|  7. Metadata & Container Forensics (EXIF, Streams, Container Checks)   |
|  8. Video Frame Analysis Summary (Sample Interval, Aggregation)       |
|  9. Advanced Temporal Dynamics (Deltas, Volatility, Face Continuity)  |
| 10. Multi-Model Cross-Validation (MesoNet-4 Comparison & Agreement)    |
| 11. Independent Evidence Domains (Status & Factual Observations)       |
| 12. Dynamic Executive Summary (Objective Case Synthesis)              |
| 13. Scientific Methodology (10-Stage Systematic Forensic Pipeline)     |
| 14. Technical Limitations & Known Failure Modes (Standard Disclaimers) |
| 15. Automated Analysis Statement & Report Generation Notice            |
+-----------------------------------------------------------------------+
```

---

## 3. Cryptographic Chain of Custody & Evidence Integrity

### 3.1 Real Cryptographic Hashing
Upon ingestion, the media file is subjected to SHA-256 cryptographic digest calculation over 64 KB binary chunks:
```python
hasher = hashlib.sha256()
with open(file_path, "rb") as f:
    while chunk := f.read(65536):
        hasher.update(chunk)
file_hash = hasher.hexdigest()
```
The resulting 64-character hexadecimal digest is embedded into the report and immutable case record. Any alteration of the physical media invalidates this hash.

### 3.2 Report Snapshot Persistence
To prevent report tampering and ensure reproducibility:
- Every completed analysis generates an immutable snapshot file: `backend/uploads/reports/REP-{case_id}.json`.
- Historical requests (`GET /api/reports/{id}`) read directly from the persistent snapshot if available.
- Reports cannot be retroactively altered by retraining or restarting backend models.

### 3.3 Security & Path Sanitization
Under no circumstances are local server filesystem paths (such as `C:\path\...` or `/home/...`) exposed in API responses or printable reports:
- Checkpoints are sanitized to model basenames (e.g. `xception_best.pth`).
- File paths are sanitized to relative uploaded filenames.
- Operating system details report high-level platform names (`Windows 10` or `Linux`) without exposing username directories.

---

## 4. Multi-Layer Evidence Synthesis

| Evidence Layer | Analytical Source | Primary Forensic Indicators |
| :--- | :--- | :--- |
| **Primary Detector** | DeepfakeBench Xception (`xception_best.pth`) | Manipulated-class logit, model probability (softmax), categorical verdict |
| **Face Detection** | OpenCV YuNet DNN | Spatial face bounding boxes, multi-face counts, detection confidences |
| **Visual Attribution** | Grad-CAM (Target layer: `conv4` / final conv) | Salience maps, activation overlays, spatial focus localization |
| **Frequency Analysis** | 2D Fast Fourier Transform (FFT) | Radial spectrum, high-frequency energy ratio, spectral falloff |
| **Metadata Forensics** | EXIF & FFprobe Container Parsing | Software editing signatures, encoder tags, container stream mismatches |
| **Temporal Dynamics** | Video Frame Interpolation & Tracking | Probability volatility, transition spikes, face presence gaps, bbox deltas |
| **Cross-Validation** | DeepfakeBench MesoNet-4 (`meso4_deepfakebench`) | Independent comparison probability, architecture agreement assessment |

### 4.1 Neutral Forensic Verdict Categorization
DEEPTRACE utilizes standardized, non-sensationalist terminology:
- **`LIKELY MANIPULATED`**: The primary detector extracted high statistical anomalies (manipulated probability > 0.65) supported by spatial or temporal indicators.
- **`REAL` / `AUTHENTIC`**: The primary detector identified authentic facial features consistent with natural photographic capture (manipulated probability < 0.35).
- **`INCONCLUSIVE`**: Feature activations lie within the ambiguous margin (0.35 - 0.65) or contradictory forensic signals are observed.
- **`NO FACE DETECTED`**: Automated facial tracking algorithms could not reliably locate human facial geometry meeting the minimum confidence threshold (0.60).

---

## 5. Forensic Neutrality & Scientific Disclaimers

Forensic science requires objectivity. All DEEPTRACE reports strictly observe the following principles:

1. **Non-Sensationalist Language**: Prohibited terms include *"confirmed fake"*, *"100% fake"*, *"absolute proof"*, *"guaranteed authentic"*, or *"truth detector"*.
2. **Probabilistic Nature of Deep Learning**: Predictions represent mathematical likelihoods based on empirical training distributions and must be evaluated alongside independent corroborating physical evidence.
3. **No Legal Certification Claim**: Reports state clearly that they represent automated technical forensic analysis and do not constitute an autonomous legal judgment.

---

## 6. API Specifications

### 6.1 List Reports
- **Endpoint**: `GET /api/reports`
- **Response**: List of forensic report summaries compatible with previous stages, enriched with `detailed_report` when available.

### 6.2 Get Detailed Report
- **Endpoint**: `GET /api/reports/{report_id}`
- **Response**: Full `DetailedForensicReport` JSON object containing all 15 analytical sections.

### 6.3 Printable HTML Report
- **Endpoint**: `GET /api/reports/{report_id}/printable`
- **Response**: Standalone, print-optimized HTML document with embedded CSS `@media print` rules for clean PDF export.

---

## 7. Verification & Automated Test Suite

The reporting engine is fully verified via unit and regression test suites:
- `backend/test_forensic_reports.py`: 20 specialized tests validating all report generation, sanitization, hashing, persistence, and HTML export features.
- Complete test suite: 112 passed tests across all 9 subsystems.
