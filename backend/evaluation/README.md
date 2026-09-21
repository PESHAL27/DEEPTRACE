# DEEPTRACE Detection Reliability & Benchmarking Subsystem

## 1. Overview & Purpose
The DEEPTRACE Benchmarking Subsystem provides a reproducible, isolated framework for measuring the empirical classification performance of the production DeepfakeBench Xception detector on ground-truth labeled evaluation datasets.

Benchmarking operations are logically and architecturally separated from production case analysis:
- **Production Analysis** (`POST /api/analyze`): Operates on unverified user media; produces probabilistic risk estimates, Grad-CAM attribution, 2D FFT spectral features, and forensic reports.
- **Benchmarking Engine** (`POST /api/benchmark/run`): Operates strictly on ground-truth labeled manifests; computes statistical classification metrics (Accuracy, F1, Precision, Recall, Specificity, ROC-AUC, PR-AUC, FPR, FNR), decision threshold sensitivity sweeps, and False Positive / False Negative error distributions.

---

## 2. Dataset Manifest Format
Evaluation datasets are defined through structured JSON manifests stored in `backend/evaluation/manifests/`.

### Schema Specification
```json
{
  "manifest_version": "1.0",
  "dataset_name": "FaceForensics++_c23_Test",
  "dataset_version": "1.0",
  "description": "Evaluation partition of FF++ c23 deepfake benchmarks.",
  "created_at": "2026-09-20",
  "samples": [
    {
      "sample_id": "ff_001",
      "file_path": "datasets/ff_sample_01.jpg",
      "sha256": "3a7b9c...",
      "media_type": "image",
      "ground_truth": "manipulated",
      "source": "FaceForensics++",
      "dataset_name": "FF++-Deepfakes",
      "split": "test",
      "manipulation_type": "face_swap",
      "resolution": "1024x1024",
      "compression_level": "c23",
      "notes": "Verified deepfake subject."
    }
  ]
}
```

### Manifest Fields
- `sample_id` (string, required): Unique identifier for the sample.
- `file_path` (string, required): Relative or absolute path to the media file.
- `sha256` (string, optional): SHA-256 cryptographic digest used for integrity and leakage detection.
- `media_type` (string, required): `"image"` or `"video"`.
- `ground_truth` (string, required): Strictly `"authentic"` or `"manipulated"`.
- `source` (string, optional): Benchmark dataset source (e.g. CelebA, FF++, DFDC).
- `split` (string, optional): `"train"`, `"val"`, or `"test"`.
- `manipulation_type` (string, optional): Manipulation category (e.g. `"face_swap"`, `"face_reenactment"`, `"none"`).
- `resolution` (string, optional): Spatial dimensions (e.g. `"1024x1024"`).
- `compression_level` (string, optional): Codec quality setting (e.g. `"raw"`, `"c23"`, `"c40"`).

---

## 3. Ground-Truth Requirements & Scientific Rules
1. **Never Infer Ground Truth from Model Predictions**: Labels must originate from known, externally verified dataset provenance.
2. **Binary Positive/Negative Alignment**:
   - **Positive Class ($1$)**: `"manipulated"`
   - **Negative Class ($0$)**: `"authentic"`
3. **AI-Generated Media**: If testing diffusion or full-synthesis images, maintain separate descriptive categorization rather than conflating them with traditional face-swaps unless explicitly defined by the dataset protocol.
4. **Zero-Face Handling**: Samples with no detectable facial structures are tracked under `no_face_detected` status.

---

## 4. Evaluation Metrics Formulation
Metrics are computed over all samples with valid facial detections:

| Metric | Mathematical Formula | Definition |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall proportion of correct classifications |
| **Precision** | $\frac{TP}{TP + FP}$ | Proportion of positive predictions that are truly manipulated |
| **Recall (TPR)** | $\frac{TP}{TP + FN}$ | Proportion of actual manipulated samples correctly flagged |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | Proportion of actual authentic samples correctly flagged |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of Precision and Recall |
| **False Positive Rate (FPR)** | $\frac{FP}{FP + TN}$ | Rate of authentic samples misclassified as manipulated |
| **False Negative Rate (FNR)** | $\frac{FN}{FN + TP}$ | Rate of manipulated samples misclassified as authentic |
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Evaluated when both authentic and manipulated samples are present |
| **PR-AUC** | Area under the Precision-Recall curve | Evaluated when both classes are present |

*Zero-Denominator Handling*: All divisions check denominator $> 0$ and return $0.0$ to prevent `NaN` or zero-division crashes.

---

## 5. Threshold Sensitivity Sweep
The benchmark runner evaluates detector performance across 17 decision thresholds:
$$\tau \in [0.10, 0.15, 0.20, \dots, 0.90]$$
For every threshold $\tau$:
- If $P(\text{manipulated}) \ge \tau$, predicted class = `"manipulated"`.
- Re-evaluates $TP, TN, FP, FN$, Accuracy, Precision, Recall, Specificity, F1, FPR, and FNR.
- **Evaluation Only**: The sweep provides empirical trade-off analysis and does **not** automatically modify the production threshold (`DEEPFAKE_THRESHOLD = 0.50`).

---

## 6. Confusion Matrix & Error Analysis
1. **Confusion Matrix**: Generates both raw counts and normalized rates:
   $$\begin{pmatrix} \text{TN Rate} & \text{FP Rate} \\ \text{FN Rate} & \text{TP Rate} \end{pmatrix}$$
2. **Error Distribution**:
   - **False Positives**: Samples where `ground_truth == "authentic"` and `predicted == "manipulated"`.
   - **False Negatives**: Samples where `ground_truth == "manipulated"` and `predicted == "authentic"`.
   - Errors are ranked by prediction probability deviation to facilitate root-cause investigation.

---

## 7. Data Leakage & Integrity Protection
- **SHA-256 Hashing**: Validates that evaluation media files have not undergone unintended corruption or modification.
- **Duplicate Detection**: Flags multiple occurrences of identical SHA-256 digests in a single benchmark run.
- **Train/Test Split Leakage**: Alerts investigators if an identical file hash appears in both training/validation and test partitions.

---

## 8. Benchmark API Endpoints
- `POST /api/benchmark/run`: Execute a benchmark run using a manifest path or inline JSON.
- `GET /api/benchmark/runs`: List historical benchmark runs and executive metrics.
- `GET /api/benchmark/runs/{run_id}`: Retrieve detailed benchmark report.
- `GET /api/benchmark/runs/{run_id}/metrics`: Retrieve metrics and confusion matrix.
- `GET /api/benchmark/runs/{run_id}/errors`: Retrieve False Positive and False Negative samples.
- `GET /api/benchmark/template`: Retrieve sample manifest template.

---

## 9. Known Limitations
- The current DeepfakeBench Xception detector is trained as a binary classifier on facial crops; it is not calibrated for full-scene image synthesis without detected faces.
- Performance on in-the-wild media may diverge from clean benchmark metrics due to aggressive re-encoding, low resolution, or non-frontal facial poses.
