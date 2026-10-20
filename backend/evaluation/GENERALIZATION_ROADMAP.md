# DEEPTRACE — Dataset Preservation and Generalization Roadmap

**Document Status:** ACTIVE & AUDITED (Phase 2C Pipeline Consistency & Evaluation Gate Completed)  
**Effective Date:** 2026-10-20  
**Repository Module:** `backend/evaluation/`  

---

## 1. Fixed Evaluation Set Preservation Policy

The 100-sample paired facial manipulation evaluation set defined in `evaluation/manifests/external_benchmark_manifest.json` remains designated as a **STRICTLY HELD-OUT FIXED EVALUATION SET**.

### Non-Negotiable Constraints
1. **Zero Tuning Against Evaluation Split:**  
   The 100 samples (`EXT-REAL-0000` through `EXT-REAL-0049` and `EXT-FAKE-0000` through `EXT-FAKE-0049`) remain **permanently frozen** and were NOT used for model training, feature selection, or threshold calibration.
2. **Deterministic Immutability:**  
   The SHA-256 hashes of all 100 benchmark images match on every verification run.
3. **Production Stability:**  
   The production Xception detector weights ([`xception_best.pth`](file:///c:/Users/pecul/Desktop/Peshal/college/Hackathon/DEEPTRACE/backend/app/models/weights/xception_best.pth)), crop padding ($0.20$), YuNet parameters, and decision threshold ($0.50$) remain 100% untouched.

---

## 2. Upstream Provenance, Licensing, and Identity Claims Audit

### Upstream Repository Details
- **Repository:** Hugging Face `Sowaiba01/Deepfake` (DeepGuard AI Project)
- **Commit Revision:** `d8b189507fc026b3e0c794a06c318cb13d0c8743`
- **Declared License:** MIT License
- **Base Canvas Source:** CelebA dataset (evidenced by image filenames like `62330.jpg`, `00082.jpg` in `metadata.csv`).
- **Manipulated Generator:** InsightFace `inswapper_128.onnx`.

### Clarification of License vs. Biometric Privacy & Ethics
> [!IMPORTANT]
> **Legal License Distinction:** The MIT license is a permissive copyright license granting rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies for both commercial and non-commercial purposes.  
> Operational or deployment restrictions arise from **biometric privacy legislation (e.g. EU GDPR Article 9, Illinois BIPA, California CCPA), personality rights, and the absence of verified subject consent or model releases**, NOT from copyright terms.

### Status of Training-Data Overlap & Identity Leakage Claims
- **Within-Split Leakage:** Target canvas files and donor identity files are 100% disjoint across the Benchmark, Development-Train, and Development-Validation splits ($0$ target canvas overlap, $0$ donor overlap).
- **CelebA Multi-Image Cluster Limitation:** CelebA images average ~20 photos per celebrity identity. Without cross-referencing CelebA's cluster metadata (`identity_CelebA.txt`), multiple images of the same individual celebrity across splits cannot be completely excluded.
- **Cross-Dataset Training Overlap:** <mark>**UNVERIFIED**</mark>. The Xception model was pretrained on FaceForensics++ (YouTube video sequences). Overlap between the FF++ training partition and CelebA subjects cannot be verified without biometric feature matching.

---

## 3. Implementation Status of Dataset Partitions

| Partition / Dataset | Status | Sample Count | Implemented Location | Identity Overlap Safeguards |
| :--- | :--- | :---: | :--- | :--- |
| **External Benchmark Set** | **IMPLEMENTED & FROZEN** | 100 images (50 pairs) | `evaluation/datasets/external_benchmark/` | Pairs 0000–0049 permanently held out |
| **Development Train Split** | **IMPLEMENTED** | 200 images (100 pairs) | `evaluation/datasets/development_train/` | Pairs 0050–0149 (disjoint target/donor IDs) |
| **Development Validation Split** | **IMPLEMENTED** | 100 images (50 pairs) | `evaluation/datasets/development_validation/` | Pairs 0150–0199 (disjoint target/donor IDs) |
| **Cross-Generator Test Set** | **BLOCKED (UNAVAILABLE)** | 0 | *No verified non-inswapper dataset present* | Formal blocker logged pending authorized acquisition |
| **Celeb-DF (v2) Benchmark** | **PLANNED** | 518 test videos | *Planned future expansion* | Official subject-disjoint test split |
| **FaceForensics++ Multi-Generator** | **PLANNED** | 1,000 video sequences | *Planned future expansion* | Standard c23/c40 compression splits |

---

## 4. Phase 2C Input-Representation Ablation Results (Validation Split N=100)

A controlled 5-variant ablation was executed in Phase 2C to resolve the full-canvas versus face-crop feature discrepancy:

| Variant | Representation Description | Accuracy | Recall | Specificity | F1 | ROC-AUC | AP | Trapezoidal PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A** | Frozen Spatial Xception Baseline | 65.00% | 48.00% | 82.00% | 0.5783 | 0.6334 | 0.7200 | 0.7164 |
| **Variant B** | Frequency-Only (Full Canvas) | 81.00% | 72.00% | 90.00% | 0.7912 | 0.8668 | 0.8889 | 0.8878 |
| **Variant C** | Frequency-Only (Primary Face Crop) | 83.00% | 78.00% | 88.00% | 0.8211 | 0.8816 | 0.8941 | 0.8930 |
| **Variant D** | Fused (Xception + Full Canvas) | 84.00% | 80.00% | 88.00% | 0.8333 | 0.9124 | 0.9180 | 0.9171 |
| **Variant E** | Fused (Xception + Primary Face Crop) | **86.00%** | **84.00%** | 88.00% | **0.8571** | **0.9188** | 0.9150 | 0.9139 |

### Paired Bootstrap Comparisons (B=2,000 Replicates)
- **Variant E vs Variant A (Fused Face-Crop vs Xception Baseline):**
  - $\Delta \text{Accuracy} = +21.00\%$ (95% CI: $[+11.00\%, +30.02\%]$)
  - $\Delta \text{Recall} = +36.00\%$ (95% CI: $[+18.00\%, +54.00\%]$)
  - $\Delta \text{Specificity} = +6.00\%$ (95% CI: $[-6.00\%, +18.00\%]$)
  - $\Delta \text{ROC-AUC} = +0.2854$ (95% CI: $[+0.1748, +0.4012]$)
- **Variant E vs Variant D (Face Crop vs Full Canvas Fusion):**
  - $\Delta \text{Accuracy} = +2.00\%$ (95% CI: $[-4.00\%, +8.00\%]$)
  - $\Delta \text{ROC-AUC} = +0.0064$ (95% CI: $[-0.0232, +0.0360]$)
  - **Conclusion:** Face-crop representation matches production inference architecture with no loss in predictive performance. Neither representation is statistically distinguishable as superior.

---

## 5. Robustness & Perturbation Vulnerability Profile

Diagnostic stress testing on linked validation samples revealed distinct operational boundaries:
1. **Lossy JPEG Recompression:**
   - Feature drift at $Q=95$: $0.04\%$
   - Feature drift at $Q=75$: $3.90\%$
   - Feature drift at $Q=50$: **$10.20\%$**
   - *Vulnerability:* Heavy JPEG recompression degrades high-frequency spectral residuals.
2. **Dimension Scaling:**
   - Downscaling to $128 \times 128$: **$28.72\%$ drift** (destroys high-frequency artifacts).
   - Upscaling to $512 \times 512$: **$28.40\%$ drift** (bilinear interpolation smooths spectrum).
3. **Crop Padding Sensitivity:**
   - $0.15$ padding ratio: $8.35\%$ drift vs reference $0.20$.
   - $0.25$ padding ratio: $14.97\%$ drift vs reference $0.20$.
   - *Operational Rule:* Strict fixed $0.20$ crop padding must be maintained across both training and inference.
4. **Edge Cases:**
   - No-face inputs cleanly halt with `no_face_detected` status.
   - Multi-face inputs deterministically select the primary face by bounding box area.

---

## 6. Mathematical Metric Reconciliation Summary

1. **Trapezoidal PR-AUC vs. Average Precision:**  
   Reconciliation established that Trapezoidal PR-AUC (`auc(recall, precision)`) and Average Precision (`average_precision_score`) are mathematically distinct integration methods. Differences (e.g. 0.7164 vs 0.7200 for Model A, 0.9171 vs 0.9180 for Model C) are mathematically expected and not software bugs.
2. **OpenCV Grayscale Decoding Difference:**  
   A minor delta in reported ROC-AUC ($0.8652$ vs $0.8668$ for frequency-only; $0.9108$ vs $0.9124$ for fusion) was traced to `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` vs `cv2.cvtColor(cv2.imread(..., BGR), cv2.COLOR_BGR2GRAY)`. Floating-point conversion shifted the ranking of 1 near-boundary sample pair.
3. **Threshold-Dependent Consistency:**  
   All confusion matrices, accuracy, precision, recall, specificity, and F1 scores matched with **zero discrepancy** across all artifacts.

---

## 7. Cross-Generator Evaluation Blocker & Next Steps

### The Blocker
All current development and validation datasets originate from InsightFace `inswapper_128`. Evaluating on `development_validation` establishes within-generator validation on held-out samples, but cannot substantiate claims of cross-generator generalization against alternative architectures (e.g. diffusion models, GAN reenactment, SimSwap).

No non-`inswapper_128` dataset with verified provenance, open licensing, and ground-truth pairing exists in the workspace. Academic benchmarks (FaceForensics++, Celeb-DF v2) require formal institutional data access agreements.

### Formal Evaluation Gating Protocol
Before the experimental fusion model may be considered for a production integration trial:
1. **Acquire Independent Benchmark:** Secure authorized access to a verified external benchmark (e.g., FaceForensics++ c23 Deepfakes or SimSwap).
2. **Cryptographic Manifest:** Generate SHA-256 manifests freezing sample selection.
3. **Zero-Tuning Test:** Evaluate frozen Spatial Xception and already-fitted fusion models without refitting scalers or classifiers.
4. **Gate Threshold:** Fusion model must maintain statistically significant improvement ($\Delta \text{Accuracy} > 0$ with 95% CI strictly above zero) on the independent generator.

---

## 8. Final Decisions Summary

1. **Production Xception Status:** **FROZEN / UNCHANGED.**
2. **Experimental Fusion Model Status:** **GATED / EXPERIMENTAL.**
3. **Frequency Representation Suitability:** **SUITABLE FOR FUTURE TRIAL** using the Primary-Face Crop representation (Variant E).
4. **Cross-Generator Evaluation Status:** **NOT COMPLETED (FORMALLY BLOCKED).**
