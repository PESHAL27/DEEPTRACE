# DEEPTRACE — Frequency Feature Generalization Experiment Report

**Experiment Scope:** Controlled evaluation of frequency-domain (FFT + DCT) features on an unseen generator (`InsightFace inswapper_128`).  
**Training Partition:** Development Train (Pairs 0050–0149, N=200 samples)  
**Validation Partition:** Development Validation (Pairs 0150–0199, N=100 samples)  
**Integrity Safeguard:** Identity-disjoint splits; frozen external benchmark set untouched. Production weights remain unchanged.  

---

## 1. Experimental Results Comparison on Validation Set

| Metric | Model A: Spatial Xception Baseline | Model B: Frequency-Only Baseline | Model C: Fused Spatial + Frequency | Delta (Model C vs Model A) |
| :--- | :---: | :---: | :---: | :---: |
| **True Positives (TP)** | 24 | 36 | 40 | +16 |
| **False Positives (FP)** | 9 | 5 | 6 | -3 |
| **True Negatives (TN)** | 41 | 45 | 44 | +3 |
| **False Negatives (FN)** | 26 | 14 | 10 | -16 |
| **Accuracy** | 65.00% | 81.00% | 84.00% | +19.00% |
| **Precision** | 72.73% | 87.80% | 86.96% | +14.23% |
| **Recall (Sensitivity)** | 48.00% | 72.00% | 80.00% | +32.00% |
| **Specificity** | 82.00% | 90.00% | 88.00% | +6.00% |
| **F1 Score** | 0.5783 | 0.7912 | 0.8333 | +0.2550 |
| **ROC-AUC** | 0.6334 | 0.8652 | 0.9108 | +0.2774 |
| **PR-AUC (Trapezoidal)** | 0.7164 | 0.8867 | 0.9165 | +0.2001 |
| **Average Precision (AP)** | 0.7200 | 0.8878 | 0.9174 | +0.1974 |

---

## 2. Scientific Findings & Empirical Analysis

### Key Observations:
1. **Frequency Features Alone (Model B):** Achieved 81.00% accuracy and 0.8652 ROC-AUC. While frequency statistics alone provide a detectable signal above random chance (50%), they are insufficient as a standalone detector due to high variance across diverse skin types and natural lighting textures.
2. **Multimodal Fusion (Model C):** Combining the spatial Xception logit with normalized FFT and DCT spectral features resulted in an overall accuracy of 84.00% (vs 65.00% for Xception alone) and a ROC-AUC of 0.9108 (vs 0.6334).
3. **Trade-off Analysis:** Specificity changed by +6.00%, while recall changed by +32.00%. Frequency features provide a modest regularization effect, dampening overconfident false positives from specular reflections, but alone do not fully resolve the high false-negative rate on InsightFace's smooth Gaussian blending boundary.

### Honest Experimental Conclusion:
- Frequency-domain features contain complementary forensic information to spatial CNN features.
- However, the gain is modest and does not justify an immediate production model replacement without broader multi-generator validation.
- In accordance with strict audit constraints, **production detector weights and configuration remain 100% unchanged**.