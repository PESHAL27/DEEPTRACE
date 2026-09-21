# DEEPTRACE — Model Sources & Multi-Model Cross-Validation Documentation

## Overview
This document provides complete, transparent documentation for all deepfake detection models deployed in **DEEPTRACE**. In accordance with strict digital media forensic standards, the system distinguishes between the **Primary Production Detector** (DeepfakeBench Xception) and the **Comparison Model** (DeepfakeBench MesoNet / Meso4).

---

## 1. Primary Production Model: DeepfakeBench Xception

- **Role**: Primary Production Detector
- **Model ID**: `xception_deepfakebench`
- **Display Name**: DeepfakeBench Xception
- **Architecture**: Deep Convolutional Neural Network with Depthwise Separable Convolutions (Chollet, CVPR 2017) adapted for deepfake forensics (Rössler et al., ICCV 2019).
- **Framework**: DeepfakeBench (NeurIPS 2023 Datasets and Benchmarks Track)
- **Official Repository**: [https://github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- **Checkpoint Source**: Official DeepfakeBench Release v1.0.1 (`xception_best.pth`, ~87.8 MB)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Training Dataset**: FaceForensics++ (c23 / high quality compression), encompassing Deepfakes, Face2Face, FaceSwap, and NeuralTextures.
- **Input Specifications**:
  - Resolution: $256 \times 256$ pixels
  - Color Format: RGB
  - Face Crop: OpenCV YuNet tight face detection with $0.20$ contextual padding.
  - Normalization: Mean $[0.5, 0.5, 0.5]$, Standard Deviation $[0.5, 0.5, 0.5]$, scaling pixel intensities to $[-1.0, 1.0]$.
- **Output Format & Class Mapping**:
  - Output Logits: 2-dimensional continuous tensor $[z_0, z_1]$.
  - Activation: $\text{Softmax}(z) = [\sigma_0, \sigma_1]$.
  - Index 0: `authentic` (real)
  - Index 1: `manipulated` (fake)
  - Probability: $\hat{p} = \sigma_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$
  - Default Operating Threshold: $T = 0.50$ (classifies as manipulated when $\hat{p} \ge 0.50$).
- **Explainability**: Integrated Layer-CAM / Grad-CAM visual attribution computed on the final feature extraction layer `conv4` ($728 \times 16 \times 16$).
- **Known Limitations**:
  - Sensitive to heavy spatial compression and blur perturbations.
  - Cross-dataset generalization drops on unaligned, whole-head diffusion synthesis (StyleGAN / Midjourney) compared to face-swap datasets.

---

## 2. Comparison Model (Research): DeepfakeBench MesoNet (Meso4)

- **Role**: Comparison Model (Independent Cross-Validation & Disagreement Analysis)
- **Model ID**: `meso4_deepfakebench`
- **Display Name**: DeepfakeBench MesoNet (Meso4)
- **Architecture**: Compact 4-layer convolutional neural network (Afchar et al., IEEE WIFS 2018) designed to detect microscopic facial mesoscopic properties and compression-resistant boundary artifacts.
- **Framework**: DeepfakeBench (NeurIPS 2023 Datasets and Benchmarks Track)
- **Official Repository**: [https://github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
- **Checkpoint Source**: Official DeepfakeBench Release v1.0.1 (`meso4_best.pth`, 117.9 KB)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))
- **Training Dataset**: FaceForensics++ (c23)
- **Input Specifications**:
  - Resolution: $256 \times 256$ pixels
  - Color Format: RGB
  - Face Crop: Same OpenCV YuNet detected facial crop.
  - Normalization: Mean $[0.5, 0.5, 0.5]$, Standard Deviation $[0.5, 0.5, 0.5]$, scaling pixel intensities to $[-1.0, 1.0]$.
- **Output Format & Class Mapping**:
  - Output Logits: 2-dimensional tensor $[z_0, z_1]$.
  - Activation: $\text{Softmax}(z) = [\sigma_0, \sigma_1]$.
  - Index 0: `authentic`
  - Index 1: `manipulated`
  - Probability: $\hat{p} = \sigma_1$
  - Default Operating Threshold: $T = 0.50$
- **Explainability**: Not currently implemented for the Meso4 architecture; UI displays *"Explainability not available for this detector"*, avoiding artificial or fabricated heatmaps.
- **Known Limitations**:
  - Low parameter capacity (27.9K parameters) makes Meso4 less expressive on subtle blended edges than deeper networks like Xception.
  - Higher susceptibility to false negatives on high-resolution diffusion-generated imagery.

---

## 3. Scientific Methodology for Multi-Model Cross-Validation

### A. Independent Detector Principle
The primary and comparison models operate completely independently:
1. They maintain separate weights, activations, and parameter graphs.
2. They do not share latent features.
3. No ensemble averaging, voting, or probability combination is performed. The primary model's verdict and confidence remain sovereign in production.

### B. Same-Dataset & Aligned-Frame Requirement
When evaluating model agreement or comparative benchmark performance:
- Both models are evaluated on the **exact same sample files and ground-truth labels**.
- For video cross-validation, both models receive the **exact same sampled frame indices and facial bounding box crops**.

### C. Interpretation of Model Disagreement
- When Model A predicts `authentic` and Model B predicts `manipulated`, DEEPTRACE flags:
  `"Detector disagreement detected. Review recommended."`
- Disagreement is **never** presented as proof of manipulation or proof of authenticity.
- Disagreement indicates an edge-case facial sample where differences in inductive bias between architectures (e.g. depthwise separable spatial filters in Xception vs. compact convolutional filters in Meso4) yielded conflicting signals.

### D. Inter-Model Agreement Metrics
- **Agreement Percentage**: $\frac{\text{Agreeing Frames}}{\text{Compared Frames}} \times 100\%$
- **Cohen's Kappa ($\kappa$)**: Measures inter-rater categorical agreement adjusted for chance:
  $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
- **Linear Correlation**: Pearson's $r$ and Spearman's $\rho$ continuous probability correlation across shared evaluation samples.

---

## 4. Academic Citations

```bibtex
@inproceedings{yan2023deepfakebench,
  title={DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2023}
}

@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={R{\"o}ssler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nie{\ss}ner, Matthias},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1--11},
  year={2019}
}

@inproceedings{afchar2018mesonet,
  title={MesoNet: a Compact Facial Video Forgery Detection Network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  booktitle={2018 IEEE International Workshop on Information Forensics and Security (WIFS)},
  pages={1--7},
  year={2018},
  organization={IEEE}
}
```
