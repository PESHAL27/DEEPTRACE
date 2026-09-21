# DEEPTRACE Media Forensics — Advanced Temporal Video Analysis (Stage 8F)

## 1. Executive Summary

In digital video forensics, inspecting isolated frames in isolation can overlook temporal dynamics that provide crucial forensic context. However, naively blending inter-frame fluctuations into an artificial "temporal deepfake score" introduces severe statistical vulnerabilities and high false-positive rates due to natural video phenomena (compression, motion blur, lighting changes, camera motion).

**Stage 8F** introduces an advanced, deterministic **Temporal Video Analysis** subsystem into DEEPTRACE. Rather than modifying the primary production decision model or fabricating a synthetic aggregate score, this subsystem measures and documents empirical sequence-level properties across analyzed video frames:
1. **Consecutive Frame Probability Steps ($\Delta_i$) and Statistical Volatility ($\sigma$, dispersion)**
2. **Classification Decision Boundary Transitions (Prediction Flips)**
3. **Facial Presence Continuity and Detection Streaks/Gaps**
4. **Normalized Spatial Bounding-Box Displacement and Scale Dynamics**
5. **Cross-Model Temporal Alignment (Xception vs. MesoNet-4)**
6. **Discrete Temporal Anomaly Events**

All temporal metrics are strictly calculated from actual analyzed frames. The sovereign production verdict remains anchored to the validated DeepfakeBench Xception detector.

---

## 2. Temporal Metrics & Mathematical Formulations

### 2.1 Frame Sampling & Validity Criteria
For an input video $\mathcal{V}$ sampled with interval $S$ up to $N_{\text{max}}$ frames, let the sequence of sampled frames be:
$$\mathcal{F} = \{ f_0, f_1, \dots, f_{N-1} \}$$

Each frame $f_k$ is evaluated by the OpenCV YuNet face detector. A frame is deemed **valid** ($v_k = 1$) if and only if at least one facial structure is localized:
$$v_k = \begin{cases} 1, & \text{if } \text{faces}(f_k) \ge 1 \\ 0, & \text{otherwise} \end{cases}$$

Let $\mathcal{F}_{\text{valid}} = \{ f_{k_0}, f_{k_1}, \dots, f_{k_{M-1}} \}$ be the subsequence of valid face frames ($M \le N$).

### 2.2 Consecutive Frame Probability Deltas
To avoid generating artificial probability leaps across missing face segments, probability steps are computed **strictly between consecutive valid frames**:
$$\Delta_i = P_{\text{fake}}(f_{k_{i+1}}) - P_{\text{fake}}(f_{k_i}), \quad i \in \{0, 1, \dots, M-2\}$$
$$|\Delta_i| = \left| P_{\text{fake}}(f_{k_{i+1}}) - P_{\text{fake}}(f_{k_i}) \right|$$

Where:
- $\Delta_i > 0$: An upward step in suspicion between consecutive detections.
- $\Delta_i < 0$: A downward relaxation in suspicion.
- $|\Delta_i|$: Magnitude of the probability delta.

### 2.3 Sequence Volatility & Statistical Dispersion
For valid frames with predicted manipulation probabilities $\mathbf{p} = [p_0, p_1, \dots, p_{M-1}]$:
- **Sample Mean ($\mu$):** $\mu = \frac{1}{M} \sum_{i=0}^{M-1} p_i$
- **Median ($\tilde{p}$):** 50th percentile of sorted probabilities.
- **Sample Standard Deviation ($\sigma$):**
  $$\sigma = \sqrt{\frac{1}{M-1} \sum_{i=0}^{M-1} (p_i - \mu)^2} \quad (M \ge 2)$$
- **Mean Absolute Delta ($\overline{|\Delta|}$):**
  $$\overline{|\Delta|} = \frac{1}{M-1} \sum_{i=0}^{M-2} |\Delta_i|$$
- **Maximum Absolute Delta ($\Delta_{\text{max}}$):**
  $$\Delta_{\text{max}} = \max_{i} |\Delta_i|$$

### 2.4 Prediction Transitions (Classification Boundary Flips)
A prediction transition occurs between consecutive valid frames $f_{k_i}$ and $f_{k_{i+1}}$ when the categorical label crosses the decision boundary:
$$\text{Transition}(i) \iff \text{Verdict}(f_{k_i}) \ne \text{Verdict}(f_{k_{i+1}})$$

For binary threshold $\tau = 0.50$:
$$\text{Authentic} \to \text{Manipulated}: \quad p_i < 0.50 \land p_{i+1} \ge 0.50$$
$$\text{Manipulated} \to \text{Authentic}: \quad p_i \ge 0.50 \land p_{i+1} < 0.50$$

Repeated transitions indicate high model uncertainty or intermittent manipulation artifacts.

### 2.5 Facial Presence Continuity & Streaks
- **Presence Percentage ($C_{\text{pres}}$):**
  $$C_{\text{pres}} = \frac{M}{N} \times 100\%$$
- **Longest Present Streak ($S_{\text{pres}}$):** Maximum consecutive sampled frames where $v_k = 1$.
- **Longest Missing Streak ($S_{\text{miss}}$):** Maximum consecutive sampled frames where $v_k = 0$.

Intervals of $\ge 2$ consecutive missing frames generate a `face_presence_gap` temporal event.

### 2.6 Spatial Bounding Box & Scale Dynamics
For each valid frame with localized primary face bounding box $B_i = (x_i, y_i, w_i, h_i)$ on a video of dimensions $(W, H)$:
- **Normalized Center Coordinates:**
  $$c_x(i) = \frac{x_i + w_i / 2}{W}, \quad c_y(i) = \frac{y_i + h_i / 2}{H}$$
- **Normalized Center Displacement ($\Delta c_i$):**
  $$\Delta c_i = \sqrt{(c_x(i+1) - c_x(i))^2 + (c_y(i+1) - c_y(i))^2}$$
- **Normalized Face Area ($A_i$):**
  $$A_i = \frac{w_i \times h_i}{W \times H}$$
- **Facial Scale Delta ($\Delta A_i$):**
  $$\Delta A_i = |A_{i+1} - A_i|$$

*Note: Spatial tracking operates on the primary representative face per frame. Identity re-identification is not applied.*

### 2.7 Cross-Model Temporal Alignment
Between the primary Xception detector and comparison MesoNet-4 across shared valid frames:
- **Agreement Rate ($R_{\text{agree}}$):** Percentage of frames where both models produce identical categorical predictions.
- **Cohen's Kappa ($\kappa$):** Chance-corrected inter-rater agreement:
  $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
- **Pearson Correlation ($r$):** Linear correlation between continuous output probabilities:
  $$r = \frac{\sum (p_{X, i} - \bar{p}_X)(p_{M, i} - \bar{p}_M)}{\sqrt{\sum (p_{X, i} - \bar{p}_X)^2 \sum (p_{M, i} - \bar{p}_M)^2}}$$

---

## 3. Discrete Temporal Anomaly Events

When continuous metrics exceed engineering thresholds, discrete `TemporalEvent` items are generated for the forensic audit log:

| Event Type | Trigger Condition | Severity | Forensic Interpretation |
| :--- | :--- | :--- | :--- |
| `probability_jump` | $|\Delta_i| \ge 0.30$ | Notable ($\ge 0.40$) / Informational | Rapid shift in suspicion between consecutive detections. |
| `prediction_transition` | $\text{Verdict}(i) \ne \text{Verdict}(i+1)$ | Notable | Classification state change across the 0.50 boundary. |
| `face_presence_gap` | $\ge 2$ consecutive frames without face | Informational | Face occlusion, extreme rotation, or detector failure. |
| `face_bbox_change` | $\Delta c_i \ge 0.20$ | Notable ($\ge 0.35$) / Informational | Rapid spatial translation of detected face in the frame. |
| `face_scale_change` | $\Delta A_i \ge 0.40$ | Informational | Rapid zoom, abrupt perspective cut, or scale change. |
| `detector_disagreement` | $\text{Pred}_X(i) \ne \text{Pred}_M(i)$ | Notable | Xception and MesoNet disagree on frame classification. |

---

## 4. Non-Deepfake Causes of Temporal Inconsistency

It is a fundamental principle of media forensics that **temporal fluctuations do not constitute proof of manipulation**. An analyst must consider the following benign technical explanations:

1. **Video Compression Artifacts:**
   - Temporal predictive encoding (H.264/H.265 P-frames and B-frames) uses motion vectors and macroblock quantization. Inter-frame compression noise frequently triggers high-frequency variance that inflates CNN detector probabilities on specific frames.
2. **Camera Motion and Motion Blur:**
   - Rapid camera panning, shaky handheld capture, or abrupt acceleration softens facial textures, destroying high-frequency features and causing detector probability drops or spikes.
3. **Lighting & Exposure Variations:**
   - Flickering ambient light, strobe effects, changing white balance, or shadows crossing the subject alter facial pixel distributions without any digital alteration.
4. **Subject Pose and Occlusion:**
   - Profile turns, hand gestures covering parts of the chin or cheeks, or glasses glare can degrade YuNet facial alignment and distort normalized face crops.
5. **Sampling Aliasing:**
   - Sampling every $K$ frames creates stroboscopic gaps where intermediate transitional movement is unobserved.

---

## 5. Forensic Governance Principles

1. **Sovereignty of Primary Verdict:**
   - Temporal metrics are purely observational evidence. They **never override, blend with, or mutate** the primary Xception video aggregation verdict.
2. **No Fabricated Scores:**
   - DEEPTRACE does not compute a synthetic "temporal fake score". All values represent direct mathematical observations (standard deviation, delta, displacement, counts).
3. **Transparent Reporting:**
   - Official forensic audit reports (`/api/reports`) summarize temporal metrics as factual findings (e.g., "Analyzed 10 sampled frames. Face presence 100.0%, volatility mean=0.882, std=0.041, 0 prediction flips, 0 temporal events").
4. **Traceability:**
   - Every temporal event links directly to the specific frame number and timestamp, allowing the forensic examiner to inspect the original frame, YuNet crop, Grad-CAM activation map, and 2D FFT spectrum simultaneously.
