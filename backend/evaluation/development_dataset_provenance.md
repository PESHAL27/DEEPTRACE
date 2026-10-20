# DEEPTRACE — Development Dataset Provenance & Split Audit

**Document Status:** ACTIVE & DOCUMENTED  
**Effective Date:** 2026-10-20  
**Upstream Repository:** Hugging Face [`Sowaiba01/Deepfake`](https://huggingface.co/datasets/Sowaiba01/Deepfake)  
**Exact Upstream Commit Revision:** `d8b189507fc026b3e0c794a06c318cb13d0c8743`  
**Declared License:** MIT License  

---

## 1. Upstream Dataset Architecture & Verified Provenance

A comprehensive inspection of `Sowaiba01/Deepfake` at revision `d8b189507fc026b3e0c794a06c318cb13d0c8743` reveals:
- **Total Images in Repository:** 10,852 image files (5,426 authentic portraits in `real/` and 5,426 paired manipulated portraits in `fake/`).
- **Metadata Documentation:** The repository provides an authoritative `metadata.csv` mapping every pair with columns:
  `['id', 'fake_file', 'real_file', 'method', 'source_identity', 'target_face', 'label_fake', 'label_real']`.
- **Underlying Photography Source:** The `target_face` and `source_identity` attributes reference image IDs formatted as 5-digit image numbers (e.g., `62330.jpg`, `66595.jpg`, `00082.jpg`), identifying the underlying source canvas as the standard **CelebA (CelebFaces Attributes)** dataset (202,599 cropped celebrity portraits).
- **Manipulated Generator:** The column `method` explicitly confirms that all deepfakes were synthesized using InsightFace's ONNX-based model: `inswapper_128`.
- **Pair Numbering Correspondence:** The pair indices `idx` in `real/real_XXXXX.jpg` and `fake/fake_XXXXX.jpg` directly map 1-to-1 to the row index in `metadata.csv`.

---

## 2. Split Architecture & Identity Isolation

To prevent data contamination and ensure evaluation integrity, the data is partitioned into three strictly segregated splits:

| Split Name | Pair Index Range | Authentic Samples | Manipulated Samples | Total Samples | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **External Benchmark (Frozen)** | `0000` – `0049` | 50 (`real_00000`–`real_00049`) | 50 (`fake_00000`–`fake_00049`) | **100** | Strictly held-out evaluation gate |
| **Development Train** | `0050` – `0149` | 100 (`real_00050`–`real_00149`) | 100 (`fake_00050`–`fake_00149`) | **200** | Feature extraction & model development |
| **Development Validation** | `0150` – `0199` | 50 (`real_00150`–`real_00199`) | 50 (`fake_00150`–`fake_00199`) | **100** | Generalization & frequency experiments |

### Verified Identity Isolation
Analyzing `target_face` (target canvas) and `source_identity` (donor face) across the three splits demonstrates:
- **Benchmark vs. Train Target Overlap:** **0 / 100** ($0.0\%$)
- **Benchmark vs. Validation Target Overlap:** **0 / 50** ($0.0\%$)
- **Train vs. Validation Target Overlap:** **0 / 50** ($0.0\%$)
- **Benchmark vs. Train Donor Overlap:** **0 / 100** ($0.0\%$)
- **Benchmark vs. Validation Donor Overlap:** **0 / 50** ($0.0\%$)
- **Train vs. Validation Donor Overlap:** **0 / 50** ($0.0\%$)

Every target canvas image and donor image is 100% unique and mutually exclusive across the splits.

---

## 3. Disclosed Identity-Overlap & Provenance Limitations

To maintain scientific integrity without unverified assertions, the following limitations are explicitly noted:
1. **CelebA Multi-Image Identity Clustering:**  
   `metadata.csv` identifies specific CelebA image files (e.g. `00082.jpg`), which are mutually exclusive across splits. However, in CelebA, the 202,599 images correspond to 10,177 distinct celebrity identities (an average of ~20 images per person). Without cross-referencing CelebA's proprietary `identity_CelebA.txt` cluster mappings, we cannot guarantee that two different image files (e.g. in Train and Validation) do not depict the same celebrity at different ages or events. Complete identity leakage prevention cannot be claimed without cluster-level biometric indexing.
2. **FaceForensics++ Pretraining Overlap:**  
   Xception was pretrained on FaceForensics++ (YouTube video sequences). Because CelebA consists of celebrity photographs and FaceForensics++ contains celebrity/public YouTube videos, cross-dataset identity overlap between the training set of Xception and the CelebA canvases cannot be proven or disproven without biometric facial feature matching.
3. **Subject Consent & Privacy Limitations:**  
   The dataset is governed by the permissive open-source **MIT License**, permitting academic and non-commercial research experimentation. However, the subjects in CelebA were originally scraped from public web imagery without individual biometric releases. In compliance with ethical standards and biometric privacy frameworks (GDPR Article 9, Illinois BIPA), this data is suitable solely for offline laboratory benchmarking and evaluation, not commercial training deployment.

---

## 4. Manifest Integrity and Perceptual Hashing

Every sample in `development_train_manifest.json` and `development_validation_manifest.json` is registered with:
- Exact relative filesystem path
- Cryptographic SHA-256 hash (64 hex characters)
- Perceptual difference hash (`dhash`, 16 hex characters) to detect perceptual near-duplicates
- Image resolution (`256 x 256 x 3` RGB)
- Ground-truth label (`authentic` vs `manipulated`)
- Target canvas ID and source donor ID
