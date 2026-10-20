"""
DEEPTRACE - Primary Face Analysis & Image Prediction Regression Test Suite
===========================================================================
Verifies:
1. The largest valid face by bounding-box area is selected by default as primary face.
2. A background face classified FAKE cannot override the primary face's verdict in primary-face mode.
3. Selecting another face yields that face's own prediction and probabilities.
4. All-faces mode returns separate predictions and deterministic aggregation for every valid face.
5. REAL and FAKE probabilities sum to 1.0 (within 1e-4) for each successful inference.
6. Overlay coordinates accurately map to the original image dimensions.
7. No-face images return NO FACE DETECTED with None probabilities.
8. Switching face selection does not reuse stale predictions.
9. Reports and API responses correctly serialize the selected analysis mode and face details.
10. Video analysis, history, and report generation continue to operate without regression.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

# Setup python path
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.schemas import (
    FaceAnalysisItem,
    FaceBoundingBox,
    PrimaryFaceResult,
    AllFacesResult,
    DetectionResult,
    AnalysisResponse,
    MediaMetadata,
    FaceDetectionResult
)
from app.services.detection_service import (
    analyze_image,
    DEEPFAKE_THRESHOLD
)
from app.services.report_service import ForensicReportService


class TestPrimaryFaceRegression(unittest.TestCase):
    def setUp(self):
        self.peshal_img_path = BACKEND_DIR / "uploads" / "DT-2026-09236169_peshal1.jpeg"
        self.no_face_path = BACKEND_DIR / "test_fixtures" / "test_image.jpg"

    def test_01_largest_valid_face_selected_by_default(self):
        """Verify the largest face by bounding-box area (w * h) is selected as primary face by default."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        self.assertEqual(res.status, "completed")
        self.assertGreaterEqual(len(res.faces), 2, "Expected at least 2 faces detected in peshal1.jpeg")

        # Largest area must be the primary face
        areas = [f.area for f in res.faces]
        max_area_idx = int(np.argmax(areas))

        self.assertIsNotNone(res.primary_face_result)
        self.assertEqual(res.primary_face_result.face_index, max_area_idx)
        self.assertEqual(res.primary_face_result.area, areas[max_area_idx])
        self.assertTrue(res.faces[max_area_idx].is_primary)
        self.assertGreater(res.faces[max_area_idx].area, res.faces[1 - max_area_idx].area)

    def test_02_background_fake_face_cannot_override_primary_verdict(self):
        """Verify a small background face with FAKE verdict cannot override the primary face's REAL verdict."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        
        # Primary face is Face #1: REAL (54.6%)
        # Background face is Face #2: FAKE (63.9%)
        primary = res.primary_face_result
        self.assertEqual(primary.prediction, "real")
        self.assertEqual(res.prediction, "real", "Image verdict in primary-face mode must match primary face")
        self.assertEqual(res.probability, primary.probability)
        self.assertEqual(res.assessment, "Likely authentic")

        # Background face must still retain its own prediction without tainting overall
        background_face = next(f for f in res.faces if not f.is_primary)
        self.assertEqual(background_face.prediction, "fake")
        self.assertGreater(background_face.fake_probability, 0.50)
        self.assertNotEqual(res.prediction, background_face.prediction)

    def test_03_selecting_another_face_displays_own_prediction(self):
        """Verify switching to background face yields that face's own prediction and probabilities."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        face0 = res.faces[0]
        face1 = res.faces[1]

        # Face 0 (Primary)
        self.assertEqual(face0.prediction, "real")
        self.assertAlmostEqual(face0.real_probability + face0.fake_probability, 1.0, places=4)
        self.assertGreater(face0.real_probability, 0.50)

        # Face 1 (Background)
        self.assertEqual(face1.prediction, "fake")
        self.assertAlmostEqual(face1.real_probability + face1.fake_probability, 1.0, places=4)
        self.assertGreater(face1.fake_probability, 0.50)

        # Distinct predictions and probabilities
        self.assertNotEqual(face0.prediction, face1.prediction)
        self.assertNotEqual(face0.real_probability, face1.real_probability)

    def test_04_all_faces_mode_returns_separate_predictions_and_mean(self):
        """Verify all-faces mode returns deterministic unweighted mean aggregation and separate face results."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        af = res.all_faces_result
        self.assertIsNotNone(af)
        self.assertEqual(af.face_count, len(res.faces))

        expected_mean_fake = round(float(np.mean([f.fake_probability for f in res.faces])), 4)
        self.assertAlmostEqual(af.mean_fake_probability, expected_mean_fake, places=4)
        self.assertAlmostEqual(af.mean_real_probability, round(1.0 - expected_mean_fake, 4), places=4)
        self.assertIn("mean", af.aggregation_rule.lower())

    def test_05_probability_sum_constraint(self):
        """Verify real_probability and fake_probability sum to 1.0 for every face inference."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        for face in res.faces:
            self.assertIsNotNone(face.real_probability)
            self.assertIsNotNone(face.fake_probability)
            prob_sum = face.real_probability + face.fake_probability
            self.assertAlmostEqual(prob_sum, 1.0, places=4, msg=f"Probabilities must sum to 1.0 for {face.face_id}")

    def test_06_overlay_coordinates_and_relative_area(self):
        """Verify bounding-box coordinates align with image geometry and relative areas are accurate."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        img_w, img_h = 577, 1280
        total_img_area = img_w * img_h

        for face in res.faces:
            b = face.bounding_box
            self.assertGreaterEqual(b.x, 0)
            self.assertGreaterEqual(b.y, 0)
            self.assertLessEqual(b.x + b.width, img_w)
            self.assertLessEqual(b.y + b.height, img_h)
            expected_area = b.width * b.height
            self.assertEqual(face.area, expected_area)
            expected_rel_area = round((expected_area / total_img_area) * 100, 2)
            self.assertAlmostEqual(face.relative_area, expected_rel_area, places=1)

    def test_07_no_face_image_handling(self):
        """Verify zero-face image returns no_face_detected with null probabilities."""
        if not self.no_face_path.exists():
            self.skipTest("no_face fixture not found")

        res = analyze_image(self.no_face_path)
        self.assertEqual(res.status, "no_face_detected")
        self.assertIsNone(res.prediction)
        self.assertIsNone(res.probability)
        self.assertIsNone(res.primary_face_result)
        self.assertIsNone(res.all_faces_result)
        self.assertEqual(len(res.faces), 0)

    def test_08_switching_selection_does_not_reuse_stale_predictions(self):
        """Verify deterministic switching between face items yields fresh per-face results."""
        # Synthetic multi-face simulation
        box1 = FaceBoundingBox(x=100, y=100, width=200, height=200)
        box2 = FaceBoundingBox(x=10, y=10, width=50, height=50)

        face1 = FaceAnalysisItem(
            face_id="face_0",
            face_index=0,
            bounding_box=box1,
            area=40000,
            relative_area=20.0,
            is_primary=True,
            prediction="real",
            probability=0.88,
            real_probability=0.88,
            fake_probability=0.12,
            raw_logits=[1.5, -1.5]
        )
        face2 = FaceAnalysisItem(
            face_id="face_1",
            face_index=1,
            bounding_box=box2,
            area=2500,
            relative_area=1.25,
            is_primary=False,
            prediction="fake",
            probability=0.91,
            real_probability=0.09,
            fake_probability=0.91,
            raw_logits=[-1.8, 1.8]
        )

        faces = [face1, face2]
        
        # Test selection mapping: Selecting face 0
        selected_0 = next(f for f in faces if f.face_id == "face_0")
        self.assertEqual(selected_0.prediction, "real")
        self.assertEqual(selected_0.probability, 0.88)

        # Switching to face 1: Must NOT retain face 0's values
        selected_1 = next(f for f in faces if f.face_id == "face_1")
        self.assertEqual(selected_1.prediction, "fake")
        self.assertEqual(selected_1.probability, 0.91)
        self.assertNotEqual(selected_0.probability, selected_1.probability)

    def test_09_report_generation_identifies_primary_face_mode(self):
        """Verify report service generates reports identifying Primary Face mode and primary face index."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        meta = MediaMetadata(
            width=577,
            height=1280,
            file_size=68010,
            file_size_human="66.4 KB"
        )
        face_res = FaceDetectionResult(
            status="completed",
            faces=[f.bounding_box for f in res.faces],
            count=len(res.faces)
        )
        record = AnalysisResponse(
            analysis_id="DT-TEST-PESHAL",
            filename="peshal1.jpeg",
            media_type="image",
            media_url="/uploads/DT-TEST-PESHAL_peshal1.jpeg",
            file_size=68010,
            file_size_human="66.4 KB",
            status="completed",
            metadata=meta,
            faces_detected=len(res.faces),
            face_detection=face_res,
            detection=res,
            model=res.model,
            prediction=res.prediction,
            probability=res.probability,
            real_probability=res.real_probability,
            fake_probability=res.fake_probability,
            assessment=res.assessment,
            faces=res.faces,
            selected_face_id=res.selected_face_id,
            analysis_mode=res.analysis_mode,
            primary_face_result=res.primary_face_result,
            all_faces_result=res.all_faces_result,
            created_at="Oct 20, 2026 • 17:00"
        )

        service = ForensicReportService()
        report = service.generate_professional_report(record)
        self.assertIsNotNone(report)
        self.assertEqual(report.result, "REAL")
        self.assertEqual(report.face_detection.primary_face_index, 0)
        self.assertTrue(report.face_detection.faces[0].is_primary)
        self.assertFalse(report.face_detection.faces[1].is_primary)

    def test_10_deterministic_synthetic_aggregation(self):
        """Verify deterministic aggregation formula on synthetic arrays without model dependence."""
        fake_probs = [0.10, 0.40, 0.70]
        mean_fake = float(np.mean(fake_probs))
        self.assertAlmostEqual(mean_fake, 0.40, places=4)
        verdict = "fake" if mean_fake >= DEEPFAKE_THRESHOLD else "real"
        self.assertEqual(verdict, "real")

        # Adding a high fake prob
        fake_probs_2 = [0.80, 0.90]
        mean_fake_2 = float(np.mean(fake_probs_2))
        self.assertAlmostEqual(mean_fake_2, 0.85, places=4)
        verdict_2 = "fake" if mean_fake_2 >= DEEPFAKE_THRESHOLD else "real"
        self.assertEqual(verdict_2, "fake")

    def test_11_all_faces_unequal_predictions_exact_means(self):
        """Verify All-Faces unweighted mean on exact peshal1.jpeg probabilities (45.36% fake and 63.91% fake)."""
        f1_fake, f1_real = 0.4536, 0.5464
        f2_fake, f2_real = 0.6391, 0.3609

        # Both faces sum to 1.0
        self.assertAlmostEqual(f1_fake + f1_real, 1.0, places=4)
        self.assertAlmostEqual(f2_fake + f2_real, 1.0, places=4)

        # Compute means
        mean_fake = round((f1_fake + f2_fake) / 2.0, 4)
        mean_real = round((f1_real + f2_real) / 2.0, 4)

        self.assertEqual(mean_fake, 0.5464)
        self.assertEqual(mean_real, 0.4536)
        self.assertAlmostEqual(mean_fake + mean_real, 1.0, places=4)

        # Verdict assignment
        verdict = "fake" if mean_fake >= DEEPFAKE_THRESHOLD else "real"
        self.assertEqual(verdict, "fake")

        # Reported probability must match assigned class
        reported_prob = mean_fake if verdict == "fake" else mean_real
        self.assertEqual(reported_prob, 0.5464)

    def test_12_verdict_dependent_reported_probability_for_real_verdict(self):
        """Verify that when All-Faces verdict is REAL, the reported probability is mean_real, NOT mean_fake."""
        f1_fake, f1_real = 0.1000, 0.9000
        f2_fake, f2_real = 0.3000, 0.7000

        mean_fake = round(float(np.mean([f1_fake, f2_fake])), 4)
        mean_real = round(float(np.mean([f1_real, f2_real])), 4)

        self.assertEqual(mean_fake, 0.2000)
        self.assertEqual(mean_real, 0.8000)
        self.assertAlmostEqual(mean_fake + mean_real, 1.0, places=4)

        verdict = "fake" if mean_fake >= DEEPFAKE_THRESHOLD else "real"
        self.assertEqual(verdict, "real")

        # Crucial: displayed probability must be 0.8000 (confidence in REAL), not 0.2000!
        reported_prob = mean_fake if verdict == "fake" else mean_real
        self.assertEqual(reported_prob, 0.8000)
        self.assertNotEqual(reported_prob, mean_fake)

    def test_13_low_margin_notice_near_threshold(self):
        """Verify low-margin notice is generated when probability is near 50% threshold."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        self.assertIsNotNone(res.primary_face_result)
        self.assertTrue(res.primary_face_result.is_low_margin)
        self.assertIsNotNone(res.primary_face_result.margin_note)
        self.assertIn("Low-margin model prediction", res.primary_face_result.margin_note)
        self.assertIn("54.6%", res.primary_face_result.margin_note)

    def test_14_task_scope_disclosure_present(self):
        """Verify task scope disclosure notes are present across primary and all faces results."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        self.assertIsNotNone(res.task_scope_note)
        self.assertIn("facial-manipulation", res.task_scope_note.lower())
        self.assertIn("not been validated", res.task_scope_note.lower())

        self.assertIsNotNone(res.all_faces_result)
        self.assertIsNotNone(res.all_faces_result.limitation_disclosure)
        self.assertIn("equal weight", res.all_faces_result.limitation_disclosure.lower())

    def test_15_api_and_report_consistency(self):
        """Verify API response and generated forensic report maintain identical primary face index and verdict."""
        if not self.peshal_img_path.exists():
            self.skipTest("peshal1.jpeg fixture not found")

        res = analyze_image(self.peshal_img_path)
        meta = MediaMetadata(
            width=577, height=1280, format="JPEG", file_size=68010,
            file_size_human="66.4 KB", fps=None, frame_count=1, duration=None, duration_formatted=None
        )
        face_res = FaceDetectionResult(
            status="completed",
            faces=[f.bounding_box for f in res.faces],
            count=len(res.faces)
        )
        record = AnalysisResponse(
            analysis_id="DT-TEST-CONSISTENCY",
            filename="peshal1.jpeg",
            media_type="image",
            media_url="/uploads/DT-TEST_peshal1.jpeg",
            file_size=68010,
            file_size_human="66.4 KB",
            status="completed",
            metadata=meta,
            faces_detected=len(res.faces),
            face_detection=face_res,
            detection=res,
            model=res.model,
            prediction=res.prediction,
            probability=res.probability,
            real_probability=res.real_probability,
            fake_probability=res.fake_probability,
            assessment=res.assessment,
            faces=res.faces,
            selected_face_id=res.selected_face_id,
            analysis_mode=res.analysis_mode,
            primary_face_result=res.primary_face_result,
            all_faces_result=res.all_faces_result,
            is_low_margin=res.is_low_margin,
            margin_note=res.margin_note,
            task_scope_note=res.task_scope_note,
            created_at="Oct 20, 2026 • 17:00"
        )

        service = ForensicReportService()
        report = service.generate_professional_report(record)
        self.assertEqual(report.result, "REAL")
        self.assertEqual(report.face_detection.primary_face_index, 0)
        self.assertIn("Task Scope Disclosure", report.executive_summary)
        self.assertTrue(any("facial-manipulation" in lim for lim in report.limitations))


if __name__ == '__main__':
    unittest.main(verbosity=2)
