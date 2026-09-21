"""
Unit and Integration Test Suite for DEEPTRACE Forensic Reporting (Stage 8G).
Covers all 20 required report scenarios: image/video reports, face presence/absence,
Grad-CAM, FFT, metadata, temporal dynamics, cross-model comparison, missing artifacts,
immutability/persistence, SHA-256 consistency, sanitization, security, and printable export.
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.schemas import (
    AnalysisResponse,
    MediaMetadata,
    FaceDetectionResult,
    DetectionResult,
    FaceAnalysisItem,
    VideoFrameAnalysisItem,
    MetadataAnalysisResult,
    FrequencyAnalysisResult
)
from app.models.temporal_schemas import (
    TemporalAnalysisResult,
    TemporalFacePresence,
    TemporalProbabilityVolatility,
    TemporalProbabilityDelta,
    TemporalPredictionTransition,
    TemporalEvent,
    TemporalModelAgreement
)
from app.models.report_schemas import DetailedForensicReport
from app.services.report_service import ForensicReportService, sanitize_path, compute_file_sha256


class TestForensicReporting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = ForensicReportService(reports_dir=Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_analysis(
        self,
        analysis_id: str = "DT-2026-TEST01",
        media_type: str = "image",
        status: str = "completed",
        faces_detected: int = 1,
        probability: float = 0.85,
        prediction: str = "fake",
        gradcam: bool = True,
        frequency: bool = True,
        metadata_present: bool = True,
        temporal: bool = False
    ) -> AnalysisResponse:
        meta = MediaMetadata(
            width=640, height=480, file_size=50000,
            file_size_human="50.0 KB", format="JPEG",
            frame_count=1 if media_type == "image" else 30,
            fps=30.0 if media_type == "video" else None,
            duration=1.0 if media_type == "video" else None,
            duration_formatted="00:01" if media_type == "video" else None
        )

        face_items = []
        for i in range(faces_detected):
            face_items.append(
                FaceAnalysisItem(
                    face_index=i,
                    bounding_box={"x": 50 * i, "y": 50, "width": 100, "height": 100},
                    prediction=prediction,
                    probability=probability
                )
            )

        face_result = FaceDetectionResult(
            status=status if faces_detected > 0 else "no_face_detected",
            faces=[f.bounding_box for f in face_items],
            count=faces_detected
        )

        freq_res = None
        if frequency:
            freq_res = FrequencyAnalysisResult(
                available=True,
                analysis_region="face_region" if faces_detected > 0 else "full_image",
                frequency_spectrum_url="/uploads/forensics/frequency/spec.jpg",
                high_frequency_ratio=0.185,
                spectral_centroid=0.32,
                total_energy=15000.0,
                low_frequency_energy=9000.0,
                mid_frequency_energy=3500.0,
                high_frequency_energy=2500.0
            )

        meta_res = None
        if metadata_present:
            meta_res = MetadataAnalysisResult(
                available=True,
                metadata_present=True,
                metadata_fields_count=12,
                format="JPEG",
                width=640,
                height=480,
                camera_make="Canon",
                camera_model="EOS 5D",
                software="Adobe Photoshop CC",
                editing_software_detected=True
            )
        else:
            meta_res = MetadataAnalysisResult(
                available=True,
                metadata_present=False,
                metadata_fields_count=0
            )

        temp_res = None
        if temporal and media_type == "video":
            temp_res = TemporalAnalysisResult(
                status="available",
                analyzed_frames=2,
                valid_frames=2,
                face_presence=TemporalFacePresence(
                    total_sampled_frames=2,
                    frames_with_face=2,
                    frames_without_face=0,
                    face_presence_percentage=100.0,
                    longest_consecutive_face_present=2,
                    longest_consecutive_face_missing=0
                ),
                statistics=TemporalProbabilityVolatility(
                    mean_probability=probability,
                    median_probability=probability,
                    min_probability=probability - 0.05,
                    max_probability=probability + 0.05,
                    std_probability=0.035,
                    mean_absolute_delta=0.05,
                    max_absolute_delta=0.05
                ),
                deltas=[
                    TemporalProbabilityDelta(
                        from_frame=0, to_frame=15, from_timestamp=0.0, to_timestamp=0.5,
                        delta=0.05, absolute_delta=0.05
                    )
                ],
                prediction_transitions=[],
                temporal_events=[
                    TemporalEvent(
                        event_id="evt_01", frame_number=15, timestamp=0.5,
                        event_type="probability_jump", value=0.35, threshold=0.30,
                        severity="notable", description="Probability jump"
                    )
                ],
                model_agreement=TemporalModelAgreement(
                    frames_compared=2, agreeing_frames=2, disagreeing_frames=0,
                    agreement_rate=1.0, cohen_kappa=1.0, pearson_correlation=0.98
                )
            )

        frame_results = None
        if media_type == "video":
            frame_results = [
                VideoFrameAnalysisItem(
                    frame_number=0, timestamp=0.0, face_detected=faces_detected > 0,
                    prediction=prediction, probability=probability,
                    bounding_box={"x": 50, "y": 50, "width": 100, "height": 100} if faces_detected > 0 else None,
                    gradcam_available=gradcam,
                    gradcam_overlay_url="/uploads/f0_overlay.jpg" if gradcam else None
                ),
                VideoFrameAnalysisItem(
                    frame_number=15, timestamp=0.5, face_detected=faces_detected > 0,
                    prediction=prediction, probability=probability,
                    bounding_box={"x": 52, "y": 50, "width": 100, "height": 100} if faces_detected > 0 else None,
                    gradcam_available=gradcam,
                    gradcam_overlay_url="/uploads/f15_overlay.jpg" if gradcam else None
                )
            ]

        det_result = DetectionResult(
            status=status if faces_detected > 0 else "no_face_detected",
            model="DeepfakeBench Xception",
            prediction=prediction if faces_detected > 0 else None,
            probability=probability if faces_detected > 0 else None,
            faces=face_items,
            frame_results=frame_results,
            frames_analyzed=len(frame_results) if frame_results else 1,
            faces_detected=faces_detected,
            gradcam_available=gradcam,
            gradcam_heatmap_url="/uploads/heatmap.jpg" if gradcam else None,
            gradcam_overlay_url="/uploads/overlay.jpg" if gradcam else None,
            original_face_crop_url="/uploads/face.jpg" if gradcam else None,
            metadata_analysis=meta_res,
            frequency_analysis=freq_res,
            temporal_analysis=temp_res
        )

        return AnalysisResponse(
            analysis_id=analysis_id,
            filename="sample_evidence.jpg" if media_type == "image" else "sample_evidence.mp4",
            media_type=media_type,
            file_size=50000,
            file_size_human="50.0 KB",
            status=status if faces_detected > 0 else "no_face_detected",
            metadata=meta,
            faces_detected=faces_detected,
            face_detection=face_result,
            detection=det_result,
            model="DeepfakeBench Xception",
            prediction=prediction if faces_detected > 0 else None,
            probability=probability if faces_detected > 0 else None,
            faces=face_items,
            frame_results=frame_results,
            frames_analyzed=len(frame_results) if frame_results else 1,
            gradcam_available=gradcam,
            gradcam_heatmap_url="/uploads/heatmap.jpg" if gradcam else None,
            gradcam_overlay_url="/uploads/overlay.jpg" if gradcam else None,
            original_face_crop_url="/uploads/face.jpg" if gradcam else None,
            metadata_analysis=meta_res,
            frequency_analysis=freq_res,
            temporal_analysis=temp_res,
            created_at="2026-09-20 20:00:00"
        )

    # 1. Image report with face
    def test_01_image_report_with_face(self):
        rec = self._create_mock_analysis(media_type="image", faces_detected=1, probability=0.88, prediction="fake")
        report = self.service.generate_professional_report(rec)
        self.assertIsInstance(report, DetailedForensicReport)
        self.assertEqual(report.result, "LIKELY MANIPULATED")
        self.assertEqual(report.face_detection.faces_detected, 1)
        self.assertIn("Xception", report.environment.primary_detector_name)
        self.assertTrue(report.primary_detector.confidence_score > 50.0)

    # 2. Image report without face
    def test_02_image_report_without_face(self):
        rec = self._create_mock_analysis(media_type="image", faces_detected=0, prediction=None)
        report = self.service.generate_professional_report(rec)
        self.assertEqual(report.result, "NO FACE DETECTED")
        self.assertEqual(report.primary_detector.verdict, "NO FACE DETECTED")
        self.assertEqual(report.face_detection.faces_detected, 0)
        self.assertIn("No facial structures", report.primary_detector.assessment)

    # 3. Video report
    def test_03_video_report(self):
        rec = self._create_mock_analysis(media_type="video", faces_detected=1, temporal=True)
        report = self.service.generate_professional_report(rec)
        self.assertEqual(report.mediaType, "video")
        self.assertIsNotNone(report.video_evidence)
        self.assertEqual(report.video_evidence.sampled_frames_count, 2)
        self.assertEqual(report.video_evidence.valid_face_frames_count, 2)

    # 4. Multiple-face report
    def test_04_multiple_face_report(self):
        rec = self._create_mock_analysis(media_type="image", faces_detected=3)
        report = self.service.generate_professional_report(rec)
        self.assertEqual(report.face_detection.faces_detected, 3)
        self.assertEqual(len(report.face_detection.faces), 3)
        self.assertTrue(report.face_detection.faces[0].is_primary)

    # 5. Grad-CAM available
    def test_05_gradcam_available(self):
        rec = self._create_mock_analysis(gradcam=True)
        report = self.service.generate_professional_report(rec)
        self.assertTrue(report.gradcam_evidence.available)
        self.assertEqual(report.gradcam_evidence.overlay_url, "/uploads/overlay.jpg")
        self.assertIn("conv4", report.gradcam_evidence.target_layer)

    # 6. Grad-CAM unavailable
    def test_06_gradcam_unavailable(self):
        rec = self._create_mock_analysis(gradcam=False)
        report = self.service.generate_professional_report(rec)
        self.assertFalse(report.gradcam_evidence.available)
        self.assertIsNone(report.gradcam_evidence.overlay_url)

    # 7. Frequency analysis available
    def test_07_frequency_available(self):
        rec = self._create_mock_analysis(frequency=True)
        report = self.service.generate_professional_report(rec)
        self.assertTrue(report.frequency_evidence.available)
        self.assertEqual(report.frequency_evidence.analysis_region, "face_region")
        self.assertAlmostEqual(report.frequency_evidence.high_frequency_percentage, 18.5, places=1)

    # 8. Metadata present
    def test_08_metadata_present(self):
        rec = self._create_mock_analysis(metadata_present=True)
        report = self.service.generate_professional_report(rec)
        self.assertTrue(report.metadata_evidence.metadata_present)
        self.assertEqual(report.metadata_evidence.camera_make, "Canon")
        self.assertTrue(report.metadata_evidence.editing_software_detected)

    # 9. Metadata absent (neutral disclaimer)
    def test_09_metadata_absent(self):
        rec = self._create_mock_analysis(metadata_present=False)
        report = self.service.generate_professional_report(rec)
        self.assertFalse(report.metadata_evidence.metadata_present)
        self.assertIn("not, by itself, indicate malicious tampering", report.metadata_evidence.interpretation_disclaimer)

    # 10. Temporal data present
    def test_10_temporal_data_present(self):
        rec = self._create_mock_analysis(media_type="video", temporal=True)
        report = self.service.generate_professional_report(rec)
        self.assertIsNotNone(report.temporal_evidence)
        self.assertEqual(report.temporal_evidence.status, "available")
        self.assertEqual(report.temporal_evidence.face_presence_percentage, 100.0)
        self.assertEqual(report.temporal_evidence.temporal_events_count, 1)

    # 11. Temporal data absent
    def test_11_temporal_data_absent(self):
        rec = self._create_mock_analysis(media_type="video", temporal=False)
        report = self.service.generate_professional_report(rec)
        self.assertIsNone(report.temporal_evidence)

    # 12. Meso4 comparison available
    def test_12_meso4_comparison_available(self):
        rec = self._create_mock_analysis(media_type="video", temporal=True)
        report = self.service.generate_professional_report(rec)
        self.assertIsNotNone(report.cross_model_comparison)
        self.assertTrue(report.cross_model_comparison.evaluated)
        self.assertEqual(report.cross_model_comparison.primary_model_name, "Xception (DeepfakeBench)")
        self.assertEqual(report.cross_model_comparison.comparison_model_name, "MesoNet-4 (Afchar et al.)")

    # 13. Meso4 comparison unavailable
    def test_13_meso4_comparison_unavailable(self):
        rec = self._create_mock_analysis(media_type="image", temporal=False)
        report = self.service.generate_professional_report(rec)
        self.assertIsNone(report.cross_model_comparison)

    # 14. Missing optional artifacts handled gracefully
    def test_14_missing_optional_artifacts(self):
        rec = self._create_mock_analysis(gradcam=False, frequency=False, metadata_present=False, temporal=False)
        report = self.service.generate_professional_report(rec)
        self.assertFalse(report.gradcam_evidence.available)
        self.assertFalse(report.frequency_evidence.available)
        self.assertFalse(report.metadata_evidence.metadata_present)

    # 15. Historical report snapshot persistence and retrieval
    def test_15_report_persistence(self):
        rec = self._create_mock_analysis(analysis_id="DT-2026-PERSIST01")
        report = self.service.generate_professional_report(rec)
        # Verify file exists on disk
        persisted_file = self.service.reports_dir / f"{report.id}.json"
        self.assertTrue(persisted_file.exists())
        # Load persisted
        loaded = self.service.load_persisted_report(report.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.id, report.id)
        self.assertEqual(loaded.caseId, report.caseId)
        self.assertEqual(loaded.hashSha256, report.hashSha256)

    # 16. Evidence hash consistency
    def test_16_evidence_hash_consistency(self):
        rec = self._create_mock_analysis(analysis_id="DT-2026-HASH01")
        h1 = compute_file_sha256(rec.analysis_id, rec.filename)
        h2 = compute_file_sha256(rec.analysis_id, rec.filename)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    # 17. Filename sanitization
    def test_17_filename_sanitization(self):
        dirty_name = "../../../etc/passwd"
        cleaned = sanitize_path(dirty_name)
        self.assertEqual(cleaned, "passwd")

        windows_dirty = "C:\\Windows\\System32\\cmd.exe"
        cleaned_win = sanitize_path(windows_dirty)
        self.assertEqual(cleaned_win, "cmd.exe")

    # 18. Security: No absolute filesystem paths exposed in report
    def test_18_no_absolute_paths_exposed(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        json_str = report.model_dump_json()
        self.assertNotIn("c:\\Users\\", json_str.lower())
        self.assertNotIn("/users/", json_str.lower())
        self.assertNotIn("/home/", json_str.lower())

    # 19. Security: No secrets, API keys, or stack traces
    def test_19_no_secrets_exposed(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        json_str = report.model_dump_json()
        self.assertNotIn("secret", json_str.lower())
        self.assertNotIn("password", json_str.lower())
        self.assertNotIn("traceback", json_str.lower())

    # 20. Printable HTML report generation
    def test_20_printable_html_export(self):
        rec = self._create_mock_analysis(temporal=True, media_type="video")
        report = self.service.generate_professional_report(rec)
        html = self.service.render_printable_report_html(report)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn(report.id, html)
        self.assertIn(report.hashSha256, html)
        self.assertIn("@media print", html)
        self.assertIn("Executive Summary", html)
        self.assertIn("Technical Limitations", html)

    # 21. No overall weighted forensic score
    def test_21_no_overall_weighted_forensic_score(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        report_dict = report.model_dump()
        self.assertNotIn("overall_score", report_dict)
        self.assertNotIn("evidence_score", report_dict)
        self.assertNotIn("forensic_score", report_dict)
        self.assertNotIn("weighted_score", report_dict)
        self.assertNotIn("severity_score", report_dict)

    # 22. No Low/Medium/High/Critical synthetic evidence rating
    def test_22_no_synthetic_evidence_severity_ratings(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        for cat in report.evidence_summary_categories:
            self.assertNotIn(cat.status.lower(), ["critical", "high risk", "medium risk", "low risk"])
            self.assertIn(cat.status, ["COMPLETED", "AVAILABLE", "UNAVAILABLE", "PRESENT", "ABSENT", "LOCATED", "NO FACE DETECTED", "ALIGNED", "DIVERGENT", "NOT EVALUATED"])

    # 23. Probability is labeled model probability or manipulated-class softmax probability
    def test_23_probability_labeled_model_probability(self):
        rec = self._create_mock_analysis(probability=0.813)
        report = self.service.generate_professional_report(rec)
        self.assertEqual(report.primary_detector.model_probability, 0.813)
        self.assertEqual(report.primary_detector.manipulated_class_softmax_probability, 0.813)
        self.assertIn("manipulated-class softmax", report.primary_detector.classification_note.lower())

    # 24. No calibrated probability claim
    def test_24_no_calibrated_probability_claim(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        json_dump = report.model_dump_json().lower()
        self.assertNotIn("calibrated probability", json_dump)
        self.assertNotIn("calibrated_probability", json_dump)

    # 25. Executive summary contains no synthetic overall severity
    def test_25_executive_summary_contains_no_synthetic_severity(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        summary_lower = report.executive_summary.lower()
        self.assertNotIn("overall forensic risk", summary_lower)
        self.assertNotIn("critical manipulation", summary_lower)
        self.assertNotIn("evidence score", summary_lower)
        self.assertNotIn("forensic severity", summary_lower)
        self.assertIn("automated analysis statement", summary_lower)

    # 26. Automated analysis statement present, no fake certification
    def test_26_automated_analysis_statement_present_no_fake_certification(self):
        rec = self._create_mock_analysis()
        report = self.service.generate_professional_report(rec)
        self.assertTrue(hasattr(report, "automated_analysis_statement"))
        self.assertIn("does not constitute human examiner certification", report.automated_analysis_statement)
        self.assertIn("Automated Analysis Engine", report.examiner)

    # 27. Evidence domains remain independent
    def test_27_evidence_domains_remain_independent(self):
        rec = self._create_mock_analysis(temporal=True, media_type="video")
        report = self.service.generate_professional_report(rec)
        categories = [c.category for c in report.evidence_summary_categories]
        self.assertIn("PRIMARY MODEL OUTPUT", categories)
        self.assertIn("MODEL EXPLAINABILITY", categories)
        self.assertIn("FREQUENCY ANALYSIS", categories)
        self.assertIn("METADATA", categories)
        self.assertIn("FACIAL LOCALIZATION", categories)
        self.assertIn("VIDEO FRAME ANALYSIS", categories)
        self.assertIn("TEMPORAL DYNAMICS", categories)

    # 28. Historical reports remain readable and backward compatible
    def test_28_historical_reports_readable(self):
        rec = self._create_mock_analysis(analysis_id="DT-2026-HIST01")
        report = self.service.generate_professional_report(rec)
        persisted = self.service.load_persisted_report(report.id)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted.result, "LIKELY MANIPULATED")
        self.assertEqual(persisted.confidence, 85.0)
        self.assertEqual(persisted.caseId, "DT-2026-HIST01")
        self.assertTrue(len(persisted.hashSha256) == 64)


if __name__ == "__main__":
    unittest.main()
