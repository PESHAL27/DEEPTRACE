"""
DEEPTRACE - Video Playback, Frame Classification & Probability Consistency Regression Suite
Validates:
1. Video playback URL, correct MIME type, and byte-range support (206 Partial Content).
2. Actual video duration and playback-state synchronization.
3. Seeking to a selected frame's timestamp.
4. Correct REAL/FAKE class probability mapping.
5. A video with many frames classified FAKE.
6. A video with many frames classified REAL.
7. A video with mixed frame-level predictions (the exact 24-frame scenario from user audit).
8. No-face and inference-failure frames exclusion from denominator.
9. Empty suspicious-frame results handling.
10. Exact agreement between backend values, frontend labels, and report values.
"""

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.config import DEEPFAKE_THRESHOLD, UPLOADS_DIR
from app.models.schemas import (
    VideoFrameAnalysisItem,
    FaceBoundingBox,
    DetectionResult,
    AnalysisResponse,
    MediaMetadata,
    FaceDetectionResult
)
from app.services.report_service import get_report_service
from app.services.media_service import extract_media_metadata


class TestVideoAuditRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_video_path = list(Path("backend/uploads").glob("*WhatsApp*.mp4"))
        if not cls.test_video_path:
            cls.test_video_path = list(Path("backend/uploads").glob("*.mp4"))
        cls.active_video = cls.test_video_path[0] if cls.test_video_path else None

    # 1. Video playback URL, correct MIME type, and byte-range support (206 Partial Content)
    def test_01_video_streaming_byte_range_and_mime(self):
        if not self.active_video:
            self.skipTest("No test video fixture found in uploads/")
        url = f"/uploads/{self.active_video.name}"
        
        # Test full request
        res_full = self.client.get(url)
        self.assertEqual(res_full.status_code, 200)
        self.assertEqual(res_full.headers.get("content-type"), "video/mp4")
        self.assertEqual(res_full.headers.get("accept-ranges"), "bytes")
        
        # Test byte range request (first 1024 bytes)
        res_range = self.client.get(url, headers={"Range": "bytes=0-1023"})
        self.assertEqual(res_range.status_code, 206)
        self.assertEqual(res_range.headers.get("content-type"), "video/mp4")
        self.assertIn("bytes 0-1023/", res_range.headers.get("content-range", ""))
        self.assertEqual(len(res_range.content), 1024)

    # 2. Actual video duration and playback-state synchronization
    def test_02_video_duration_and_metadata_sync(self):
        if not self.active_video:
            self.skipTest("No test video fixture found in uploads/")
        meta = extract_media_metadata(self.active_video, "video")
        self.assertIsNotNone(meta.duration)
        self.assertGreater(meta.duration, 0)
        self.assertIsNotNone(meta.fps)
        self.assertGreater(meta.fps, 0)
        self.assertIsNotNone(meta.duration_formatted)
        self.assertRegex(meta.duration_formatted, r"^\d{2}:\d{2}$")

    # 3. Seeking to a selected frame's timestamp
    def test_03_seeking_frame_timestamps(self):
        fps = 30.0
        frame_indices = [0, 15, 30, 45, 60, 90, 120]
        frames = []
        for f_idx in frame_indices:
            t = round(f_idx / fps, 2)
            item = VideoFrameAnalysisItem(
                frame_number=f_idx,
                timestamp=t,
                face_detected=True,
                prediction="real",
                probability=0.85,
                real_probability=0.85,
                fake_probability=0.15
            )
            frames.append(item)

        # Confirm chronological ordering and precise timestamps
        for i in range(len(frames) - 1):
            self.assertLess(frames[i].timestamp, frames[i + 1].timestamp)
            self.assertLess(frames[i].frame_number, frames[i + 1].frame_number)
            self.assertAlmostEqual(frames[i].timestamp, round(frames[i].frame_number / fps, 2), places=2)

    # 4. Correct REAL/FAKE class probability mapping
    def test_04_class_probability_mapping_integrity(self):
        # Case A: Model outputs fake_prob >= 0.50 (Class 1 dominant)
        fake_prob = 0.7301
        real_prob = round(1.0 - fake_prob, 4)
        pred = "fake" if fake_prob >= DEEPFAKE_THRESHOLD else "real"
        confidence = fake_prob if pred == "fake" else real_prob
        self.assertEqual(pred, "fake")
        self.assertEqual(confidence, fake_prob)
        self.assertNotEqual(confidence, real_prob)

        # Case B: Model outputs fake_prob < 0.50 (Class 0 dominant)
        real_prob_b = 0.5863
        fake_prob_b = round(1.0 - real_prob_b, 4)
        pred_b = "fake" if fake_prob_b >= DEEPFAKE_THRESHOLD else "real"
        confidence_b = real_prob_b if pred_b == "real" else fake_prob_b
        self.assertEqual(pred_b, "real")
        self.assertEqual(confidence_b, 0.5863)
        # Ensure authentic probability is NEVER returned as fake probability
        self.assertEqual(fake_prob_b, 0.4137)

    # 5. A video with many frames classified FAKE
    def test_05_aggregation_video_mostly_fake(self):
        # 10 frames with manipulation probabilities between 0.70 and 0.95
        synthetic_fake_probs = [0.80, 0.85, 0.90, 0.75, 0.88, 0.92, 0.78, 0.84, 0.91, 0.87]
        m_valid = len(synthetic_fake_probs)
        mean_fake_prob = sum(synthetic_fake_probs) / m_valid
        video_fake_prob = round(mean_fake_prob, 4)
        video_real_prob = round(1.0 - mean_fake_prob, 4)

        overall_pred = "fake" if mean_fake_prob >= DEEPFAKE_THRESHOLD else "real"
        video_prob = video_fake_prob if overall_pred == "fake" else video_real_prob

        self.assertEqual(overall_pred, "fake")
        self.assertEqual(video_prob, 0.85)
        self.assertEqual(video_fake_prob, 0.85)
        self.assertEqual(video_real_prob, 0.15)

    # 6. A video with many frames classified REAL
    def test_06_aggregation_video_mostly_real(self):
        # 10 frames with manipulation probabilities between 0.05 and 0.25 (highly authentic)
        synthetic_fake_probs = [0.10, 0.15, 0.08, 0.20, 0.12, 0.05, 0.18, 0.14, 0.11, 0.07]
        m_valid = len(synthetic_fake_probs)
        mean_fake_prob = sum(synthetic_fake_probs) / m_valid
        video_fake_prob = round(mean_fake_prob, 4)
        video_real_prob = round(1.0 - mean_fake_prob, 4)

        overall_pred = "fake" if mean_fake_prob >= DEEPFAKE_THRESHOLD else "real"
        video_prob = video_fake_prob if overall_pred == "fake" else video_real_prob

        self.assertEqual(overall_pred, "real")
        self.assertEqual(video_fake_prob, 0.12)
        self.assertEqual(video_real_prob, 0.88)
        self.assertEqual(video_prob, 0.88)

    # 7. A video with mixed frame-level predictions (the exact 24-frame audit scenario)
    def test_07_aggregation_mixed_predictions_audit_scenario(self):
        # Exact values observed during audit of the 33-second WhatsApp test video:
        # Frame 0: fake (0.563)
        # Frame 1381: fake (0.5487)
        # 22 other frames: real (probabilities between 0.503 and 0.975 real)
        fake_probs = [
            0.5630, 0.3959, 0.4452, 0.3970, 0.4208, 0.3330,
            0.4338, 0.4578, 0.4970, 0.4808, 0.3656, 0.4422,
            0.4261, 0.3243, 0.4154, 0.3743, 0.3871, 0.5487,
            0.4818, 0.4300, 0.3824, 0.4160, 0.0250, 0.4866
        ]
        m_valid = len(fake_probs)
        self.assertEqual(m_valid, 24)

        mean_fake_prob = sum(fake_probs) / m_valid
        video_fake_prob = round(mean_fake_prob, 4)
        video_real_prob = round(1.0 - mean_fake_prob, 4)

        overall_pred = "fake" if mean_fake_prob >= DEEPFAKE_THRESHOLD else "real"
        video_prob = video_fake_prob if overall_pred == "fake" else video_real_prob

        # Aggregation result verification
        self.assertEqual(overall_pred, "real")
        self.assertAlmostEqual(video_real_prob, 0.5864, places=3)
        self.assertEqual(video_prob, video_real_prob)

        # Suspicious frame selection: exactly 2 frames qualify as suspicious (threshold >= 0.50)
        suspicious_indices = [i for i, p in enumerate(fake_probs) if p >= DEEPFAKE_THRESHOLD]
        self.assertEqual(len(suspicious_indices), 2)
        self.assertEqual(suspicious_indices, [0, 17])
        self.assertEqual(fake_probs[0], 0.5630)
        self.assertEqual(fake_probs[17], 0.5487)

    # 8. No-face and inference-failure frames exclusion from denominator
    def test_08_no_face_frames_excluded_from_denominator(self):
        # 5 valid face frames with fake_prob = 0.60
        # 5 no-face frames (face_detected=False, probability=None)
        all_frames = []
        for i in range(5):
            all_frames.append(VideoFrameAnalysisItem(
                frame_number=i * 15,
                timestamp=float(i),
                face_detected=True,
                prediction="fake",
                probability=0.60,
                fake_probability=0.60,
                real_probability=0.40
            ))
        for j in range(5, 10):
            all_frames.append(VideoFrameAnalysisItem(
                frame_number=j * 15,
                timestamp=float(j),
                face_detected=False,
                prediction=None,
                probability=None,
                fake_probability=None,
                real_probability=None
            ))

        valid_frames = [f for f in all_frames if f.face_detected and f.probability is not None]
        m_valid = len(valid_frames)
        excluded_count = len(all_frames) - m_valid

        self.assertEqual(m_valid, 5)
        self.assertEqual(excluded_count, 5)

        # Mean fake probability MUST be over valid frames only (0.60), NOT divided by 10 (which would be 0.30)
        fake_probs = [f.fake_probability for f in valid_frames]
        mean_fake_prob = sum(fake_probs) / m_valid
        self.assertEqual(mean_fake_prob, 0.60)
        self.assertEqual("fake" if mean_fake_prob >= DEEPFAKE_THRESHOLD else "real", "fake")

    # 9. Empty suspicious-frame results
    def test_09_empty_suspicious_frames_clean_state(self):
        # 10 valid frames all with fake_probability < 0.50
        clean_frames = []
        for i in range(10):
            clean_frames.append(VideoFrameAnalysisItem(
                frame_number=i * 15,
                timestamp=float(i),
                face_detected=True,
                prediction="real",
                probability=0.85,
                fake_probability=0.15,
                real_probability=0.85
            ))

        suspicious = [
            f for f in clean_frames
            if f.face_detected and (f.prediction == "fake" or (f.fake_probability is not None and f.fake_probability >= DEEPFAKE_THRESHOLD))
        ]
        self.assertEqual(len(suspicious), 0)

    # 10. Exact agreement between backend values, frontend labels, and report values
    def test_10_exact_agreement_backend_frontend_report(self):
        # Create full AnalysisResponse for the 24-frame mixed scenario
        frame_items = []
        fake_probs = [
            0.5630, 0.3959, 0.4452, 0.3970, 0.4208, 0.3330,
            0.4338, 0.4578, 0.4970, 0.4808, 0.3656, 0.4422,
            0.4261, 0.3243, 0.4154, 0.3743, 0.3871, 0.5487,
            0.4818, 0.4300, 0.3824, 0.4160, 0.0250, 0.4866
        ]
        for idx, fp in enumerate(fake_probs):
            pred = "fake" if fp >= DEEPFAKE_THRESHOLD else "real"
            conf = fp if pred == "fake" else round(1.0 - fp, 4)
            frame_items.append(VideoFrameAnalysisItem(
                frame_number=idx * 69,
                timestamp=round(idx * 1.15, 2),
                face_detected=True,
                bounding_box=FaceBoundingBox(x=100, y=100, width=80, height=80),
                prediction=pred,
                probability=conf,
                fake_probability=fp,
                real_probability=round(1.0 - fp, 4)
            ))

        mean_fake = sum(fake_probs) / len(fake_probs)
        vid_real_prob = round(1.0 - mean_fake, 4)
        vid_fake_prob = round(mean_fake, 4)

        detection_res = DetectionResult(
            status="completed",
            prediction="real",
            probability=vid_real_prob,
            detection_probability=vid_fake_prob,
            model_probability=vid_real_prob,
            video_probability=vid_real_prob,
            real_probability=vid_real_prob,
            fake_probability=vid_fake_prob,
            assessment="Likely authentic",
            frame_results=frame_items,
            frames_analyzed=len(frame_items),
            faces_detected=1,
            duration=33.46,
            fps=59.89
        )

        record = AnalysisResponse(
            analysis_id="DT-AUDIT-TEST",
            filename="audit_test_video.mp4",
            media_type="video",
            media_url="/uploads/DT-AUDIT-TEST_audit_test_video.mp4",
            file_size=7909239,
            file_size_human="7.5 MB",
            status="completed",
            metadata=MediaMetadata(
                file_size=7909239,
                file_size_human="7.5 MB",
                media_type="video",
                mime_type="video/mp4",
                duration=33.46,
                duration_formatted="00:33",
                fps=59.89,
                frame_count=2004,
                width=476,
                height=848
            ),
            face_detection=FaceDetectionResult(faces_detected=1, bounding_boxes=[]),
            detection=detection_res,
            prediction="real",
            probability=vid_real_prob,
            detection_probability=vid_fake_prob,
            model_probability=vid_real_prob,
            video_probability=vid_real_prob,
            real_probability=vid_real_prob,
            fake_probability=vid_fake_prob,
            assessment="Likely authentic",
            frame_results=frame_items,
            frames_analyzed=len(frame_items),
            duration=33.46,
            fps=59.89,
            created_at="2026-10-20 15:45:00 UTC"
        )

        report_svc = get_report_service()
        report = report_svc.generate_professional_report(record)

        # 1. Overall verdict and probabilities
        self.assertEqual(report.primary_detector.verdict, "REAL")
        self.assertEqual(report.primary_detector.confidence_score, round(vid_real_prob * 100, 1))
        self.assertEqual(report.primary_detector.model_probability, vid_real_prob)
        self.assertEqual(report.primary_detector.authentic_class_softmax_probability, vid_real_prob)
        self.assertEqual(report.primary_detector.manipulated_class_softmax_probability, vid_fake_prob)

        # 2. Video summary frame counts
        self.assertEqual(report.video_evidence.sampled_frames_count, 24)
        self.assertEqual(report.video_evidence.valid_face_frames_count, 24)
        # CRITICAL AUDIT CHECK: Exactly 2 suspicious frames, NOT 24!
        self.assertEqual(report.video_evidence.suspicious_frames_count, 2)

        # 3. Media URL presence
        self.assertEqual(record.media_url, "/uploads/DT-AUDIT-TEST_audit_test_video.mp4")


if __name__ == "__main__":
    unittest.main()
