"""
Unit and Integration Test Suite for DEEPTRACE Temporal Video Analysis Service (Stage 8F).
Covers deterministic metrics, volatility calculations, transition detection,
bounding box tracking, inter-model temporal agreement, edge cases, and end-to-end video pipeline.
"""

import os
import sys
import unittest
import numpy as np

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.models.temporal_schemas import TemporalFrameItem
from app.services.temporal_analysis_service import (
    TemporalAnalysisService,
    compute_probability_deltas,
    compute_probability_volatility,
    detect_prediction_transitions,
    analyze_face_presence,
    track_face_bounding_boxes,
    compute_model_agreement_over_time,
    extract_temporal_events,
    PROBABILITY_JUMP_THRESHOLD,
    FACE_AREA_CHANGE_THRESHOLD,
    CENTER_DISPLACEMENT_THRESHOLD,
)


class TestTemporalAnalysisService(unittest.TestCase):
    def setUp(self):
        self.service = TemporalAnalysisService()

    # 1. Delta calculation across consecutive frames
    def test_01_compute_probability_deltas_standard(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.20),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_probability=0.65),
            TemporalFrameItem(frame_number=60, timestamp=2.0, valid=True, xception_probability=0.40),
        ]
        deltas = compute_probability_deltas(frames)
        self.assertEqual(len(deltas), 2)
        # Delta 1: 0.65 - 0.20 = +0.45
        self.assertEqual(deltas[0].from_frame, 0)
        self.assertEqual(deltas[0].to_frame, 30)
        self.assertAlmostEqual(deltas[0].delta, 0.45, places=4)
        self.assertAlmostEqual(deltas[0].absolute_delta, 0.45, places=4)
        # Delta 2: 0.40 - 0.65 = -0.25
        self.assertEqual(deltas[1].from_frame, 30)
        self.assertEqual(deltas[1].to_frame, 60)
        self.assertAlmostEqual(deltas[1].delta, -0.25, places=4)
        self.assertAlmostEqual(deltas[1].absolute_delta, 0.25, places=4)

    # 2. Delta calculation with insufficient frames returns empty list
    def test_02_compute_probability_deltas_insufficient(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.85)
        ]
        deltas = compute_probability_deltas(frames)
        self.assertEqual(deltas, [])

    # 3. Probability volatility calculations
    def test_03_compute_probability_volatility_standard(self):
        probs = [0.10, 0.30, 0.80, 0.90]
        frames = [
            TemporalFrameItem(frame_number=i * 30, timestamp=float(i), valid=True, xception_probability=p)
            for i, p in enumerate(probs)
        ]
        deltas = compute_probability_deltas(frames)
        vol = compute_probability_volatility(frames, deltas)
        self.assertIsNotNone(vol)
        self.assertAlmostEqual(vol.mean_probability, float(np.mean(probs)), places=4)
        self.assertAlmostEqual(vol.min_probability, 0.10, places=4)
        self.assertAlmostEqual(vol.max_probability, 0.90, places=4)
        self.assertAlmostEqual(vol.max_absolute_delta, 0.50, places=4)  # 0.80 - 0.30 = 0.50
        self.assertTrue(vol.std_probability > 0.0)

    # 4. Volatility with flat probabilities
    def test_04_compute_probability_volatility_flat(self):
        probs = [0.85, 0.85, 0.85]
        frames = [
            TemporalFrameItem(frame_number=i * 30, timestamp=float(i), valid=True, xception_probability=p)
            for i, p in enumerate(probs)
        ]
        deltas = compute_probability_deltas(frames)
        vol = compute_probability_volatility(frames, deltas)
        self.assertIsNotNone(vol)
        self.assertAlmostEqual(vol.std_probability, 0.0, places=4)
        self.assertAlmostEqual(vol.mean_absolute_delta, 0.0, places=4)
        self.assertAlmostEqual(vol.max_absolute_delta, 0.0, places=4)

    # 5. Prediction transition detection
    def test_05_detect_prediction_transitions(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.2, xception_prediction="authentic"),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_probability=0.8, xception_prediction="manipulated"),
            TemporalFrameItem(frame_number=60, timestamp=2.0, valid=True, xception_probability=0.9, xception_prediction="manipulated"),
            TemporalFrameItem(frame_number=90, timestamp=3.0, valid=True, xception_probability=0.3, xception_prediction="authentic"),
        ]
        transitions = detect_prediction_transitions(frames)
        self.assertEqual(len(transitions), 2)
        # Flip 1: authentic -> manipulated
        self.assertEqual(transitions[0].from_prediction, "authentic")
        self.assertEqual(transitions[0].to_prediction, "manipulated")
        self.assertEqual(transitions[0].frame_before, 0)
        self.assertEqual(transitions[0].frame_after, 30)
        # Flip 2: manipulated -> authentic
        self.assertEqual(transitions[1].from_prediction, "manipulated")
        self.assertEqual(transitions[1].to_prediction, "authentic")
        self.assertEqual(transitions[1].frame_before, 60)
        self.assertEqual(transitions[1].frame_after, 90)

    # 6. Prediction transition detection when stable
    def test_06_detect_prediction_transitions_stable(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.9, xception_prediction="manipulated"),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_probability=0.85, xception_prediction="manipulated"),
            TemporalFrameItem(frame_number=60, timestamp=2.0, valid=True, xception_probability=0.92, xception_prediction="manipulated"),
        ]
        transitions = detect_prediction_transitions(frames)
        self.assertEqual(len(transitions), 0)

    # 7. Face presence counts and percentages
    def test_07_analyze_face_presence_counts(self):
        all_frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, face_count=1),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=False, face_count=0),
            TemporalFrameItem(frame_number=60, timestamp=2.0, valid=True, face_count=1),
            TemporalFrameItem(frame_number=90, timestamp=3.0, valid=True, face_count=1),
        ]
        fp = analyze_face_presence(all_frames)
        self.assertEqual(fp.total_sampled_frames, 4)
        self.assertEqual(fp.frames_with_face, 3)
        self.assertEqual(fp.frames_without_face, 1)
        self.assertAlmostEqual(fp.face_presence_percentage, 75.0, places=1)

    # 8. Face presence streak tracking
    def test_08_analyze_face_presence_streaks(self):
        flags = [True, True, True, False, False, True, True, False]
        all_frames = [
            TemporalFrameItem(frame_number=i * 30, timestamp=float(i), valid=flag, face_count=1 if flag else 0)
            for i, flag in enumerate(flags)
        ]
        fp = analyze_face_presence(all_frames)
        self.assertEqual(fp.longest_consecutive_face_present, 3)
        self.assertEqual(fp.longest_consecutive_face_missing, 2)

    # 9. Face presence with 100% presence
    def test_09_analyze_face_presence_all_present(self):
        all_frames = [
            TemporalFrameItem(frame_number=i * 30, timestamp=float(i), valid=True, face_count=1)
            for i in range(5)
        ]
        fp = analyze_face_presence(all_frames)
        self.assertEqual(fp.face_presence_percentage, 100.0)
        self.assertEqual(fp.longest_consecutive_face_present, 5)
        self.assertEqual(fp.longest_consecutive_face_missing, 0)

    # 10. Face presence with 0% presence
    def test_10_analyze_face_presence_all_missing(self):
        all_frames = [
            TemporalFrameItem(frame_number=i * 30, timestamp=float(i), valid=False, face_count=0)
            for i in range(5)
        ]
        fp = analyze_face_presence(all_frames)
        self.assertEqual(fp.face_presence_percentage, 0.0)
        self.assertEqual(fp.longest_consecutive_face_present, 0)
        self.assertEqual(fp.longest_consecutive_face_missing, 5)

    # 11. Bounding box tracking displacement and areas
    def test_11_track_face_bounding_boxes_metrics(self):
        # 1000x1000 video resolution
        frames = [
            TemporalFrameItem(
                frame_number=0, timestamp=0.0, valid=True,
                face_bbox=[100, 100, 200, 200]  # center=(200, 200), area=40000
            ),
            TemporalFrameItem(
                frame_number=30, timestamp=1.0, valid=True,
                face_bbox=[120, 100, 200, 200]  # center=(220, 200) -> displacement=20/1000=0.02
            )
        ]
        bbox_track = track_face_bounding_boxes(frames, frame_width=1000, frame_height=1000)
        self.assertIsNotNone(bbox_track)
        self.assertAlmostEqual(bbox_track.mean_center_displacement, 0.02, places=3)
        self.assertAlmostEqual(bbox_track.max_center_displacement, 0.02, places=3)
        self.assertAlmostEqual(bbox_track.mean_face_area, 0.04, places=3)  # 40000 / 1000000 = 0.04

    # 12. Bounding box scale change tracking
    def test_12_track_face_bounding_boxes_scale_change(self):
        frames = [
            TemporalFrameItem(
                frame_number=0, timestamp=0.0, valid=True,
                face_bbox=[100, 100, 100, 100]  # area = 10,000
            ),
            TemporalFrameItem(
                frame_number=30, timestamp=1.0, valid=True,
                face_bbox=[100, 100, 200, 200]  # area = 40,000 (area change = 0.03 normalized on 1000x1000)
            )
        ]
        bbox_track = track_face_bounding_boxes(frames, frame_width=1000, frame_height=1000)
        self.assertIsNotNone(bbox_track)
        self.assertAlmostEqual(bbox_track.max_area_delta, 0.03, places=3)

    # 13. Temporal event: probability jump
    def test_13_extract_temporal_events_probability_jump(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.10, xception_prediction="authentic"),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_probability=0.75, xception_prediction="manipulated"),  # Jump +0.65 >= 0.30
        ]
        deltas = compute_probability_deltas(frames)
        transitions = detect_prediction_transitions(frames)
        events = extract_temporal_events(frames, frames, deltas, transitions, frame_width=1000, frame_height=1000)
        jump_events = [e for e in events if e.event_type == "probability_jump"]
        self.assertEqual(len(jump_events), 1)
        self.assertEqual(jump_events[0].frame_number, 30)
        self.assertAlmostEqual(jump_events[0].value, 0.65, places=2)

    # 14. Temporal event: prediction flip
    def test_14_extract_temporal_events_prediction_flip(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_probability=0.48, xception_prediction="authentic"),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_probability=0.52, xception_prediction="manipulated"),
        ]
        deltas = compute_probability_deltas(frames)
        transitions = detect_prediction_transitions(frames)
        events = extract_temporal_events(frames, frames, deltas, transitions, frame_width=1000, frame_height=1000)
        flip_events = [e for e in events if e.event_type == "prediction_transition"]
        self.assertEqual(len(flip_events), 1)
        self.assertEqual(flip_events[0].frame_number, 30)

    # 15. Temporal event: bounding box instability
    def test_15_extract_temporal_events_bbox_instability(self):
        frames = [
            TemporalFrameItem(
                frame_number=0, timestamp=0.0, valid=True, xception_probability=0.5,
                face_bbox=[100, 100, 100, 100]
            ),
            TemporalFrameItem(
                frame_number=30, timestamp=1.0, valid=True, xception_probability=0.5,
                # Jump 300px on 1000px frame = 0.30 displacement >= 0.20 threshold
                face_bbox=[400, 100, 100, 100]
            ),
        ]
        deltas = compute_probability_deltas(frames)
        events = extract_temporal_events(frames, frames, deltas, [], frame_width=1000, frame_height=1000)
        bbox_events = [e for e in events if e.event_type == "face_bbox_change"]
        self.assertEqual(len(bbox_events), 1)

    # 16. Temporal event: model disagreement
    def test_16_extract_temporal_events_model_disagreement(self):
        frames = [
            TemporalFrameItem(
                frame_number=0, timestamp=0.0, valid=True,
                xception_prediction="manipulated", xception_probability=0.88,
                meso4_prediction="authentic", meso4_probability=0.15
            )
        ]
        events = extract_temporal_events(frames, frames, [], [], frame_width=1000, frame_height=1000)
        disagree_events = [e for e in events if e.event_type == "detector_disagreement"]
        self.assertEqual(len(disagree_events), 1)
        self.assertEqual(disagree_events[0].frame_number, 0)

    # 17. Inter-model temporal agreement calculation
    def test_17_compute_model_agreement_over_time(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, xception_prediction="manipulated", meso4_prediction="manipulated", xception_probability=0.8, meso4_probability=0.7),
            TemporalFrameItem(frame_number=30, timestamp=1.0, valid=True, xception_prediction="authentic", meso4_prediction="authentic", xception_probability=0.2, meso4_probability=0.3),
            TemporalFrameItem(frame_number=60, timestamp=2.0, valid=True, xception_prediction="manipulated", meso4_prediction="authentic", xception_probability=0.7, meso4_probability=0.4),
            TemporalFrameItem(frame_number=90, timestamp=3.0, valid=True, xception_prediction="manipulated", meso4_prediction="manipulated", xception_probability=0.9, meso4_probability=0.8),
        ]
        agreement = compute_model_agreement_over_time(frames)
        self.assertIsNotNone(agreement)
        self.assertEqual(agreement.frames_compared, 4)
        self.assertEqual(agreement.agreeing_frames, 3)
        self.assertEqual(agreement.disagreeing_frames, 1)
        self.assertAlmostEqual(agreement.agreement_rate, 0.75, places=2)
        self.assertIsNotNone(agreement.cohen_kappa)
        self.assertIsNotNone(agreement.pearson_correlation)

    # 18. Service edge case: empty frame list
    def test_18_service_edge_case_empty_frames(self):
        result = self.service.analyze_temporal_consistency([])
        self.assertEqual(result.status, "not_available")
        self.assertIn("No video frames sampled", result.reason)
        self.assertEqual(result.analyzed_frames, 0)

    # 19. Service edge case: only 1 valid frame
    def test_19_service_edge_case_single_valid_frame(self):
        frames = [
            TemporalFrameItem(frame_number=0, timestamp=0.0, valid=True, face_count=1, xception_probability=0.7)
        ]
        result = self.service.analyze_temporal_consistency(frames)
        self.assertEqual(result.status, "not_available")
        self.assertIn("Insufficient valid face frames", result.reason)
        self.assertEqual(result.valid_frames, 1)

    # 20. End-to-end integration test with video analysis pipeline
    def test_20_pipeline_video_temporal_integration(self):
        fixture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fixtures", "test_fake_video.mp4")
        if not os.path.exists(fixture_path):
            self.skipTest(f"Test fixture not found: {fixture_path}")

        from app.services.detection_service import analyze_video
        result = analyze_video(fixture_path)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.temporal_analysis)
        self.assertIn(result.temporal_analysis.status, ["available", "not_available"])
        self.assertIsNotNone(result.temporal_analysis.analysis_disclaimer)
        self.assertIn("proof of manipulation", result.temporal_analysis.analysis_disclaimer)

        if result.temporal_analysis.status == "available":
            self.assertGreaterEqual(result.temporal_analysis.valid_frames, 2)
            self.assertIsNotNone(result.temporal_analysis.face_presence)
            self.assertIsNotNone(result.temporal_analysis.statistics)
            self.assertIsNotNone(result.temporal_analysis.deltas)
            self.assertIsNotNone(result.temporal_analysis.temporal_events)


if __name__ == "__main__":
    unittest.main()
