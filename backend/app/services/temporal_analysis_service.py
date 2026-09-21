import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from ..models.temporal_schemas import (
    TemporalFrameItem,
    TemporalProbabilityDelta,
    TemporalProbabilityVolatility,
    TemporalPredictionTransition,
    TemporalFacePresence,
    TemporalBBoxTracking,
    TemporalEvent,
    TemporalModelAgreement,
    TemporalAnalysisResult
)
from ..utils.logger import logger

# Configurable Analysis Thresholds (Engineering thresholds, not proven manipulation bounds)
PROBABILITY_JUMP_THRESHOLD: float = 0.30
FACE_AREA_CHANGE_THRESHOLD: float = 0.40
CENTER_DISPLACEMENT_THRESHOLD: float = 0.20


def compute_probability_deltas(
    valid_frames: List[TemporalFrameItem]
) -> List[TemporalProbabilityDelta]:
    """
    Calculates the step in manipulated probability between consecutive valid frames.
    For frames i and i+1:
      delta_i = P(i+1) - P(i)
      absolute_delta_i = |P(i+1) - P(i)|
    Deltas are strictly computed across consecutive valid frames without bridging missing frames.
    """
    deltas: List[TemporalProbabilityDelta] = []
    if len(valid_frames) < 2:
        return deltas

    for i in range(len(valid_frames) - 1):
        f_cur = valid_frames[i]
        f_next = valid_frames[i + 1]

        p_cur = f_cur.xception_probability if f_cur.xception_probability is not None else 0.0
        p_next = f_next.xception_probability if f_next.xception_probability is not None else 0.0

        step = round(p_next - p_cur, 4)
        abs_step = round(abs(p_next - p_cur), 4)

        deltas.append(
            TemporalProbabilityDelta(
                from_frame=f_cur.frame_number,
                to_frame=f_next.frame_number,
                from_timestamp=f_cur.timestamp,
                to_timestamp=f_next.timestamp,
                delta=step,
                absolute_delta=abs_step
            )
        )

    return deltas


def compute_probability_volatility(
    valid_frames: List[TemporalFrameItem],
    deltas: List[TemporalProbabilityDelta]
) -> Optional[TemporalProbabilityVolatility]:
    """
    Computes statistical dispersion and volatility metrics over valid frame sequence:
    mean, median, min, max, standard deviation, mean absolute delta, max absolute delta.
    """
    if len(valid_frames) < 2:
        return None

    probs = [
        f.xception_probability for f in valid_frames
        if f.xception_probability is not None
    ]
    if not probs:
        return None

    mean_prob = round(float(np.mean(probs)), 4)
    median_prob = round(float(np.median(probs)), 4)
    min_prob = round(float(np.min(probs)), 4)
    max_prob = round(float(np.max(probs)), 4)
    std_prob = round(float(np.std(probs, ddof=1)), 4) if len(probs) >= 2 else 0.0

    abs_deltas = [d.absolute_delta for d in deltas]
    mean_abs_delta = round(float(np.mean(abs_deltas)), 4) if abs_deltas else 0.0
    max_abs_delta = round(float(np.max(abs_deltas)), 4) if abs_deltas else 0.0

    return TemporalProbabilityVolatility(
        mean_probability=mean_prob,
        median_probability=median_prob,
        min_probability=min_prob,
        max_probability=max_prob,
        std_probability=std_prob,
        mean_absolute_delta=mean_abs_delta,
        max_absolute_delta=max_abs_delta
    )


def detect_prediction_transitions(
    valid_frames: List[TemporalFrameItem]
) -> List[TemporalPredictionTransition]:
    """
    Detects categorical changes between consecutive valid frames:
    e.g. authentic -> manipulated or manipulated -> authentic.
    """
    transitions: List[TemporalPredictionTransition] = []
    if len(valid_frames) < 2:
        return transitions

    for i in range(len(valid_frames) - 1):
        f_cur = valid_frames[i]
        f_next = valid_frames[i + 1]

        pred_cur = (f_cur.xception_prediction or "").lower()
        pred_next = (f_next.xception_prediction or "").lower()

        if pred_cur and pred_next and pred_cur != pred_next:
            transitions.append(
                TemporalPredictionTransition(
                    frame_before=f_cur.frame_number,
                    frame_after=f_next.frame_number,
                    timestamp=f_next.timestamp,
                    from_prediction=pred_cur,
                    to_prediction=pred_next,
                    from_probability=f_cur.xception_probability or 0.0,
                    to_probability=f_next.xception_probability or 0.0,
                    description=f"Prediction transition detected: '{pred_cur}' -> '{pred_next}'."
                )
            )

    return transitions


def analyze_face_presence(
    all_frames: List[TemporalFrameItem]
) -> TemporalFacePresence:
    """
    Measures facial presence continuity across all sampled frames.
    Computes frames with/without face, presence percentage, longest present & missing streaks.
    """
    total = len(all_frames)
    if total == 0:
        return TemporalFacePresence(
            total_sampled_frames=0,
            frames_with_face=0,
            frames_without_face=0,
            face_presence_percentage=0.0,
            longest_consecutive_face_present=0,
            longest_consecutive_face_missing=0
        )

    with_face = sum(1 for f in all_frames if f.face_count > 0)
    without_face = total - with_face
    presence_pct = round((with_face / total) * 100, 1)

    max_present = 0
    cur_present = 0
    max_missing = 0
    cur_missing = 0

    for f in all_frames:
        if f.face_count > 0:
            cur_present += 1
            if cur_present > max_present:
                max_present = cur_present
            cur_missing = 0
        else:
            cur_missing += 1
            if cur_missing > max_missing:
                max_missing = cur_missing
            cur_present = 0

    return TemporalFacePresence(
        total_sampled_frames=total,
        frames_with_face=with_face,
        frames_without_face=without_face,
        face_presence_percentage=presence_pct,
        longest_consecutive_face_present=max_present,
        longest_consecutive_face_missing=max_missing
    )


def track_face_bounding_boxes(
    valid_frames: List[TemporalFrameItem],
    frame_width: int,
    frame_height: int
) -> Optional[TemporalBBoxTracking]:
    """
    Tracks normalized spatial center displacement and bounding box area changes
    between consecutive valid frames.
    """
    if len(valid_frames) < 2 or frame_width <= 0 or frame_height <= 0:
        return None

    frame_area = frame_width * frame_height
    displacements: List[float] = []
    face_areas: List[float] = []
    area_deltas: List[float] = []

    for f in valid_frames:
        if f.face_bbox and len(f.face_bbox) >= 4:
            w, h = f.face_bbox[2], f.face_bbox[3]
            norm_area = (w * h) / frame_area
            face_areas.append(norm_area)

    for i in range(len(valid_frames) - 1):
        b_cur = valid_frames[i].face_bbox
        b_next = valid_frames[i + 1].face_bbox

        if b_cur and b_next and len(b_cur) >= 4 and len(b_next) >= 4:
            # Normalized center coordinates
            cx_cur = (b_cur[0] + b_cur[2] / 2.0) / frame_width
            cy_cur = (b_cur[1] + b_cur[3] / 2.0) / frame_height

            cx_next = (b_next[0] + b_next[2] / 2.0) / frame_width
            cy_next = (b_next[1] + b_next[3] / 2.0) / frame_height

            disp = math.sqrt((cx_next - cx_cur) ** 2 + (cy_next - cy_cur) ** 2)
            displacements.append(disp)

            area_cur = (b_cur[2] * b_cur[3]) / frame_area
            area_next = (b_next[2] * b_next[3]) / frame_area
            area_deltas.append(abs(area_next - area_cur))

    if not face_areas:
        return None

    mean_disp = round(float(np.mean(displacements)), 4) if displacements else 0.0
    max_disp = round(float(np.max(displacements)), 4) if displacements else 0.0
    mean_area = round(float(np.mean(face_areas)), 4)
    min_area = round(float(np.min(face_areas)), 4)
    max_area = round(float(np.max(face_areas)), 4)
    max_area_delta = round(float(np.max(area_deltas)), 4) if area_deltas else 0.0

    return TemporalBBoxTracking(
        mean_center_displacement=mean_disp,
        max_center_displacement=max_disp,
        mean_face_area=mean_area,
        min_face_area=min_area,
        max_face_area=max_area,
        max_area_delta=max_area_delta
    )


def compute_model_agreement_over_time(
    valid_frames: List[TemporalFrameItem]
) -> Optional[TemporalModelAgreement]:
    """
    Compares primary Xception and secondary Meso4 predictions over all shared valid frames.
    """
    paired = [
        f for f in valid_frames
        if f.xception_prediction is not None
        and f.meso4_prediction is not None
        and f.xception_probability is not None
        and f.meso4_probability is not None
    ]

    total = len(paired)
    if total == 0:
        return None

    agreeing = sum(1 for f in paired if f.xception_prediction.lower() == f.meso4_prediction.lower())
    disagreeing = total - agreeing
    rate = round(agreeing / total, 4)

    # Calculate Pearson correlation if possible
    x_probs = [f.xception_probability for f in paired]
    m_probs = [f.meso4_probability for f in paired]

    pearson_r = None
    if total >= 2:
        try:
            if np.std(x_probs) > 1e-6 and np.std(m_probs) > 1e-6:
                corr = float(np.corrcoef(x_probs, m_probs)[0, 1])
                if not math.isnan(corr):
                    pearson_r = round(corr, 4)
        except Exception:
            pass

    # Calculate simple Cohen's kappa for two raters
    cohen_k = None
    if total >= 2:
        try:
            p_o = agreeing / total
            # Marginal frequencies
            x_manip = sum(1 for f in paired if f.xception_prediction.lower() == "manipulated") / total
            x_auth = 1.0 - x_manip
            m_manip = sum(1 for f in paired if f.meso4_prediction.lower() == "manipulated") / total
            m_auth = 1.0 - m_manip
            p_e = (x_manip * m_manip) + (x_auth * m_auth)
            if abs(1.0 - p_e) > 1e-6:
                k = (p_o - p_e) / (1.0 - p_e)
                cohen_k = round(float(k), 4)
            else:
                cohen_k = 1.0 if p_o == 1.0 else 0.0
        except Exception:
            pass

    return TemporalModelAgreement(
        frames_compared=total,
        agreeing_frames=agreeing,
        disagreeing_frames=disagreeing,
        agreement_rate=rate,
        cohen_kappa=cohen_k,
        pearson_correlation=pearson_r
    )


def extract_temporal_events(
    all_frames: List[TemporalFrameItem],
    valid_frames: List[TemporalFrameItem],
    deltas: List[TemporalProbabilityDelta],
    transitions: List[TemporalPredictionTransition],
    frame_width: int = 1920,
    frame_height: int = 1080,
    jump_threshold: float = PROBABILITY_JUMP_THRESHOLD,
    area_change_threshold: float = FACE_AREA_CHANGE_THRESHOLD,
    displacement_threshold: float = CENTER_DISPLACEMENT_THRESHOLD
) -> List[TemporalEvent]:
    """
    Constructs discrete temporal events for measurable anomalies and transitions.
    Allowed types:
    - probability_jump
    - prediction_transition
    - face_presence_gap
    - face_bbox_change
    - face_scale_change
    - detector_disagreement
    """
    events: List[TemporalEvent] = []

    # 1. Probability Jumps
    for d in deltas:
        if d.absolute_delta >= jump_threshold:
            severity = "notable" if d.absolute_delta >= 0.40 else "informational"
            events.append(
                TemporalEvent(
                    event_id=f"evt_prob_jump_f{d.to_frame}",
                    frame_number=d.to_frame,
                    timestamp=d.to_timestamp,
                    event_type="probability_jump",
                    value=d.absolute_delta,
                    threshold=jump_threshold,
                    severity=severity,
                    description=(
                        f"Significant probability step ({d.absolute_delta:.2f}) observed "
                        f"between frame {d.from_frame} and {d.to_frame}."
                    )
                )
            )

    # 2. Prediction Transitions
    for t in transitions:
        events.append(
            TemporalEvent(
                event_id=f"evt_pred_flip_f{t.frame_after}",
                frame_number=t.frame_after,
                timestamp=t.timestamp,
                event_type="prediction_transition",
                value=round(abs(t.to_probability - t.from_probability), 4),
                threshold=0.50,
                severity="notable",
                description=(
                    f"Prediction transition detected: categorical classification flipped from "
                    f"'{t.from_prediction}' ({t.from_probability:.2f}) to '{t.to_prediction}' ({t.to_probability:.2f})."
                )
            )
        )

    # 3. Face Presence Gaps
    gap_frames = 0
    gap_start_idx = 0
    for idx, f in enumerate(all_frames):
        if f.face_count == 0:
            if gap_frames == 0:
                gap_start_idx = idx
            gap_frames += 1
        else:
            if gap_frames >= 2 and gap_start_idx > 0:
                g_frame = all_frames[gap_start_idx]
                events.append(
                    TemporalEvent(
                        event_id=f"evt_face_gap_f{g_frame.frame_number}",
                        frame_number=g_frame.frame_number,
                        timestamp=g_frame.timestamp,
                        event_type="face_presence_gap",
                        value=float(gap_frames),
                        threshold=2.0,
                        severity="informational",
                        description=(
                            f"Face presence gap: no facial structure detected across {gap_frames} consecutive frames "
                            f"(starting at frame {g_frame.frame_number})."
                        )
                    )
                )
            gap_frames = 0

    # Final check if video ends on a face gap
    if gap_frames >= 2 and gap_start_idx > 0:
        g_frame = all_frames[gap_start_idx]
        events.append(
            TemporalEvent(
                event_id=f"evt_face_gap_f{g_frame.frame_number}",
                frame_number=g_frame.frame_number,
                timestamp=g_frame.timestamp,
                event_type="face_presence_gap",
                value=float(gap_frames),
                threshold=2.0,
                severity="informational",
                description=f"Face presence gap: no facial structure detected across {gap_frames} frames at video end."
            )
        )

    # 4. Face Bounding Box Displacements & Scale Changes
    if frame_width > 0 and frame_height > 0:
        frame_area = frame_width * frame_height
        for i in range(len(valid_frames) - 1):
            f1 = valid_frames[i]
            f2 = valid_frames[i + 1]

            if f1.face_bbox and f2.face_bbox:
                cx1 = (f1.face_bbox[0] + f1.face_bbox[2] / 2.0) / frame_width
                cy1 = (f1.face_bbox[1] + f1.face_bbox[3] / 2.0) / frame_height
                cx2 = (f2.face_bbox[0] + f2.face_bbox[2] / 2.0) / frame_width
                cy2 = (f2.face_bbox[1] + f2.face_bbox[3] / 2.0) / frame_height

                disp = round(float(math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)), 4)
                if disp >= displacement_threshold:
                    severity = "notable" if disp >= 0.35 else "informational"
                    events.append(
                        TemporalEvent(
                            event_id=f"evt_bbox_disp_f{f2.frame_number}",
                            frame_number=f2.frame_number,
                            timestamp=f2.timestamp,
                            event_type="face_bbox_change",
                            value=disp,
                            threshold=displacement_threshold,
                            severity=severity,
                            description=(
                                f"Rapid spatial displacement of detected face bounding box ({disp:.2f} normalized) "
                                f"between frame {f1.frame_number} and {f2.frame_number}."
                            )
                        )
                    )

                area1 = (f1.face_bbox[2] * f1.face_bbox[3]) / frame_area
                area2 = (f2.face_bbox[2] * f2.face_bbox[3]) / frame_area
                area_step = round(float(abs(area2 - area1)), 4)
                if area_step >= area_change_threshold:
                    events.append(
                        TemporalEvent(
                            event_id=f"evt_scale_f{f2.frame_number}",
                            frame_number=f2.frame_number,
                            timestamp=f2.timestamp,
                            event_type="face_scale_change",
                            value=area_step,
                            threshold=area_change_threshold,
                            severity="informational",
                            description=(
                                f"Substantial facial area scale step ({area_step:.2f}) "
                                f"observed between frame {f1.frame_number} and {f2.frame_number}."
                            )
                        )
                    )

    # 5. Detector Disagreements Over Time
    for f in valid_frames:
        if (
            f.xception_prediction is not None
            and f.meso4_prediction is not None
            and f.xception_prediction.lower() != f.meso4_prediction.lower()
        ):
            x_p = f.xception_probability or 0.0
            m_p = f.meso4_probability or 0.0
            events.append(
                TemporalEvent(
                    event_id=f"evt_disagree_f{f.frame_number}",
                    frame_number=f.frame_number,
                    timestamp=f.timestamp,
                    event_type="detector_disagreement",
                    value=round(abs(x_p - m_p), 4),
                    threshold=0.50,
                    severity="notable",
                    description=(
                        f"Detector disagreement detected: Xception classified as '{f.xception_prediction}' ({x_p:.2f}) "
                        f"while Meso4 classified as '{f.meso4_prediction}' ({m_p:.2f})."
                    )
                )
            )

    # Sort events chronologically by frame number
    events.sort(key=lambda e: (e.frame_number, e.event_type))
    return events


class TemporalAnalysisService:
    """Service performing advanced temporal consistency forensics on video sequences."""

    def analyze_temporal_consistency(
        self,
        frames: List[TemporalFrameItem],
        frame_width: int = 1920,
        frame_height: int = 1080,
        jump_threshold: float = PROBABILITY_JUMP_THRESHOLD,
        area_change_threshold: float = FACE_AREA_CHANGE_THRESHOLD,
        displacement_threshold: float = CENTER_DISPLACEMENT_THRESHOLD
    ) -> TemporalAnalysisResult:
        """
        Executes end-to-end temporal consistency analysis over the given sequence of sampled frames.
        """
        total_sampled = len(frames)
        face_presence = analyze_face_presence(frames)
        valid_frames = [f for f in frames if f.valid and f.face_count > 0]
        valid_count = len(valid_frames)

        # Handle edge cases gracefully
        if total_sampled == 0:
            return TemporalAnalysisResult(
                status="not_available",
                reason="No video frames sampled for temporal consistency analysis.",
                analyzed_frames=0,
                valid_frames=0,
                face_presence=face_presence
            )

        if valid_count == 0:
            return TemporalAnalysisResult(
                status="not_available",
                reason="No facial structures detected across sampled video frames.",
                analyzed_frames=total_sampled,
                valid_frames=0,
                face_presence=face_presence,
                frame_series=frames
            )

        if valid_count < 2:
            return TemporalAnalysisResult(
                status="not_available",
                reason=f"Insufficient valid face frames ({valid_count} found, minimum 2 required for temporal analysis).",
                analyzed_frames=total_sampled,
                valid_frames=valid_count,
                face_presence=face_presence,
                frame_series=frames
            )

        # 1. Deltas
        deltas = compute_probability_deltas(valid_frames)

        # 2. Volatility
        volatility = compute_probability_volatility(valid_frames, deltas)

        # 3. Transitions
        transitions = detect_prediction_transitions(valid_frames)

        # 4. Bounding Box & Scale Tracking
        bbox_tracking = track_face_bounding_boxes(valid_frames, frame_width, frame_height)

        # 5. Model Disagreement
        model_agreement = compute_model_agreement_over_time(valid_frames)

        # 6. Discrete Temporal Events
        events = extract_temporal_events(
            all_frames=frames,
            valid_frames=valid_frames,
            deltas=deltas,
            transitions=transitions,
            frame_width=frame_width,
            frame_height=frame_height,
            jump_threshold=jump_threshold,
            area_change_threshold=area_change_threshold,
            displacement_threshold=displacement_threshold
        )

        return TemporalAnalysisResult(
            status="available",
            reason=None,
            analyzed_frames=total_sampled,
            valid_frames=valid_count,
            face_presence=face_presence,
            statistics=volatility,
            deltas=deltas,
            prediction_transitions=transitions,
            bbox_tracking=bbox_tracking,
            temporal_events=events,
            model_agreement=model_agreement,
            frame_series=frames
        )


# Singleton instance
_temporal_service: Optional[TemporalAnalysisService] = None

def get_temporal_analysis_service() -> TemporalAnalysisService:
    global _temporal_service
    if _temporal_service is None:
        _temporal_service = TemporalAnalysisService()
    return _temporal_service
