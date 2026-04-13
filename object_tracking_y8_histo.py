"""
sentinel/object_tracking_y8_histo.py
======================================
Appearance-based person detection and tracking using YOLOv8s-VisDrone
with HSV color histogram identity matching.

Why this replaces the ByteTrack approach
-----------------------------------------
ByteTrack was designed for static or slowly-moving surveillance cameras.
Its Kalman filter predicts each object's next position by assuming the
camera is roughly stationary.  When the drone yaws to center on a target,
every person in the frame shifts simultaneously — ByteTrack's filter
predicted each one stayed in place, so all tracks fail at once.  This is
not a tuning problem; it is a fundamental design mismatch.

The previous attempts to work around this (match_thresh, IoU fallback,
center-distance fallback) were compensations layered on top of a tool that
cannot work correctly under camera ego-motion.

Architecture
------------
Two clean phases:

  SEARCH — run YOLO detection every `detect_every_n` frames.  The
  highest-confidence detection above `detect_confidence` becomes the
  candidate target.  The bbox crop is converted to HSV and a color
  histogram is computed and stored as the target's identity signature.
  Tracking begins immediately on the same frame.

  TRACK — run YOLO detection every frame (no stride; fresh detections
  required for identity comparison).  For every detected person bbox,
  extract a color histogram of the crop and compare it to the stored
  target signature using histogram correlation.  The detection with the
  best correlation score above HISTO_MATCH_THRESHOLD is accepted as the
  current target position.  If no detection scores above threshold,
  increment the miss counter.  After TRACKER_MISSES_MAX consecutive
  misses, declare the target lost and return to search.

Why color histograms
--------------------
* Camera-motion invariant.  A pan that shifts every bbox by 30 px does
  not change the person's clothing color.  Position-based tracking
  (ByteTrack, CSRT, Kalman) fails under ego-motion.  Histogram matching
  does not.
* Fast.  Histogram extraction on a small crop is microseconds.  The
  bottleneck remains YOLO inference, not identity comparison.
* No drift.  Every frame is an independent detection + identity check.
  There is no accumulated error from a prediction-correction loop.
* Natural re-acquisition.  If the person temporarily leaves the frame
  and returns, the histogram match finds them without any special logic.
* Simple to reason about.  No tuning of Kalman noise covariances,
  match thresholds, or track buffer lengths for ByteTrack.

Limitation
----------
If two people in the scene wear very similar colors, the identity match
may swap to the wrong person.  In typical outdoor drone scenarios with one
subject and incidental bystanders, this is unlikely.  The upgrade path is
to replace the histogram with a re-identification embedding network, but
that adds a second model and the histogram approach should be validated
first.

Public API — identical external surface to object_tracking_y8.py
----------------------------------------------------------------
  load_model(engine_path, imgsz, conf, **kwargs)
      Load the YOLO engine.  **kwargs silently absorbs ByteTrack-specific
      parameters (track_imgsz, match_thresh, track_buffer) so mission.py's
      existing load_model() call works without modification.

  set_object_to_track(frame, bbox, **kwargs)
      Lock onto the person in bbox.  Computes the initial histogram
      signature from the crop.  **kwargs absorbs legacy parameters.

  track_with_confirm(img, img_write, show_img, **kwargs)
      Run one tracking frame.  Returns the standard 6-tuple.

  run_perception_step(frame, display, tracker_active, frame_counter,
                      detect_every_n, detect_confidence,
                      tracker_misses_max, reset_fn)
      The single entry point used by mission.py and run_bench.py.
      Return signature is unchanged:
          (tracker_active, target_bbox, target_center,
           perception_state, message)

Module-level globals read / written by mission.py
--------------------------------------------------
  TRACKER_MISSES_MAX     Set at startup from config.tracker_misses_max.
  total_track_misses     Read each frame for telemetry.
  _target_track_id       Set to None by mission._reset_tracking_state()
                         to force a clean tracker reset.  Non-None while
                         tracking is active.  (Repurposed as a boolean
                         sentinel; the concept of a numeric ByteTrack ID
                         does not exist in this module.)

Tunable module-level constants
-------------------------------
  HISTO_MATCH_THRESHOLD  Minimum histogram correlation to accept a
                         detection as the tracked person.  Range 0-1.
                         Default 0.70.  Raise to reduce false positives
                         in crowded scenes; lower if the target is
                         frequently missed due to lighting changes.

  HISTO_UPDATE_ALPHA     Blend rate for the adaptive signature update.
                         Each confirmed-match frame blends the new crop's
                         histogram into the stored signature at this rate.
                         Default 0.05 (5% new, 95% existing).  Higher
                         values adapt faster to changing lighting but
                         increase the risk of signature drift.

  HISTO_UPDATE_MIN_SCORE Minimum correlation required before the
                         signature is updated.  Prevents updating the
                         signature on a marginal match.  Default 0.85.

  H_BINS / S_BINS        Histogram resolution for the H and S channels.
                         Defaults: 36 / 32 (total signature = 68 floats).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import cam_handler

LOG = logging.getLogger("sentinel_drone.object_tracking_histo")

# ── Model ──────────────────────────────────────────────────────────────────────
_model: Optional[YOLO] = None
_model_imgsz: int = 640
_model_conf: float = 0.25

# VisDrone class indices that count as a valid target.
# 0 = pedestrian (individual), 1 = people (group / partially visible)
_PERSON_CLASSES: list[int] = [0, 1]

# ── Histogram parameters ───────────────────────────────────────────────────────
# H channel: OpenCV hue range is 0-179 (half of 360 degrees).
# 36 bins = one bin per 5 degrees.  Enough resolution to distinguish clothing
# colors without being brittle to lighting-induced hue shifts.
H_BINS: int = 36

# S channel: 0-255.
# 32 bins gives coarse saturation resolution.  We care whether the person is
# wearing a saturated color vs. neutral (white/grey/black), not fine gradations.
S_BINS: int = 32

# Minimum histogram correlation (0.0-1.0) for a detection to be accepted
# as the tracked target.  0.70 is intentionally somewhat permissive — it
# handles lighting variation and partial occlusion without being so strict
# that it drops the target when the person turns or moves into shadow.
HISTO_MATCH_THRESHOLD: float = 0.70

# Signature blend rate per confirmed-match frame.
# 0.05 = the stored signature shifts 5% toward the new crop each frame.
# At 10 FPS this means the signature fully updates over ~200 frames (~20 s),
# which is slow enough to be stable but fast enough to track gradual
# changes in lighting across a full mission.
HISTO_UPDATE_ALPHA: float = 0.05

# Only update the stored signature when the match score is this high or above.
# Prevents a marginal match (barely above HISTO_MATCH_THRESHOLD) from slowly
# corrupting the signature toward a neighboring detection.
HISTO_UPDATE_MIN_SCORE: float = 0.85

# Pixel margin to shrink the crop inward on each side before computing the
# histogram.  Removes background contamination from the bbox edges where
# YOLO detections are often slightly loose.
CROP_SHRINK_PX: int = 4

# ── Tracking state ─────────────────────────────────────────────────────────────
# The stored HSV histogram signature of the locked target.  None while not
# tracking.  Shape: (H_BINS + S_BINS,) float32.
_target_signature: Optional[np.ndarray] = None

# Last confirmed detection bbox and center.  Updated each time the histogram
# match succeeds.  Used only for display annotations when in miss-hold state.
_last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
_last_center: Tuple[int, int] = (0, 0)

# Consecutive frames on which no detection scored above HISTO_MATCH_THRESHOLD.
# Reset to 0 on every successful match.  Triggers lost declaration at
# TRACKER_MISSES_MAX.
total_track_misses: int = 0

# True when the most recent tracking frame produced a successful match.
confirmed_object_tracking: bool = False

# Maximum consecutive misses before declaring the target lost.
# Overridden by mission.py at startup:
#   obj_track.TRACKER_MISSES_MAX = config.tracker_misses_max
TRACKER_MISSES_MAX: int = 20

# Boolean sentinel for tracking state.  None = not tracking.  Non-None (1)
# = tracking active.  mission.py sets this to None to force a reset, so the
# type and None-check semantics must be preserved.
_target_track_id: Optional[int] = None

# Legacy compat: the old ByteTrack module exposed _confirm_frame_counter.
_confirm_frame_counter: int = 0

# ── Camera compatibility state ─────────────────────────────────────────────────
pipeline = None
config = None

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FRAME_HORIZONTAL_CENTER: int = 320
FRAME_VERTICAL_CENTER: int = 240

streaming_distance: bool = False
latest_accel_data = None
latest_gyro_data = None
last_frame_timestamp: float = 0.0


# ── Histogram helpers ──────────────────────────────────────────────────────────

def _safe_crop(frame: np.ndarray,
               bbox: Tuple[int, int, int, int],
               shrink: int = CROP_SHRINK_PX) -> Optional[np.ndarray]:
    """
    Return a BGR crop for the given (x, y, w, h) bbox, shrunk inward by
    `shrink` pixels on each side to reduce background contamination.

    Returns None if the resulting crop would be degenerate (zero or negative
    dimension) — which can happen for very small detections at long range.
    """
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]

    x1 = max(0, x + shrink)
    y1 = max(0, y + shrink)
    x2 = min(fw, x + w - shrink)
    y2 = min(fh, y + h - shrink)

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _compute_histogram(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute a normalized combined H+S histogram from a BGR crop.

    Converts the crop to HSV, builds separate histograms for the H and S
    channels, normalizes each independently, then concatenates them into a
    single 1D float32 signature vector of length H_BINS + S_BINS.

    Using H and S (and ignoring V/brightness) makes the signature resistant
    to lighting intensity changes — the same blue jacket looks the same
    histogram whether the sun is directly overhead or partially occluded.

    Returns None if the crop is empty or conversion fails.
    """
    if crop is None or crop.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return None

    # H channel: 0-179 in OpenCV's HSV encoding.
    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180])
    # S channel: 0-255.
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256])

    # Normalize each channel histogram independently to unit sum.
    # This makes the signature independent of bbox area (number of pixels).
    h_sum = float(h_hist.sum())
    s_sum = float(s_hist.sum())
    if h_sum > 0:
        h_hist = h_hist / h_sum
    if s_sum > 0:
        s_hist = s_hist / s_sum

    return np.concatenate([h_hist.flatten(), s_hist.flatten()]).astype(np.float32)


def _histogram_correlation(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Return the cosine similarity between two histogram signature vectors.

    Range: 0.0 (no similarity) to 1.0 (identical).  Negative values
    (anti-correlated histograms) are clamped to 0.0.

    Cosine similarity is used rather than cv2.HISTCMP_CORREL (Pearson)
    because our histograms are already normalized to unit sum, so the mean
    subtraction in Pearson correlation adds no benefit and introduces
    numerical noise for near-uniform histograms.
    """
    norm_a = float(np.linalg.norm(sig_a))
    norm_b = float(np.linalg.norm(sig_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    score = float(np.dot(sig_a, sig_b)) / (norm_a * norm_b)
    return max(0.0, score)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(engine_path: str = "yolov8s_visdrone.engine",
               imgsz: int = 640,
               conf: float = 0.25,
               **kwargs) -> None:
    """
    Load the YOLOv8 TensorRT engine for person detection.

    This module uses the model for detection only — model() not model.track().
    ByteTrack is not used.  **kwargs silently absorbs the ByteTrack-specific
    parameters that mission.py still passes (track_imgsz, match_thresh,
    track_buffer) so the existing call site does not need to change.

    Parameters
    ----------
    engine_path : Path to the TensorRT .engine file or a .pt weights file.
    imgsz       : Inference resolution.  Must match the export resolution
                  if using a static TRT engine.
    conf        : Minimum YOLO detection confidence.
    **kwargs    : Accepted and silently ignored (track_imgsz, match_thresh,
                  track_buffer from the old ByteTrack configuration).
    """
    global _model, _model_imgsz, _model_conf

    ignored = {k: v for k, v in kwargs.items()
               if k in ('track_imgsz', 'match_thresh', 'track_buffer')}
    if ignored:
        LOG.info(
            "load_model: ByteTrack parameters ignored by histogram tracker: %s",
            list(ignored.keys()),
        )

    _model = YOLO(engine_path, task='detect')
    _model_imgsz = imgsz
    _model_conf = conf

    LOG.info(
        "Histogram tracker: loaded model %s  imgsz=%d  conf=%.2f  "
        "match_threshold=%.2f  update_alpha=%.2f",
        engine_path, imgsz, conf,
        HISTO_MATCH_THRESHOLD, HISTO_UPDATE_ALPHA,
    )


# ── Initial detection (non-committing scan) ────────────────────────────────────

def check_for_initial_target(
        img: Optional[np.ndarray] = None,
        img_write: Optional[np.ndarray] = None,
        show_img: bool = False,
        in_debug: bool = False,
) -> Tuple:
    """
    Non-committing single-frame person detection pass.

    Called while the drone is still in motion on its AUTO waypoint mission.
    This function only *detects* — it does not lock onto a target or build
    a histogram signature.  The caller (DroneMission.conduct_mission) uses
    the returned confidence and bbox to decide whether to record the GPS
    sighting location and later fly back for confirmation.

    The drone may be well past the subject's position by the time this
    returns a positive result.  pex03.py stores the GPS coordinates of this
    initial sighting and commands the drone back to that location before
    calling set_object_to_track() to commit to the target.

    Parameters
    ----------
    img       : BGR frame to run detection on.  If None, get_cur_frame() is
                called to obtain the latest camera frame.
    img_write : Annotation frame modified in-place.  Defaults to a copy of img.
    show_img  : Display the annotated frame in a cv2 window if True.
    in_debug  : When True, detect cars instead of pedestrians/people.
                Mirrors the behaviour of the old YOLOv4 implementation so
                that ground-truth testing with vehicle targets still works.

    Returns
    -------
    (center, confidence, (x, y), radius, img_write, bbox)
        Identical 6-tuple to the old object_tracking.check_for_initial_target().
        center     : (cx, cy) pixel centre of the best detection.
        confidence : YOLO confidence score (float), or None if nothing found.
        (x, y)     : Top-left corner of the detection bbox.
        radius     : Half the larger bbox dimension in pixels.
        img_write  : Annotated frame.
        bbox       : (x, y, w, h) bounding box.
        All position fields are (0, 0) / (0,0,0,0) when nothing is found.
    """
    if _model is None:
        LOG.warning("check_for_initial_target: model not loaded.")
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)

    if img is None:
        img = cam_handler.get_cur_frame()
    if img is None:
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)

    if img_write is None:
        img_write = img.copy()

    # In debug mode, use car class indices (VisDrone class 4 = car).
    # In normal mode, use pedestrian / people classes.
    target_classes = [4] if in_debug else _PERSON_CLASSES

    results = _model(
        img,
        classes=target_classes,
        conf=_model_conf,
        imgsz=_model_imgsz,
        half=True,
        device=0,
        max_det=20,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        cv2.putText(img_write, 'Searching...', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        if show_img:
            cv2.imshow("Real-time Detect", img_write)
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)

    # Pick the detection with the highest confidence score.
    best_idx = max(range(len(boxes)), key=lambda k: float(boxes[k].conf))
    best = boxes[best_idx]
    conf = float(best.conf)

    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
    w = int(x2 - x1)
    h = int(y2 - y1)
    bbox = (int(x1), int(y1), w, h)
    cx = int(x1 + w / 2)
    cy = int(y1 + h / 2)
    radius = float(max(w, h) / 2)

    # Annotate the display frame so the operator can see what triggered the
    # initial sighting — but do NOT modify any tracking state here.
    cv2.rectangle(img_write, (x1, y1), (x2, y2), (20, 20, 230), 2)
    cv2.putText(
        img_write,
        f"candidate: {conf:.2f}",
        (x1 + 5, y1 + 20),
        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 20, 230), 2,
    )

    if show_img:
        cv2.imshow("Real-time Detect", img_write)

    LOG.debug(
        "check_for_initial_target: candidate at bbox=%s  conf=%.2f", bbox, conf
    )

    return (cx, cy), conf, (int(x1), int(y1)), radius, img_write, bbox


# ── Target registration ────────────────────────────────────────────────────────

def set_object_to_track(frame: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        **kwargs) -> None:
    """
    Lock onto the person inside ``bbox`` as the current tracking target.

    Extracts a color histogram from the bbox crop and stores it as the
    target's identity signature.  All subsequent track_with_confirm() calls
    match detected persons against this signature.

    Parameters
    ----------
    frame   : Current BGR frame from the camera.
    bbox    : (x, y, w, h) bounding box of the person to lock onto.
    **kwargs: Accepted for API compatibility (bbox_margin, precomputed_results
              from the old ByteTrack module); silently ignored.
    """
    global _target_signature, _target_track_id, _last_bbox, _last_center
    global total_track_misses, confirmed_object_tracking, _confirm_frame_counter

    crop = _safe_crop(frame, bbox)
    if crop is None:
        LOG.warning(
            "set_object_to_track: bbox crop is degenerate (%s) — "
            "cannot build signature.",
            bbox,
        )
        _target_track_id = None
        return

    sig = _compute_histogram(crop)
    if sig is None:
        LOG.warning(
            "set_object_to_track: histogram computation failed for bbox %s.",
            bbox,
        )
        _target_track_id = None
        return

    _target_signature = sig
    x, y, w, h = bbox
    _last_bbox = bbox
    _last_center = (x + w // 2, y + h // 2)
    _target_track_id = 1          # Non-None sentinel: tracking is active.
    total_track_misses = 0
    confirmed_object_tracking = True
    _confirm_frame_counter = 0

    LOG.debug(
        "Target signature stored: bbox=%s  hist_norm=%.4f",
        bbox, float(np.linalg.norm(sig)),
    )


# ── Main tracking function ─────────────────────────────────────────────────────

def track_with_confirm(
        img: np.ndarray,
        img_write: Optional[np.ndarray] = None,
        show_img: bool = False,
        **kwargs,
) -> Tuple:
    """
    Run one tracking frame using histogram identity matching.

    Runs YOLO detection on the full frame, computes color histograms for
    every detected person, and returns the detection whose histogram best
    matches the stored target signature.

    Ego-motion (camera pan/tilt) does not affect the result because histogram
    matching is purely appearance-based — the person's clothing color is the
    same regardless of where they appear in the frame.

    Parameters
    ----------
    img       : Current BGR frame (uint8).
    img_write : Annotation frame modified in-place.  A copy of img is used
                if None.
    show_img  : Display the annotated frame in a cv2 window if True.
    **kwargs  : Absorbs legacy parameters (confirm_every_n, detect_fn, etc.)

    Returns
    -------
    (center, confidence, (x, y), radius, frame_display, bbox)
        center        : (cx, cy) pixel centre of the matched bbox.
        confidence    : Match score (0.0-1.0) when a match is found;
                        None when no detection scores above threshold.
        (x, y)        : Top-left corner of the matched bbox.
        radius        : Half the larger bbox dimension in pixels.
        frame_display : Annotated img_write.
        bbox          : (x, y, w, h) of the matched bbox.
    """
    global total_track_misses, confirmed_object_tracking
    global _last_bbox, _last_center, _target_track_id, _target_signature

    if img_write is None:
        img_write = img.copy()

    # Not tracking — nothing to do.
    if _model is None or _target_track_id is None or _target_signature is None:
        return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)

    # ── Run YOLO detection (no ByteTrack) ──────────────────────────────────
    results = _model(
        img,
        classes=_PERSON_CLASSES,
        conf=_model_conf,
        imgsz=_model_imgsz,
        half=True,
        device=0,
        max_det=20,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        # No people detected at all this frame.
        return _handle_miss(img_write, show_img)

    # ── Compare each detection to the target signature ─────────────────────
    best_score = -1.0
    best_bbox: Optional[Tuple[int, int, int, int]] = None

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        w = x2 - x1
        h = y2 - y1
        candidate_bbox = (int(x1), int(y1), int(w), int(h))

        crop = _safe_crop(img, candidate_bbox)
        if crop is None:
            continue

        sig = _compute_histogram(crop)
        if sig is None:
            continue

        score = _histogram_correlation(_target_signature, sig)

        if score > best_score:
            best_score = score
            best_bbox = candidate_bbox

    # ── No match above threshold ───────────────────────────────────────────
    if best_bbox is None or best_score < HISTO_MATCH_THRESHOLD:
        return _handle_miss(img_write, show_img)

    # ── Match found ────────────────────────────────────────────────────────
    bx, by, bw, bh = best_bbox
    cx = bx + bw // 2
    cy = by + bh // 2

    _last_bbox = best_bbox
    _last_center = (cx, cy)
    total_track_misses = 0
    confirmed_object_tracking = True

    # Adaptive signature update: only blend when the match is very confident.
    # This prevents marginal matches from drifting the signature over time.
    if best_score >= HISTO_UPDATE_MIN_SCORE:
        fresh_crop = _safe_crop(img, best_bbox)
        if fresh_crop is not None:
            fresh_sig = _compute_histogram(fresh_crop)
            if fresh_sig is not None:
                _target_signature = (
                    (1.0 - HISTO_UPDATE_ALPHA) * _target_signature
                    + HISTO_UPDATE_ALPHA * fresh_sig
                ).astype(np.float32)

    # Draw annotation.
    cv2.rectangle(img_write, (bx, by), (bx + bw, by + bh), (255, 0, 255), 2)
    cv2.putText(
        img_write,
        f"match:{best_score:.2f}",
        (bx, by - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img_write, "Tracking",
        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
    )

    if show_img:
        cv2.imshow("Real-time Detect", img_write)

    return (cx, cy), float(best_score), (bx, by), max(bw, bh) / 2.0, img_write, best_bbox


def _handle_miss(img_write: np.ndarray,
                 show_img: bool) -> Tuple:
    """
    Increment the miss counter and return the appropriate signal.

    Returns confidence=None in all miss cases.  The caller distinguishes
    within-budget misses (hold, no command) from budget-exceeded misses
    (full reset) by checking _target_track_id: it is cleared to None when
    the budget is exhausted.
    """
    global total_track_misses, confirmed_object_tracking, _target_track_id

    total_track_misses += 1
    confirmed_object_tracking = False

    cv2.putText(
        img_write,
        f"Miss {total_track_misses}/{TRACKER_MISSES_MAX}",
        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA,
    )

    if total_track_misses >= TRACKER_MISSES_MAX:
        LOG.info(
            "Target lost: no histogram match for %d consecutive frames.",
            total_track_misses,
        )
        _target_track_id = None   # Signal to caller: budget exhausted.
        total_track_misses = 0

    if show_img:
        cv2.imshow("Real-time Detect", img_write)

    return (0, 0), None, (0, 0), None, img_write, (0, 0, 0, 0)


# ── Mission / bench perception wrapper ─────────────────────────────────────────

def run_perception_step(
        frame: np.ndarray,
        display: np.ndarray,
        tracker_active: bool,
        frame_counter: int,
        detect_every_n: int = 1,
        detect_confidence: float = 0.25,
        tracker_misses_max: Optional[int] = None,
        reset_fn=None,
):
    """
    Single entry point used by mission.py and run_bench.py.

    State machine:
        SEARCH — YOLO detection every detect_every_n frames.  On the first
                 confident detection, call set_object_to_track() to store
                 the histogram signature, then immediately enter TRACK.

        TRACK  — YOLO detection + histogram matching every frame.  Returns
                 (True, bbox, center, 'track', 'tracking') on a successful
                 match.  On a miss within the budget, returns
                 (True, None, None, 'track', 'hold') so the mission loop
                 sends no commands but does not reset the tracker.  When
                 the miss budget is exhausted, calls reset_fn() and returns
                 (False, None, None, 'lost', 'target lost').

    Return signature (unchanged from object_tracking_y8.py):
        (tracker_active, target_bbox, target_center, perception_state, message)
    """
    global _model_conf, TRACKER_MISSES_MAX

    if tracker_misses_max is not None:
        TRACKER_MISSES_MAX = int(tracker_misses_max)

    _model_conf = float(detect_confidence)

    if _model is None:
        cv2.putText(display, 'Model not loaded', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return False, None, None, 'error', 'model not loaded'

    detect_stride = max(1, int(detect_every_n))

    # ── SEARCH / DETECT state ──────────────────────────────────────────────
    if (not tracker_active) and (frame_counter % detect_stride == 0):
        results = _model(
            frame,
            classes=_PERSON_CLASSES,
            conf=_model_conf,
            imgsz=_model_imgsz,
            half=True,
            device=0,
            max_det=20,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            best_idx = max(range(len(boxes)), key=lambda k: float(boxes[k].conf))
            best = boxes[best_idx]
            conf = float(best.conf)
            if conf >= _model_conf:
                x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                set_object_to_track(frame, bbox)

                # If signature storage failed (degenerate bbox), stay in search.
                if _target_track_id is None:
                    cv2.putText(display, 'Searching...', (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (180, 180, 180), 2, cv2.LINE_AA)
                    return False, None, None, 'search', 'searching'

                cv2.rectangle(display, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 2)
                cv2.putText(display, f'Detect {conf:.2f}',
                            (bbox[0], max(20, bbox[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2, cv2.LINE_AA)
                return True, bbox, center, 'detect', 'target acquired'

        cv2.putText(display, 'Searching...', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return False, None, None, 'search', 'searching'

    # ── TRACK state ────────────────────────────────────────────────────────
    if tracker_active:
        center, confidence, _xy, _radius, display_out, bbox = track_with_confirm(
            frame, display, show_img=False
        )
        if display_out is not None:
            display[:] = display_out

        if confidence is not None and bbox != (0, 0, 0, 0):
            return True, bbox, center, 'track', 'tracking'

        # confidence=None: no match this frame.
        # _target_track_id still set  → within budget, hold position.
        # _target_track_id cleared    → budget exhausted, full reset.
        if _target_track_id is not None:
            cv2.putText(
                display,
                f'Hold ({total_track_misses}/{TRACKER_MISSES_MAX})',
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 165, 255), 2, cv2.LINE_AA,
            )
            return True, None, None, 'track', 'hold'

        if reset_fn is not None:
            reset_fn()
        cv2.putText(display, 'Target lost', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return False, None, None, 'lost', 'target lost'

    cv2.putText(display, 'Searching...', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return False, None, None, 'search', 'searching'


# ── Tracker reset helper ───────────────────────────────────────────────────────

def _reset_tracker_state() -> None:
    """
    Clear all tracking state.  Called internally and by stop_camera_stream().
    mission.py resets state by directly setting:
        obj_track.total_track_misses = 0
        obj_track._target_track_id = None
    Those two assignments are sufficient to fully reset this module because
    track_with_confirm() and run_perception_step() gate on _target_track_id.
    This function provides a convenience reset for the camera stop path.
    """
    global _target_track_id, _target_signature, total_track_misses
    global confirmed_object_tracking, _confirm_frame_counter
    global _last_bbox, _last_center

    _target_track_id = None
    _target_signature = None
    total_track_misses = 0
    confirmed_object_tracking = False
    _confirm_frame_counter = 0
    _last_bbox = (0, 0, 0, 0)
    _last_center = (0, 0)


# ── Camera stream ──────────────────────────────────────────────────────────────

def start_camera_stream(use_distance: bool = False,
                        resolution_width: int = 640,
                        resolution_height: int = 480,
                        color_rate: int = 60,
                        depth_rate: int = 30,
                        enable_imu: bool = False,
                        imu_accel_rate: int = 200,
                        imu_gyro_rate: int = 400) -> None:
    """Compatibility wrapper around cam_handler.start_camera_stream()."""
    global pipeline, streaming_distance, FRAME_WIDTH, FRAME_HEIGHT
    global FRAME_HORIZONTAL_CENTER, FRAME_VERTICAL_CENTER
    global latest_accel_data, latest_gyro_data, last_frame_timestamp

    cam_handler.start_camera_stream(
        use_distance=use_distance,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        color_rate=color_rate,
        depth_rate=depth_rate,
        enable_imu=enable_imu,
        imu_accel_rate=imu_accel_rate,
        imu_gyro_rate=imu_gyro_rate,
    )
    pipeline = cam_handler.get_pipeline()
    FRAME_WIDTH, FRAME_HEIGHT, FRAME_HORIZONTAL_CENTER, FRAME_VERTICAL_CENTER = \
        cam_handler.get_frame_geometry()
    streaming_distance = cam_handler.is_streaming_distance()
    latest_accel_data = cam_handler.get_latest_accel_data()
    latest_gyro_data = cam_handler.get_latest_gyro_data()
    last_frame_timestamp = cam_handler.get_last_frame_timestamp()


def stop_camera_stream() -> None:
    """Compatibility wrapper around cam_handler.stop_camera_stream()."""
    global pipeline, latest_accel_data, latest_gyro_data, last_frame_timestamp

    cam_handler.stop_camera_stream()
    pipeline = None
    latest_accel_data = None
    latest_gyro_data = None
    last_frame_timestamp = 0.0

    _reset_tracker_state()


def get_cur_frame(attempts: int = 5, flip_v: bool = False) -> Optional[np.ndarray]:
    """Compatibility wrapper around cam_handler.get_cur_frame()."""
    global last_frame_timestamp, latest_accel_data, latest_gyro_data
    frame = cam_handler.get_cur_frame(attempts=attempts, flip_v=flip_v)
    last_frame_timestamp = cam_handler.get_last_frame_timestamp()
    latest_accel_data = cam_handler.get_latest_accel_data()
    latest_gyro_data = cam_handler.get_latest_gyro_data()
    return frame


# ── Distance measurement ───────────────────────────────────────────────────────

def get_distance_using_center() -> float:
    """Compatibility wrapper around cam_handler depth access."""
    return cam_handler.get_distance_using_center()


def get_avg_distance_to_obj(seconds: float, virtual_mode: bool = False) -> float:
    """Compatibility wrapper around cam_handler depth access."""
    return cam_handler.get_avg_distance_to_obj(seconds, virtual_mode=virtual_mode)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_centered_circle(image: np.ndarray,
                         radius: int,
                         color: Tuple,
                         thickness: int) -> None:
    """Draw a circle at the frame centre."""
    h, w = image.shape[:2]
    cv2.circle(image, (w // 2, h // 2), radius, color, thickness)


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    engine = '/home/usafa/usafa_472/sentinel_drone/yolo/yolov8m-visdrone.engine'
    if len(sys.argv) > 1:
        engine = sys.argv[1]

    start_camera_stream(use_distance=True, enable_imu=True)
    load_model(engine, imgsz=640, conf=0.25)

    object_identified = False
    n, avg_dist, total_dist, i = 30, 0.0, 0.0, 0

    while True:
        timer = cv2.getTickCount()
        frame = get_cur_frame()
        if frame is None:
            continue

        frm_display = frame.copy()

        if not object_identified:
            results = _model(
                frame,
                classes=_PERSON_CLASSES,
                conf=_model_conf,
                imgsz=_model_imgsz,
                half=True,
                device=0,
                max_det=20,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                best = max(range(len(boxes)), key=lambda k: float(boxes[k].conf))
                conf = float(boxes[best].conf)
                if conf >= _model_conf:
                    x1, y1, x2, y2 = boxes[best].xyxy[0].cpu().numpy().astype(int)
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    set_object_to_track(frame, bbox)
                    if _target_track_id is not None:
                        object_identified = True
                        LOG.info("Initial target acquired: bbox=%s  conf=%.2f", bbox, conf)
        else:
            center, confidence, (x, y), radius, frm_display, bbox = \
                track_with_confirm(frame, frm_display, show_img=True)

            if confidence is None and _target_track_id is None:
                LOG.info("Target lost — returning to search.")
                object_identified = False

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frm_display, f"FPS: {fps:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        distance = get_distance_using_center()
        total_dist += distance
        i += 1
        if i % n == 0:
            avg_dist = total_dist / i
            total_dist = 0.0
            i = 0
            print(f"Average Distance: {avg_dist:.2f} m")

        cv2.putText(frm_display, f"Dist: {distance:.2f} m",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frm_display, f"Avg:  {avg_dist:.2f} m",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        draw_centered_circle(frm_display, 5, (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_camera_stream()
    cv2.destroyAllWindows()
