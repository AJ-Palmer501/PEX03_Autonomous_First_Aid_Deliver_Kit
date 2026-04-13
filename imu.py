"""
sentinel/imu.py
===============
RealSense IMU reader and complementary filter for roll/pitch estimation.

This module is camera-model-agnostic.  It works with any RealSense camera
that exposes accelerometer and gyroscope streams through the pyrealsense2
pipeline.  The specific camera model and IMU rates are configured in
``sentinel_config.json`` — nothing is hardcoded here.

Camera tilt measurement
-----------------------
Before the mission loop starts, the application can call
``ImuReader.measure_tilt()`` to determine the actual downward angle of the
camera by reading the pitch from the IMU while the drone is on level ground
or hovering level.  This replaces the manually-configured ``camera_tilt_deg``
with an accurate, measured value.

Roll compensation
-----------------
The camera is mounted on the front of the drone at a downward tilt.  When
the drone rolls, that roll is projected through the tilted camera onto the
image plane as a false horizontal pixel shift.  This module provides a roll
estimate so the mission loop can subtract the attitude-induced false error.

Complementary filter
--------------------
    angle = alpha * (angle + gyro_rate * dt) + (1 - alpha) * accel_angle

The accelerometer gives drift-free low-frequency tilt; the gyroscope gives
clean high-frequency response.  alpha ~ 0.98 combines the best of both.

Thread safety
-------------
``ImuReader`` runs its own daemon thread.  The latest estimates are protected
by a ``threading.Lock``.  The mission loop reads them via ``get_attitude()``.

Fallback
--------
If the IMU streams are unavailable, ``get_attitude()`` returns zeros so the
calling code always receives a valid (zero-compensation) result.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional, Tuple
import cam_handler

LOG = logging.getLogger("sentinel_drone.imu")


class ImuReader:
    """
    Reads RealSense IMU streams and maintains a complementary-filter
    estimate of camera roll and pitch.

    Parameters
    ----------
    alpha : Complementary filter coefficient for the gyroscope leg (0-1).
    """

    def __init__(
        self,
        alpha: float = 0.98,
        accel_rate: float | None = None,
        gyro_rate: float | None = None,
        **_compat_kwargs,
    ) -> None:
        """
        Create the IMU reader.

        ``accel_rate`` and ``gyro_rate`` are accepted for backward
        compatibility with existing mission / bench call sites.  They are not
        used directly here because the camera polling path now owns RealSense
        stream configuration and this reader consumes cached motion samples
        harvested by ``cam_handler``.
        """
        self.alpha = max(0.0, min(1.0, alpha))
        self.accel_rate = accel_rate
        self.gyro_rate = gyro_rate

        # Complementary filter state (radians)
        self._roll: float = 0.0
        self._pitch: float = 0.0

        # Latest gyroscope yaw-rate in rad/s (camera body frame)
        self._yaw_rate: float = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ts: float = 0.0
        self._available: bool = False
        self._sample_count: int = 0
        self._last_motion_timestamp: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────────

    def start(self, pipeline=None) -> bool:
        """
        Start the IMU polling thread.

        ``pipeline`` is accepted for compatibility with older call sites but is
        no longer consumed directly here.  All motion samples now come from
        ``sentinel.cam_handler`` so this thread never competes for the
        RealSense pipeline.
        """
        try:
            # Keep the old import fallback so attitude math remains independent
            # of the exact pyrealsense2 package layout.
            try:
                import pyrealsense2 as rs  # type: ignore
                if not hasattr(rs, "stream"):
                    import pyrealsense2.pyrealsense2 as rs  # type: ignore
            except Exception:
                import pyrealsense2.pyrealsense2 as rs  # type: ignore

            self._rs = rs
            self._available = True
            self._pipeline = pipeline
        except Exception:
            LOG.warning("pyrealsense2 not available or IMU streams not started; IMU disabled.")
            self._available = False
            return False

        if self._thread is not None and self._thread.is_alive():
            self.stop()

        self._stop_event.clear()
        self._last_ts = 0.0
        self._sample_count = 0
        self._last_motion_timestamp = 0.0
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="sentinel-imu",
            daemon=True,
        )
        self._thread.start()
        LOG.info("IMU reader started (alpha=%.3f).", self.alpha)
        return True

    def stop(self) -> None:
        """Signal the IMU polling thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        LOG.info("IMU reader stopped.")

    # ── Public API ────────────────────────────────────────────────────────────────

    def get_attitude(self) -> Tuple[float, float, float]:
        """
        Return the latest complementary-filter attitude estimate.

        Returns
        -------
        (roll_rad, pitch_rad, yaw_rate_rad_s) : All floats.
            roll_rad       : Camera roll in radians.
                             Positive = right side of camera/drone is lower.
                             This is the value used to compute the false
                             horizontal pixel error introduced by drone attitude.
            pitch_rad      : Camera pitch in radians.
                             Positive = nose UP (camera pointing toward sky).
                             Negative = nose DOWN (camera pointing toward ground).
                             A typical 37.5° downward mount reads approximately
                             −0.654 rad.
            yaw_rate_rad_s : Camera yaw angular velocity in rad/s.
                             Positive = clockwise when viewed from above.

        Thread-safe.  Returns (0, 0, 0) if the IMU is unavailable.
        """
        if not self._available:
            return 0.0, 0.0, 0.0
        with self._lock:
            return self._roll, self._pitch, self._yaw_rate

    def measure_tilt(self, duration_s: float = 2.0, sample_interval: float = 0.05) -> Optional[float]:
        """
        Measure the camera's actual downward tilt angle using the IMU.

        This should be called after ``start()`` while the drone is sitting on
        level ground or hovering in a level position.  The method samples the
        pitch estimate over ``duration_s`` seconds and returns the average tilt
        in degrees as a **positive** value (matching the ``camera_tilt_deg``
        config convention where positive = degrees below horizontal).

        Note: The raw IMU pitch is negative when the camera points downward
        (standard aviation convention: pitch up = positive).  This method
        negates it so the returned value is positive for a downward mount,
        consistent with how ``camera_tilt_deg`` is documented and used
        throughout the application (e.g. in ``compute_roll_pixel_compensation``).

        Parameters
        ----------
        duration_s      : How long to collect IMU samples (seconds).
        sample_interval : Time between samples (seconds).

        Returns
        -------
        float or None : Measured tilt in degrees (positive = below horizontal),
                        or None if the IMU is unavailable or no valid readings
                        were obtained.
        """
        if not self._available:
            LOG.warning("IMU unavailable — cannot measure camera tilt.")
            return None

        LOG.info(
            "Measuring camera tilt from IMU (sampling for %.1f s) …",
            duration_s,
        )

        # Wait briefly for the first real motion sample so we do not
        # silently measure a default zero attitude before the camera reader
        # has produced any IMU data.
        wait_deadline = time.time() + min(2.0, max(0.5, duration_s))
        while time.time() < wait_deadline:
            with self._lock:
                if self._sample_count > 0:
                    break
            time.sleep(0.01)

        with self._lock:
            if self._sample_count == 0:
                LOG.warning("No live IMU samples available yet — cannot measure camera tilt.")
                return None

        # Let the complementary filter settle for a moment before sampling.
        time.sleep(min(0.5, duration_s / 4.0))

        readings = []
        start = time.time()
        while time.time() - start < duration_s:
            _, pitch_rad, _ = self.get_attitude()
            readings.append(pitch_rad)
            time.sleep(sample_interval)

        if not readings:
            LOG.warning("No IMU readings collected — cannot determine tilt.")
            return None

        avg_pitch_rad = sum(readings) / len(readings)

        # The IMU reports negative pitch for a downward-facing camera.
        # Negate to convert to the camera_tilt_deg convention where
        # positive = degrees below horizontal.
        #   IMU pitch for 40° downward mount: -40°
        #   camera_tilt_deg convention:        +40°
        tilt_deg = -math.degrees(avg_pitch_rad)

        # Sanity check: tilt should be between 0 and 90 for a forward-mounted
        # camera.  Values outside this range suggest the drone is not level or
        # the IMU is giving bad data.
        if tilt_deg < 0 or tilt_deg > 90:
            LOG.warning(
                "IMU tilt measurement %.1f° is outside the expected 0-90° range. "
                "The drone may not be level.  Measurement will still be used, but "
                "verify your setup.",
                tilt_deg,
            )

        LOG.info(
            "Camera tilt measured: %.1f° below horizontal "
            "(from %d samples over %.1f s).",
            tilt_deg, len(readings), duration_s,
        )
        return tilt_deg

    # ── Internal ──────────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Continuously update the complementary filter with IMU data.

        Motion samples now come only from ``sentinel.cam_handler``.  That keeps
        all RealSense frame consumption on one owner path and prevents the IMU
        thread from competing with RGB frame acquisition.
        """
        last_error_log: float = 0.0
        while not self._stop_event.is_set():
            try:
                accel, gyro, motion_ts = cam_handler.get_latest_motion_data()

                if accel is None or gyro is None or motion_ts <= 0.0:
                    time.sleep(0.005)
                    continue

                if motion_ts == self._last_motion_timestamp:
                    time.sleep(0.002)
                    continue
                self._last_motion_timestamp = motion_ts

                now = time.time()
                dt = now - self._last_ts if self._last_ts > 0.0 else 0.0
                self._last_ts = now

                accel_roll = math.atan2(-accel.x, -accel.y)
                accel_pitch = math.atan2(accel.z, -accel.y)

                gyro_roll_rate = gyro.z
                gyro_pitch_rate = gyro.x
                gyro_yaw_rate = gyro.y

                if dt > 0.0 and dt < 0.5:
                    new_roll = (
                        self.alpha * (self._roll + gyro_roll_rate * dt)
                        + (1.0 - self.alpha) * accel_roll
                    )
                    new_pitch = (
                        self.alpha * (self._pitch + gyro_pitch_rate * dt)
                        + (1.0 - self.alpha) * accel_pitch
                    )
                else:
                    new_roll = accel_roll
                    new_pitch = accel_pitch

                with self._lock:
                    self._roll = new_roll
                    self._pitch = new_pitch
                    self._yaw_rate = gyro_yaw_rate
                    self._sample_count += 1

            except Exception:
                if time.time() - last_error_log > 5.0:
                    LOG.exception("IMU poll error (suppressing for 5 s).")
                    last_error_log = time.time()
                time.sleep(0.05)


def compute_roll_pixel_compensation(
    roll_rad: float,
    camera_tilt_deg: float,
    frame_width: int,
    fov_h_deg: float,
) -> float:
    """
    Compute the false horizontal pixel error caused by drone roll with a
    tilted camera.

    Parameters
    ----------
    roll_rad       : Current drone roll estimate in radians.
    camera_tilt_deg: Camera tilt below horizontal in degrees.
    frame_width    : Horizontal resolution in pixels.
    fov_h_deg      : Horizontal field-of-view in degrees.

    Returns
    -------
    float : Signed pixel compensation to SUBTRACT from the raw error.
    """
    tilt_rad = math.radians(camera_tilt_deg)
    fov_h_rad = math.radians(fov_h_deg)
    return float(frame_width) * math.sin(roll_rad) * math.cos(tilt_rad) / fov_h_rad
