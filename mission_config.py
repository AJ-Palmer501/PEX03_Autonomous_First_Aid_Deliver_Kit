"""
student/mission_config.py
=========================
Configuration loader for PEX 03 — Drone Delivery Mission.

Usage
-----
    from student.mission_config import MissionConfig

    # Load from the default search path (recommended):
    config = MissionConfig.load()

    # Or specify an explicit path:
    config = MissionConfig.load(path='/path/to/pex03_config.json')

    # Access settings as typed attributes:
    print(config.virtual_mode)          # bool
    print(config.camera_fov_v_deg)      # float
    print(config.weights_path)          # str

Design notes
------------
* All keys that start with '_' in the JSON file are silently ignored —
  they are documentation comment blocks, not settings.

* Every field has a hardcoded default so the mission can run even if
  an optional key is missing from the JSON file.  If you add a new
  setting, add it here with a sensible default AND document it in
  pex03_config.json.

* MissionConfig is intentionally a plain dataclass rather than a
  complex object — students should be able to read the whole class
  and understand every field without any prior framework knowledge.

* load() searches for the config file in a predictable order and logs
  which file it found, so it is always clear which config is active.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

LOG = logging.getLogger("pex03.mission_config")

# Ordered list of paths to check when no explicit path is given.
# The first file found is used.
_DEFAULT_SEARCH_PATHS = [
    "pex03_config.json",                    # current working directory
    "student/pex03_config.json",            # package sub-directory
    os.path.join(os.path.dirname(__file__), "pex03_config.json"),  # alongside this file
]


@dataclass
class MissionConfig:
    """
    All configurable parameters for the PEX 03 delivery mission.

    Every field maps directly to a key in pex03_config.json.
    Defaults are applied for any key not present in the file.

    Field groups
    ------------
    Connection    : connect_string, baud, virtual_mode, takeoff_altitude_m
    Paths         : weights_path, mission_log_path
    Camera HW     : camera_name, camera_has_imu, camera_resolution_*,
                    camera_frame_rate, camera_imu_*_rate, camera_fov_*_deg
    Camera mount  : flip_vertical, camera_tilt_deg, camera_tilt_from_imu,
                    camera_tilt_sample_duration_s
    IMU           : imu_enabled, imu_complementary_alpha
    Distance      : use_estimated_distance
    Detection     : detect_confidence, detect_every_n, model_imgsz,
                    tracker_misses_max, histo_match_threshold
    Mission       : max_confirm_attempts, target_acceptance_radius_px,
                    update_rate, image_log_rate
    Runtime       : camera_tilt_measured_deg (set after IMU tilt measurement,
                    not from the JSON file)
    """

    # ── Connection ────────────────────────────────────────────────────────────
    connect_string: str   = "127.0.0.1:14550"
    baud: int             = 115200
    virtual_mode: bool    = True
    takeoff_altitude_m: float = 15.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    weights_path: str     = "/home/usafa/usafa_472/sentinel_drone/yolo/yolov8s_visdrone.engine"
    mission_log_path: str = "/media/usafa/data/pex03_mission/cam"

    # ── Camera hardware ───────────────────────────────────────────────────────
    camera_name: str               = "Intel RealSense D455"
    camera_has_imu: bool           = True
    camera_resolution_width: int   = 640
    camera_resolution_height: int  = 480
    camera_frame_rate: int         = 60
    camera_imu_accel_rate: int     = 200
    camera_imu_gyro_rate: int      = 400
    camera_fov_h_deg: float        = 87.0
    camera_fov_v_deg: float        = 58.0

    # ── Camera mounting geometry ──────────────────────────────────────────────
    flip_vertical: bool                 = False
    camera_tilt_deg: float              = 37.5   # fallback if IMU not available
    camera_tilt_from_imu: bool          = True
    camera_tilt_sample_duration_s: float = 2.0

    # ── IMU ───────────────────────────────────────────────────────────────────
    imu_enabled: bool               = True
    imu_complementary_alpha: float  = 0.98

    # ── Distance estimation ───────────────────────────────────────────────────
    use_estimated_distance: bool    = True

    # ── Detection / tracking ──────────────────────────────────────────────────
    detect_confidence: float        = 0.25
    detect_every_n: int             = 1
    model_imgsz: int                = 640
    tracker_misses_max: int         = 20
    histo_match_threshold: float    = 0.70

    # ── Mission state machine ─────────────────────────────────────────────────
    max_confirm_attempts: int           = 8
    target_acceptance_radius_px: int    = 10
    update_rate: int                    = 1
    image_log_rate: int                 = 4

    # ── Runtime-only fields (never read from JSON) ────────────────────────────
    # camera_tilt_measured_deg is populated by __main__ after the IMU tilt
    # measurement completes.  It starts as None; once set it is used everywhere
    # in place of camera_tilt_deg.  This keeps the JSON value (the configured
    # fallback) separate from the live measured value.
    camera_tilt_measured_deg: Optional[float] = field(default=None, repr=False)

    # ── Derived property ──────────────────────────────────────────────────────

    @property
    def effective_camera_tilt_deg(self) -> float:
        """
        Return the camera tilt angle that should be used in calculations.

        If an IMU measurement was taken successfully at startup,
        camera_tilt_measured_deg is returned.  Otherwise the configured
        fallback (camera_tilt_deg) is returned.

        This is the single point students should call — they never need
        to check which source is active.
        """
        if self.camera_tilt_measured_deg is not None:
            return self.camera_tilt_measured_deg
        return self.camera_tilt_deg

    @property
    def tilt_source(self) -> str:
        """Human-readable label for which tilt source is active ('imu' or 'config')."""
        return "imu" if self.camera_tilt_measured_deg is not None else "config"

    # ── Loader ────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MissionConfig":
        """
        Load a MissionConfig from a JSON file.

        Parameters
        ----------
        path : Explicit path to a pex03_config.json file.
               If None, the default search path list is tried in order
               and the first file found is used.  If no file is found,
               a default MissionConfig is returned with a warning logged.

        Returns
        -------
        MissionConfig with all settings populated from the JSON file,
        falling back to class defaults for any missing keys.
        """
        resolved_path = cls._find_config_file(path)

        if resolved_path is None:
            LOG.warning(
                "No pex03_config.json found in the default search paths. "
                "Using all hardcoded defaults.  To silence this warning, "
                "create pex03_config.json in the current working directory."
            )
            return cls()

        LOG.info("Loading mission config from: %s", resolved_path)

        try:
            with open(resolved_path, "r", encoding="utf-8") as fh:
                raw: dict = json.load(fh)
        except json.JSONDecodeError as exc:
            LOG.error(
                "pex03_config.json is not valid JSON: %s.  "
                "Using hardcoded defaults.", exc
            )
            return cls()
        except OSError as exc:
            LOG.error(
                "Could not read %s: %s.  Using hardcoded defaults.",
                resolved_path, exc
            )
            return cls()

        # Strip all keys beginning with '_' (comment blocks).
        settings = {k: v for k, v in raw.items() if not k.startswith("_")}

        # Build a config object starting from defaults, then overlay
        # every key that exists in the JSON file and matches a field name.
        cfg = cls()
        unknown_keys = []

        for key, value in settings.items():
            if hasattr(cfg, key):
                # Type-coerce to match the dataclass field type so the
                # JSON value is always the right Python type.
                expected_type = type(getattr(cfg, key))
                try:
                    setattr(cfg, key, expected_type(value))
                except (TypeError, ValueError) as exc:
                    LOG.warning(
                        "Config key '%s': could not coerce value %r to %s "
                        "(%s).  Using default: %r.",
                        key, value, expected_type.__name__, exc,
                        getattr(cfg, key),
                    )
            else:
                unknown_keys.append(key)

        if unknown_keys:
            LOG.warning(
                "Unrecognised config keys (ignored): %s.  "
                "Check pex03_config.json for typos.",
                ", ".join(unknown_keys),
            )

        # Enforce dependent constraints.
        if not cfg.camera_has_imu:
            if cfg.imu_enabled:
                LOG.info(
                    "camera_has_imu is false — forcing imu_enabled to false."
                )
            cfg.imu_enabled = False
            cfg.camera_tilt_from_imu = False

        cls._log_summary(cfg)
        return cfg

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_config_file(explicit_path: Optional[str]) -> Optional[str]:
        """Return the first readable config file path, or None."""
        candidates = [explicit_path] if explicit_path else _DEFAULT_SEARCH_PATHS
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _log_summary(cfg: "MissionConfig") -> None:
        """Log a concise summary of the loaded configuration."""
        LOG.info("=" * 56)
        LOG.info("  Mission Configuration Summary")
        LOG.info("=" * 56)
        LOG.info("  Mode          : %s", "VIRTUAL (SITL)" if cfg.virtual_mode else "LIVE HARDWARE")
        LOG.info("  Autopilot     : %s  baud=%d", cfg.connect_string, cfg.baud)
        LOG.info("  Takeoff alt   : %.1f m", cfg.takeoff_altitude_m)
        LOG.info("  Log path      : %s", cfg.mission_log_path)
        LOG.info("  YOLO model    : %s", cfg.weights_path)
        LOG.info("  Camera        : %s  IMU=%s", cfg.camera_name, cfg.camera_has_imu)
        LOG.info("  Resolution    : %dx%d @ %d fps",
                 cfg.camera_resolution_width,
                 cfg.camera_resolution_height,
                 cfg.camera_frame_rate)
        LOG.info("  FOV           : H=%.1f°  V=%.1f°",
                 cfg.camera_fov_h_deg, cfg.camera_fov_v_deg)
        LOG.info("  Tilt config   : %.1f°  from_imu=%s",
                 cfg.camera_tilt_deg, cfg.camera_tilt_from_imu)
        LOG.info("  IMU           : enabled=%s  alpha=%.2f",
                 cfg.imu_enabled, cfg.imu_complementary_alpha)
        LOG.info("  Distance est. : %s",
                 "camera geometry" if cfg.use_estimated_distance else "rangefinder")
        LOG.info("  Detection     : conf=%.2f  every_n=%d  imgsz=%d",
                 cfg.detect_confidence, cfg.detect_every_n, cfg.model_imgsz)
        LOG.info("  Tracker       : misses_max=%d  histo_thresh=%.2f",
                 cfg.tracker_misses_max, cfg.histo_match_threshold)
        LOG.info("  Mission       : confirm_attempts=%d  radius_px=%d",
                 cfg.max_confirm_attempts, cfg.target_acceptance_radius_px)
        LOG.info("=" * 56)
