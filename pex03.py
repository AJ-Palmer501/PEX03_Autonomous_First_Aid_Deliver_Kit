"""
pex03.py
========
PEX 03 — Autonomous Drone Delivery Mission

This is the top-level mission file.  It contains:
  * The DroneMission class — the state machine that flies the mission.
  * The __main__ entry point — startup, hardware checks, and mission launch.

All tunable parameters live in pex03_config.json.
Do NOT hardcode values here — change the config file instead.

Mission flow (see conduct_mission() for full detail):

    SEEK → CONFIRM → TARGET → DELIVER → RTL

Dependencies
------------
  drone_lib                  : DroneKit / MAVLink flight control abstraction
  object_tracking_y8_histo   : YOLOv8 detection + histogram tracker
  pex03_utils                : Geometry helpers, gripper, frame I/O
  mission_config             : Typed config loader (reads pex03_config.json)
  imu                        : RealSense IMU reader + complementary filter
"""

import logging
import time
import cv2
import random
import sys
import traceback

import drone_lib
import object_tracking_y8_histo as obj_track
import pex03_utils
from mission_config import MissionConfig
from imu import ImuReader

# =============================================================================
# MISSION STATE CONSTANTS
# =============================================================================
# The drone is always in exactly one of these states.  All transitions are
# made in conduct_mission() — nowhere else.
#
# Full mission flow:
#
#   ┌──────────────────────────────────────────────────────────────────────┐
#   │                                                                      │
#   │  ┌────────┐  candidate    ┌─────────┐  confirmed   ┌────────────┐    │
#   │  │  SEEK  │ ────────────► │ CONFIRM │ ───────────► │   TARGET   │    │
#   │  │(AUTO)  │ ◄──────────── │(GUIDED) │              │  (GUIDED)  │    │
#   │  └────────┘  not          └─────────┘              └─────┬──────┘    │
#   │              confirmed                                   │           │
#   │                                                   centered on target │
#   │                                                          ▼           │
#   │                                                    ┌────────────┐    │
#   │                                                    │  DELIVER   │    │
#   │                                                    │  (GUIDED)  │    │
#   │                                                    └─────┬──────┘    │
#   │                                                          │           │
#   │                                                   package released   │
#   │                                                          ▼           │
#   │                                                    ┌────────────┐    │
#   │                                                    │    RTL     │    │
#   │                                                    │(return home)    │
#   │                                                    └────────────┘    │
#   └──────────────────────────────────────────────────────────────────────┘

MISSION_MODE_SEEK    = 0   # Flying AUTO waypoints, scanning for a target
MISSION_MODE_CONFIRM = 1   # Candidate spotted; repositioning to confirm
MISSION_MODE_TARGET  = 2   # Confirmed; maneuvering to centre over target
MISSION_MODE_DELIVER = 4   # centered; calculating drop point and delivering
MISSION_MODE_RTL     = 8   # Delivery done (or aborted); returning home

# Font for cv2 text annotations
IMG_FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# DroneMission CLASS
# =============================================================================
class DroneMission:
    """
    Encapsulates the autonomous delivery mission for one drone.

    The entire mission is driven by conduct_mission().  All state transitions
    happen in that single method so the flow is always readable in one place.

    Helper methods (adjust_to_target_center, deliver_package) perform the
    *work* of a state and return True/False to indicate completion — they
    never modify self.mission_mode directly.

    Parameters
    ----------
    device : DroneKit vehicle object returned by drone_lib.connect_device().
    config : MissionConfig instance loaded from pex03_config.json.
    """

    def __init__(self, device, config: MissionConfig):

        # ── Core references ───────────────────────────────────────────────────
        self.drone  = device
        self.config = config

        # ── Configuration shortcuts ───────────────────────────────────────────
        # Pull frequently-used values out of config so the code below reads
        # cleanly without long chains of self.config.xxx everywhere.
        self.virtual_mode           = config.virtual_mode
        self.update_rate            = config.update_rate
        self.target_radius          = config.target_acceptance_radius_px
        self.image_log_rate         = config.image_log_rate
        self.log_path               = config.mission_log_path
        self.max_confirm_attempts   = config.max_confirm_attempts

        # ── State machine ─────────────────────────────────────────────────────
        self.mission_mode       = MISSION_MODE_SEEK
        self.refresh_counter    = 0
        self.confirm_attempts   = 0
        self.object_identified  = False
        self.inside_circle      = False
        self.direction_x        = "unknown"
        self.direction_y        = "unknown"

        # ── Initial sighting record ───────────────────────────────────────────
        # GPS position where the drone first spotted a candidate target.
        # Used to fly back for confirmation.
        self.init_obj_lat     = None
        self.init_obj_lon     = None
        self.init_obj_alt     = None
        self.init_obj_heading = None

        # ── Last GPS snapshot ─────────────────────────────────────────────────
        # Updated every frame — records where the drone was at the exact
        # moment the corresponding camera frame was captured.
        self.last_lat_pos     = -1.0
        self.last_lon_pos     = -1.0
        self.last_alt_pos     = -1.0
        self.last_heading_pos = -1.0

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def log_info(msg):
        """Write a message to the mission log."""
        log.info(msg)

    def arm_drone(self):
        """Arm the drone (convenience wrapper)."""
        drone_lib.arm_device(self.drone)

    # =========================================================================
    # GEOMETRY HELPERS
    # These methods answer spatial questions about the target's position
    # relative to the drone's camera.  They do not issue flight commands.
    # =========================================================================

    def target_is_centered(self, target_point, frame_write=None):
        """
        Check whether the target is inside the acceptance circle and annotate
        the display frame with alignment indicators.

        Parameters
        ----------
        target_point : (cx, cy) pixel coordinates of the target's centre.
        frame_write  : Display frame to annotate (optional).

        Returns
        -------
        bool : True if the target is inside the acceptance zone.
        """
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])
        self.log_info(f"Alignment — dx={dx:.1f} px, dy={dy:.1f} px")

        x, y = target_point
        if frame_write is not None:
            # Red line: target centre → frame centre (shows offset visually)
            cv2.line(frame_write, target_point,
                     (int(obj_track.FRAME_HORIZONTAL_CENTER),
                      int(obj_track.FRAME_VERTICAL_CENTER)),
                     (0, 0, 255), 5)
            # Inner red circle = acceptance zone
            cv2.circle(frame_write, (int(x), int(y)),
                       int(self.target_radius), (0, 0, 255), 2)
            # Outer yellow circle = approach warning zone
            cv2.circle(frame_write, (int(x), int(y)),
                       int(self.target_radius * 1.5), (255, 255, 0), 2)
            # Cyan dot at target centre
            cv2.circle(frame_write, target_point, 5, (0, 255, 255), -1)

        return self.check_in_circle(target_point)

    def check_in_circle(self, target_point):
        """
        Return True if target_point is within self.target_radius pixels of
        the frame centre.  Uses:  (x − cx)² + (y − cy)² ≤ r²
        """
        return (
            (int(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER) ** 2
            + (int(target_point[1]) - obj_track.FRAME_VERTICAL_CENTER) ** 2
            <= self.target_radius ** 2
        )

    # =========================================================================
    # STATE WORKER METHODS
    # These methods do the *work* of a state.
    # IMPORTANT: None of them change self.mission_mode.
    #            ALL transitions happen in conduct_mission() only.
    # =========================================================================

    def adjust_to_target_center(self, target_point, frame_write=None):
        """
        Issue small corrective velocity commands to centre the drone over
        the target.

        Returns True when centered (signals conduct_mission() to transition
        to DELIVER).  Returns False while still maneuvering.

        Parameters
        ----------
        target_point : (cx, cy) pixel coordinates of the tracked target.
        frame_write  : Display frame to annotate with direction indicators.
        """
        # Pixel offset of target from frame centre.
        # dx > 0 → target RIGHT  → drone moves RIGHT
        # dx < 0 → target LEFT   → drone moves LEFT
        # dy > 0 → target ABOVE  → drone moves FORWARD
        # dy < 0 → target BELOW  → drone moves BACKWARD
        dx = float(target_point[0]) - obj_track.FRAME_HORIZONTAL_CENTER
        dy = obj_track.FRAME_VERTICAL_CENTER - float(target_point[1])

        # TODO: Decide how many pixels of offset are acceptable before
        #       issuing a correction command.  Consider: at mission altitude,
        #       how many metres does 1 pixel of offset represent?
        pixel_forgiveness = 1   # pixels — TODO: adjust this value!

        self.direction_x = "C"
        self.direction_y = "C"

        if abs(dx) > pixel_forgiveness:
            self.direction_x = "R" if dx > 0 else "L"
        if abs(dy) > pixel_forgiveness:
            self.direction_y = "F" if dy > 0 else "B"

        self.log_info(f"Targeting — dx={dx:.1f}, dy={dy:.1f} | "
                      f"dir_x={self.direction_x}, dir_y={self.direction_y}")

        # centered on both axes (or already inside the acceptance circle)?
        # Return True to signal conduct_mission() to begin delivery.
        if (self.direction_x == "C" and self.direction_y == "C") \
                or self.inside_circle:
            drone_lib.log_activity("On target — ready to deliver!")
            if frame_write is not None:
                cv2.putText(frame_write, "On target — delivering!",
                            (10, 400), IMG_FONT, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return True

        # ── Still off-centre: issue velocity corrections ──────────────────────

        # TODO: Choose a pixel-distance threshold.  When the offset is LARGER
        #       than this, use a faster correction velocity to close the gap
        #       quickly.  When the offset is smaller, use a slower velocity
        #       to avoid overshooting the target.
        pixel_distance_threshold = -1   # TODO: set a sensible value!

        # ── Y-axis (forward / backward) ───────────────────────────────────────
        if self.direction_y != "C":
            if self.direction_y == "F":
                if frame_write is not None:
                    cv2.putText(frame_write, "Moving forward...",
                                (10, 200), IMG_FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if abs(dy) > pixel_distance_threshold:
                    # TODO: Set a forward velocity (yv) for large offsets.
                    # yv = ???
                    pass   # TODO: remove once yv is set
                else:
                    # TODO: Set a slower forward velocity (yv) for fine tuning.
                    # yv = ???
                    pass   # TODO: remove once yv is set
                # TODO: drone_lib.small_move_forward(self.drone, velocity=yv)
                #   or: drone_lib.move_local(self.drone, yv, 0, 0.0)
                pass   # TODO: remove once move command is added
            else:
                if frame_write is not None:
                    cv2.putText(frame_write, "Moving backward...",
                                (10, 200), IMG_FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if abs(dy) > pixel_distance_threshold:
                    # TODO: yv = ???
                    pass
                else:
                    # TODO: yv = ???
                    pass
                # TODO: drone_lib.small_move_back(self.drone, velocity=yv)
                pass

        # ── X-axis (left / right) ─────────────────────────────────────────────
        if self.direction_x != "C":
            if self.direction_x == "R":
                if frame_write is not None:
                    cv2.putText(frame_write, "Moving right...",
                                (10, 300), IMG_FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if abs(dx) > pixel_distance_threshold:
                    # TODO: xv = ???
                    pass
                else:
                    # TODO: xv = ???
                    pass
                # TODO: drone_lib.small_move_right(self.drone, velocity=xv)
                pass
            else:
                if frame_write is not None:
                    cv2.putText(frame_write, "Moving left...",
                                (10, 300), IMG_FONT, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if abs(dx) > pixel_distance_threshold:
                    # TODO: xv = ???
                    pass
                else:
                    # TODO: xv = ???
                    pass
                # TODO: drone_lib.small_move_left(self.drone, velocity=xv)
                pass

        return False   # still adjusting

    def deliver_package(self, target_center, frame_write=None):
        """
        Calculate the drop point, fly there, lower the package, and release it.

        When this method returns, conduct_mission() transitions to RTL.

        Distance to the target is determined by one of two methods depending
        on config.use_estimated_distance:

          True  → estimate_ground_distance_m() uses camera geometry
                  (pixel row + tilt angle + altitude).  No rangefinder needed.
          False → get_avg_distance_to_obj() reads the rangefinder sensor.

        Parameters
        ----------
        target_center : (cx, cy) pixel coordinates of the tracked target.
                        Used by the camera-geometry distance estimator.
        frame_write   : Display frame to annotate with delivery status.
        """
        self.log_info("Deliver: beginning delivery sequence...")

        location = self.drone.location.global_relative_frame
        lon      = location.lon
        lat      = location.lat
        alt      = location.alt
        heading  = self.drone.heading

        # ── Step 1: Measure distance to target ────────────────────────────────
        if self.config.use_estimated_distance:
            # Camera-geometry estimation — no rangefinder required.
            # estimate_ground_distance_m() uses:
            #   - The target's pixel row in the frame
            #   - The camera's vertical FOV
            #   - The effective tilt angle (IMU-measured or config fallback)
            #   - The drone's current GPS altitude
            self.log_info("Deliver: estimating ground distance from camera geometry...")
            ground_dist = pex03_utils.estimate_ground_distance_m(
                camera_tilt_deg  = self.config.effective_camera_tilt_deg,
                camera_fov_v_deg = self.config.camera_fov_v_deg,
                target_y_px      = float(target_center[1]),
                frame_height     = obj_track.FRAME_HEIGHT,
                altitude_m       = alt,
            )
            self.log_info(f"Deliver: estimated ground distance = {ground_dist:.2f} m "
                          f"(tilt source: {self.config.tilt_source}, "
                          f"tilt used: {self.config.effective_camera_tilt_deg:.1f}°)")

        else:
            # Rangefinder path — hardware sensor required.
            # get_avg_distance_to_obj() averages readings over several seconds
            # to smooth out noise.  In virtual mode it returns a fixed value.
            self.log_info("Deliver: reading slant distance from rangefinder...")
            slant_dist = pex03_utils.get_avg_distance_to_obj(
                5.0, self.drone, self.virtual_mode)
            self.log_info(f"Deliver: slant distance (hypotenuse) = {slant_dist:.2f} m, "
                          f"altitude = {alt:.2f} m")

            if slant_dist <= 0:
                self.log_info("Deliver: invalid rangefinder reading — aborting delivery.")
                return

            # Convert slant distance + altitude to horizontal ground distance.
            # Pythagoras:  ground² = slant² − altitude²
            # TODO: Use pex03_utils.get_ground_distance() to calculate this.
            # HINT: ground_dist = pex03_utils.get_ground_distance(alt, slant_dist)
            ground_dist = 0   # TODO: replace this!
            self.log_info(f"Deliver: ground distance = {ground_dist:.2f} m")

        # Guard: if the distance estimate is invalid, abort.
        if ground_dist <= 0:
            self.log_info("Deliver: ground distance is invalid — aborting delivery.")
            return

        # ── Step 2: Calculate the GPS coordinates of the drop point ──────────
        # We know: current lat/lon, the drone's heading (bearing), and the
        # horizontal ground distance to the target.
        # calc_new_location() projects that distance along the heading bearing
        # to give us the GPS coordinates of the drop point.
        #
        # Aim for ~3 m (≈10 ft) from the target — close enough to be useful,
        # far enough not to risk hitting the person.
        self.log_info(f"Deliver: current position = lat={lat:.6f}, lon={lon:.6f}, "
                      f"heading={heading:.1f}°")

        # TODO: Use pex03_utils.calc_new_location() to compute the drop coords.
        # HINT: new_lat, new_lon = pex03_utils.calc_new_location(
        #           lat, lon, heading, ground_dist)
        new_lat = 0.0   # TODO: replace this!
        new_lon = 0.0   # TODO: replace this!
        self.log_info(f"Deliver: drop point = lat={new_lat:.6f}, lon={new_lon:.6f}")

        # ── Step 3: Fly to the drop point ─────────────────────────────────────
        # TODO: Command the drone to fly to the drop coordinates.
        # HINT: drone_lib.goto_point(self.drone, new_lat, new_lon, speed, alt)
        pass   # TODO: remove once goto_point() is called

        if frame_write is not None:
            cv2.putText(frame_write, "Delivering...",
                        (10, 400), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)
            pex03_utils.write_frame(self.refresh_counter, frame_write, self.log_path)

        # ── Step 4: Lower the package ─────────────────────────────────────────
        if self.virtual_mode:
            # In the simulator we land in place to demonstrate the delivery step.
            self.log_info("Deliver (virtual): landing in place to simulate delivery.")
            drone_lib.device_land(self.drone)

        else:
            # Two-phase descent: fast until near the ground, then slow/fine
            # for the final approach before releasing the package.

            # TODO: Set the altitude boundary between the fast and slow phases.
            #       At what height does a fast descent become unsafe?
            alt_thresh = -1   # TODO: set a sensible altitude (metres)!

            self.log_info("Deliver: lowering package (fast descent)...")
            while self.drone.location.global_relative_frame.alt > alt_thresh:
                if (self.drone.mode == "RTL" or self.drone.mode == "LAND"
                        or self.mission_mode == MISSION_MODE_RTL):
                    self.log_info("Deliver: abort signal — stopping descent.")
                    break
                # TODO: drone_lib.small_move_down(self.drone, velocity=<fast>)
                pass   # TODO: remove once descent command is added

            self.log_info("Deliver: switching to slow/fine descent...")
            while self.drone.location.global_relative_frame.alt > 3.20:
                if (self.drone.mode == "RTL" or self.drone.mode == "LAND"
                        or self.mission_mode == MISSION_MODE_RTL):
                    self.log_info("Deliver: abort signal — stopping fine descent.")
                    break
                # TODO: drone_lib.small_move_down(self.drone, velocity=<slow>)
                pass   # TODO: remove once descent command is added

        # ── Step 5: Release the package ───────────────────────────────────────
        # TODO: Call pex03_utils.release_grip() to open the latch.
        # HINT: pex03_utils.release_grip(self.drone, seconds=2)
        self.log_info("Deliver: releasing package...")
        time.sleep(2)   # brief pause to ensure the latch has fully opened

        drone_lib.log_activity("Package delivered — returning home.")
        if frame_write is not None:
            cv2.putText(frame_write, "Returning home...",
                        (10, 500), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # =========================================================================
    # MAIN MISSION LOOP — THE STATE MACHINE
    # =========================================================================

    def conduct_mission(self):
        """
        Run the autonomous delivery mission from start to finish.

        Loops once per camera frame.  Each iteration:
          1. Checks for abort conditions (GCS RTL / LAND override)
          2. Grabs the latest camera frame
          3. Records the drone's GPS position for that frame
          4. Executes the current state's logic
          5. Decides whether to transition — and logs every transition clearly

        ALL changes to self.mission_mode happen here and ONLY here.

        State transitions
        -----------------
        SEEK    → CONFIRM  : candidate detected above confidence threshold
        CONFIRM → TARGET   : tracker still has a lock at the sighting location
        CONFIRM → SEEK     : max confirm attempts exhausted
        TARGET  → DELIVER  : adjust_to_target_center() returns True (centered)
        TARGET  → SEEK     : tracker loses target entirely
        DELIVER → RTL      : deliver_package() returns (delivery complete)
        """
        self.log_info("=" * 60)
        self.log_info("MISSION STARTED")
        self.log_info("=" * 60)
        self.object_identified = False

        while self.drone.armed:

            # =================================================================
            # ABORT CHECK
            # =================================================================
            # Always check first — this ensures the GCS can override the
            # mission at any time by switching to RTL or LAND mode.
            if (self.drone.mode == "RTL"
                    or self.drone.mode == "LAND"
                    or self.mission_mode == MISSION_MODE_RTL):
                self.log_info("Abort/RTL detected — ending mission loop.")
                break

            # =================================================================
            # FRAME CAPTURE
            # =================================================================
            # The camera runs in a background thread (cam_handler), so this
            # returns immediately with the latest available frame.
            #
            # TODO: Get the current camera frame from the tracker module.
            # HINT: frame = obj_track.get_cur_frame()
            frame = None   # TODO: replace this!

            if frame is None:
                continue   # camera not ready yet — skip this iteration

            # =================================================================
            # GPS SNAPSHOT
            # =================================================================
            # Record the drone's GPS position at the same instant as this frame.
            # When we first spot a target we need to know exactly where the
            # drone WAS at that moment so we can fly back to confirm.
            location = self.drone.location.global_relative_frame

            # TODO: Update self.last_lat_pos, self.last_lon_pos, self.last_alt_pos
            #       from the location object so we always have a current record.
            # HINT: self.last_lat_pos = location.lat
            #       self.last_lon_pos = location.lon
            #       self.last_alt_pos = location.alt
            self.last_heading_pos = self.drone.heading

            # Annotate a copy of the frame so the original stays clean for
            # the tracker's histogram computation.
            frm_display = frame.copy()
            timer = cv2.getTickCount()

            # =================================================================
            # STATE MACHINE
            # =================================================================

            # -----------------------------------------------------------------
            # STATE: SEEK
            # -----------------------------------------------------------------
            # The drone is flying its pre-loaded AUTO waypoint mission.
            # Every update_rate-th frame we run a non-committing YOLO scan.
            # If a candidate is spotted with sufficient confidence we record
            # the sighting location and switch to CONFIRM.
            #
            # Exit: → CONFIRM  when confidence > conf_level
            # -----------------------------------------------------------------
            if self.mission_mode == MISSION_MODE_SEEK:

                self.log_info("[SEEK] Scanning for target...")

                if self.drone.mode != "AUTO":
                    self.log_info("[SEEK] Returning to AUTO mode.")
                    drone_lib.change_device_mode(device=self.drone, mode="AUTO")

                if self.refresh_counter % self.update_rate == 0:

                    # Non-committing scan — does NOT lock the tracker.
                    center, confidence, corner, radius, frm_display, bbox \
                        = obj_track.check_for_initial_target(
                            frame, frm_display,
                            show_img=self.virtual_mode)

                    # TODO: Set the detection confidence threshold.
                    #       Too HIGH → miss real targets.
                    #       Too LOW  → waste time on false positives.
                    # HINT: The model's raw confidence rarely reaches 0.99 —
                    #       a value in the 0.3–0.6 range is more realistic.
                    conf_level = 0.99   # TODO: adjust this value!

                    if confidence is not None and confidence > conf_level:

                        # ── TRANSITION: SEEK → CONFIRM ────────────────────────
                        self.log_info(
                            f"[SEEK → CONFIRM] Candidate detected "
                            f"(conf={confidence:.2f}) at "
                            f"lat={location.lat:.6f}, lon={location.lon:.6f}, "
                            f"alt={location.alt:.1f} m.")

                        # Lock the histogram tracker so we can re-identify
                        # the same person after maneuvering back.
                        # TODO: Lock the tracker onto this detection.
                        # HINT: obj_track.set_object_to_track(frame, bbox)
                        self.object_identified = True

                        # Record where the candidate was first seen.
                        self.init_obj_lat     = location.lat
                        self.init_obj_lon     = location.lon
                        self.init_obj_alt     = location.alt
                        self.init_obj_heading = self.drone.heading

                        pex03_utils.write_frame(self.refresh_counter,
                                                frm_display, self.log_path)

                        # Switch to CONFIRM and fly back to the sighting.
                        # MIS_RESTART must = 0 on the autopilot so the waypoint
                        # mission RESUMES (not restarts) when we return to AUTO.
                        self.mission_mode     = MISSION_MODE_CONFIRM
                        self.confirm_attempts = 0

                        self.log_info("[SEEK → CONFIRM] Switching to GUIDED and "
                                      "repositioning over sighting location.")
                        drone_lib.change_device_mode(device=self.drone, mode="GUIDED")
                        drone_lib.goto_point(self.drone,
                                             self.init_obj_lat,
                                             self.init_obj_lon,
                                             2.5,
                                             self.init_obj_alt)
                        drone_lib.condition_yaw(self.drone, self.last_heading_pos)
                        time.sleep(4)   # stabilise before imaging

            # -----------------------------------------------------------------
            # STATE: CONFIRM
            # -----------------------------------------------------------------
            # Hovering at the sighting location.  Check each frame whether the
            # histogram tracker still has a lock.  If so, confirmed → TARGET.
            # If not, rotate to a new angle and try again.  Give up and return
            # to SEEK after max_confirm_attempts failures.
            #
            # Exit: → TARGET  if tracker lock confirmed
            #        → SEEK   if max_confirm_attempts exhausted
            # -----------------------------------------------------------------
            elif self.mission_mode == MISSION_MODE_CONFIRM:

                self.log_info(
                    f"[CONFIRM] Attempt {self.confirm_attempts + 1} "
                    f"/ {self.max_confirm_attempts}")

                if self.object_identified:

                    # ── TRANSITION: CONFIRM → TARGET ──────────────────────────
                    self.log_info("[CONFIRM → TARGET] Target CONFIRMED. "
                                  "Beginning centring manoeuvre.")
                    self.mission_mode = MISSION_MODE_TARGET

                elif self.confirm_attempts >= self.max_confirm_attempts:

                    # ── TRANSITION: CONFIRM → SEEK ────────────────────────────
                    self.log_info(
                        f"[CONFIRM → SEEK] {self.max_confirm_attempts} attempts "
                        "exhausted.  Target NOT confirmed — resuming AUTO.")
                    self.object_identified = False
                    self.mission_mode = MISSION_MODE_SEEK
                    drone_lib.change_device_mode(device=self.drone, mode="AUTO")

                else:
                    # Try a different angle — random yaw for a fresh perspective.
                    self.log_info("[CONFIRM] Re-acquiring at a new angle...")
                    cv2.putText(frm_display, "Re-acquiring target...",
                                (10, 250), IMG_FONT, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    drone_lib.goto_point(self.drone,
                                         self.init_obj_lat,
                                         self.init_obj_lon,
                                         2.5,
                                         self.init_obj_alt)
                    drone_lib.condition_yaw(self.drone, random.random() * 180)
                    time.sleep(2)
                    self.confirm_attempts += 1

            # -----------------------------------------------------------------
            # STATE: TARGET
            # -----------------------------------------------------------------
            # Tracking the confirmed target frame-by-frame and issuing small
            # corrective velocity commands via adjust_to_target_center().
            # Transition to DELIVER when centered; back to SEEK if target lost.
            #
            # Exit: → DELIVER  when adjust_to_target_center() returns True
            #        → SEEK    when tracker loses target entirely
            # -----------------------------------------------------------------
            elif self.mission_mode == MISSION_MODE_TARGET:

                self.log_info("[TARGET] Tracking and centring...")

                center, confidence, corner, radius, frm_display, bbox = \
                    obj_track.track_with_confirm(frame, frm_display,
                                                 show_img=self.virtual_mode)

                if confidence is None and obj_track._target_track_id is None:

                    # ── TRANSITION: TARGET → SEEK ──────────────────────────────
                    self.log_info("[TARGET → SEEK] Tracker lost target. "
                                  "Resuming SEEK.")
                    cv2.putText(frm_display, "LOST TARGET — returning to SEEK",
                                (10, 400), IMG_FONT, 1, (0, 255, 255), 2, cv2.LINE_AA)
                    self.object_identified = False
                    self.mission_mode = MISSION_MODE_SEEK

                else:
                    self.inside_circle = self.target_is_centered(center, frm_display)
                    self.log_info(f"[TARGET] Inside acceptance circle: "
                                  f"{self.inside_circle}")

                    centered = self.adjust_to_target_center(center, frm_display)

                    if centered:
                        # ── TRANSITION: TARGET → DELIVER ──────────────────────
                        self.log_info("[TARGET → DELIVER] centered. "
                                      "Starting delivery.")
                        self.mission_mode = MISSION_MODE_DELIVER

            # -----------------------------------------------------------------
            # STATE: DELIVER
            # -----------------------------------------------------------------
            # centered over the target.  Calculate the drop point using either
            # camera geometry (if use_estimated_distance = true) or the
            # rangefinder.  Fly there, lower and release the package.
            #
            # Exit: → RTL  when deliver_package() returns
            # -----------------------------------------------------------------
            elif self.mission_mode == MISSION_MODE_DELIVER:

                self.log_info("[DELIVER] Delivery in progress...")
                # Pass target_center so the geometry estimator can use the
                # target's pixel row when use_estimated_distance = true.
                self.deliver_package(center, frm_display)

                # ── TRANSITION: DELIVER → RTL ──────────────────────────────────
                self.log_info("[DELIVER → RTL] Delivery complete. "
                              "Commanding return-to-launch.")
                self.mission_mode = MISSION_MODE_RTL

                # TODO: Command the drone to return home.
                # HINT: drone_lib.return_to_launch(self.drone)
                pass   # TODO: remove once return_to_launch() is called

            # -----------------------------------------------------------------
            # STATE: RTL
            # -----------------------------------------------------------------
            # Abort check at the top of the loop will catch MISSION_MODE_RTL
            # on the next iteration.  This branch provides an explicit break
            # if we reach here mid-loop.
            # -----------------------------------------------------------------
            elif self.mission_mode == MISSION_MODE_RTL:
                self.log_info("[RTL] Return-to-launch — ending mission loop.")
                break

            # =================================================================
            # FRAME DISPLAY AND LOGGING
            # =================================================================
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(frm_display, f"FPS: {int(fps)}",
                        (100, 50), IMG_FONT, 0.75, (50, 170, 50), 2)

            state_labels = {
                MISSION_MODE_SEEK:    "SEEK",
                MISSION_MODE_CONFIRM: "CONFIRM",
                MISSION_MODE_TARGET:  "TARGET",
                MISSION_MODE_DELIVER: "DELIVER",
                MISSION_MODE_RTL:     "RTL",
            }
            cv2.putText(frm_display,
                        f"State: {state_labels.get(self.mission_mode, '?')}",
                        (100, 25), IMG_FONT, 0.6, (200, 200, 0), 2)

            if self.virtual_mode:
                cv2.imshow("Real-time Detect", frm_display)
                cv2.imshow("Raw", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.log_info("User pressed Q — ending mission.")
                    break

            if self.refresh_counter % self.image_log_rate == 0:
                pex03_utils.write_frame(self.refresh_counter,
                                        frm_display, self.log_path)
            self.refresh_counter += 1

        self.log_info("=" * 60)
        self.log_info("MISSION LOOP ENDED")
        self.log_info("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':

    # ── Load configuration ────────────────────────────────────────────────────
    # MissionConfig.load() searches for pex03_config.json in the standard
    # locations and returns a fully-typed config object.  All mission
    # parameters come from this object — no hardcoded values below.
    config = MissionConfig.load()

    # ── Logging ───────────────────────────────────────────────────────────────
    pex03_utils.backup_prev_experiment(config.mission_log_path)
    log_file = time.strftime(
        config.mission_log_path + "/Cam_PEX03_%Y%m%d-%H%M%S") + ".log"
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    log = logging.getLogger(__name__)

    log.info("PEX 03 — Drone Delivery Mission — program start.")

    # ── Camera ────────────────────────────────────────────────────────────────
    # Start the camera pipeline.  If the camera has an IMU and imu_enabled
    # is true, the IMU streams are also enabled so tilt can be measured below.
    enable_imu = config.camera_has_imu and config.imu_enabled
    log.info("Starting camera stream (IMU streams: %s)...", enable_imu)
    obj_track.start_camera_stream(
        use_distance         = not config.use_estimated_distance,
        resolution_width     = config.camera_resolution_width,
        resolution_height    = config.camera_resolution_height,
        color_rate           = config.camera_frame_rate,
        enable_imu           = enable_imu,
        imu_accel_rate       = config.camera_imu_accel_rate,
        imu_gyro_rate        = config.camera_imu_gyro_rate,
    )

    # ── YOLO model ────────────────────────────────────────────────────────────
    log.info("Loading YOLOv8 VisDrone model: %s", config.weights_path)
    obj_track.load_model(
        config.weights_path,
        imgsz = config.model_imgsz,
        conf  = config.detect_confidence,
    )

    # Apply tracker tuning values from config to the tracker module's globals.
    obj_track.TRACKER_MISSES_MAX    = config.tracker_misses_max
    obj_track.HISTO_MATCH_THRESHOLD = config.histo_match_threshold
    log.info("Model loaded.  tracker_misses_max=%d  histo_threshold=%.2f",
             config.tracker_misses_max, config.histo_match_threshold)

    # ── IMU tilt measurement ──────────────────────────────────────────────────
    # Measure the camera's actual downward tilt while the drone is sitting
    # still on level ground — BEFORE arming so there is no vibration.
    # The result is stored in config.camera_tilt_measured_deg and used
    # automatically by config.effective_camera_tilt_deg wherever tilt is needed.
    imu_reader = None
    if enable_imu and config.camera_tilt_from_imu:
        log.info("Starting IMU reader for pre-arm tilt measurement...")
        imu_reader = ImuReader(alpha=config.imu_complementary_alpha)
        imu_reader.start()

        log.info("Measuring camera tilt from IMU "
                 "(%.1f s) — keep drone level and still...",
                 config.camera_tilt_sample_duration_s)
        measured_tilt = imu_reader.measure_tilt(
            duration_s=config.camera_tilt_sample_duration_s)

        if measured_tilt is not None:
            config.camera_tilt_measured_deg = measured_tilt
            log.info("Camera tilt measured: %.2f° (will use this for distance "
                     "estimation — config fallback was %.2f°)",
                     measured_tilt, config.camera_tilt_deg)
        else:
            log.warning("IMU tilt measurement failed.  "
                        "Falling back to config value: %.2f°",
                        config.camera_tilt_deg)
    else:
        if not enable_imu:
            log.info("IMU disabled — using configured tilt: %.2f°",
                     config.camera_tilt_deg)
        else:
            log.info("camera_tilt_from_imu = false — using configured "
                     "tilt: %.2f°", config.camera_tilt_deg)

    log.info("Effective camera tilt: %.2f° (source: %s)",
             config.effective_camera_tilt_deg, config.tilt_source)

    # ── Autopilot connection ──────────────────────────────────────────────────
    log.info("Connecting to autopilot: %s  baud=%d",
             config.connect_string, config.baud)
    drone = drone_lib.connect_device(config.connect_string, baud=config.baud)

    # ── Hardware checks (real flight only) ────────────────────────────────────
    if not config.virtual_mode:
        if not config.use_estimated_distance:
            # Rangefinder is required for sensor-based distance measurement.
            if drone.rangefinder.distance is None:
                log.info("Rangefinder not detected and use_estimated_distance "
                         "is false — cannot measure drop distance.  Exiting.")
                exit(99)
            log.info("Rangefinder pre-flight check: %.2f m",
                     pex03_utils.get_avg_distance_to_obj(2, drone))
        else:
            log.info("Using camera geometry for distance estimation — "
                     "no rangefinder check needed.")

    # ── Mission pre-flight check ──────────────────────────────────────────────
    drone.commands.download()
    time.sleep(1)
    log.info("Checking autopilot for a pre-loaded waypoint mission...")
    if drone.commands.count < 1:
        log.info("No mission found on the autopilot — exiting.")
        exit(99)

    # ── Arm and take off ──────────────────────────────────────────────────────
    drone_lib.arm_device(drone, log=log)
    drone_lib.device_takeoff(drone, config.takeoff_altitude_m, log=log)

    # ── Run the mission ───────────────────────────────────────────────────────
    try:
        drone_lib.change_device_mode(drone, "AUTO", log=log)

        drone_mission = DroneMission(device=drone, config=config)
        drone_mission.conduct_mission()

        log.info("Mission complete — disarming and disconnecting.")
        drone.armed = False
        drone.close()

        # Stop the IMU reader thread cleanly if it was running.
        if imu_reader is not None:
            imu_reader.stop()

        log.info("End of PEX 03.")

    except Exception:
        log.info("Unhandled exception: %s",
                 traceback.format_exception(*sys.exc_info()))
        if imu_reader is not None:
            imu_reader.stop()
        raise
