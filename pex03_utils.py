
import glob
import logging
import os
import shutil
import time
from pathlib import Path
import cv2

GRIPPER_OPEN = 1087
GRIPPER_CLOSED = 1940

# TODO: be sure to set this path to your team's name (instead of "cam")
#   - set to something that works on your text machine.
DEFAULT_LOG_PATH = '/media/usafa/data/pex02_mission/cam'


def write_log_entry(entry):
    logging.info(entry)


def write_frame(frm_num, frame, path):
    frm = "{:06d}".format(int(frm_num))
    cv2.imwrite(f"{path}/frm_{frm}.png", frame)


def estimate_ground_distance_m(
    *,
    camera_tilt_deg: float,
    camera_fov_v_deg: float,
    target_y_px: float,
    frame_height: int,
    altitude_m: float,
) -> float:
    """
    Estimate the horizontal ground distance to a target using camera geometry.

    Use this function when a rangefinder sensor is not available.  It replaces
    hardware distance measurement using only the camera's known field-of-view,
    the measured tilt angle, the target's vertical pixel position in the frame,
    and the drone's GPS altitude.

    How it works — the geometry
    ---------------------------
    The camera is tilted downward by camera_tilt_deg from horizontal.  Because
    of this tilt, looking straight along the camera's optical axis already
    points at a depression angle equal to the tilt.  A target that appears
    BELOW the frame centre will be at a steeper depression angle (farther into
    the ground) than one at the centre; a target ABOVE the centre would be at
    a shallower angle (closer to the horizon).

    The vertical field-of-view tells us how many degrees each pixel represents.
    Combining the tilt with the pixel offset from the frame centre gives the
    total depression angle to the target:

        depression = camera_tilt
                   + ((target_y / frame_height) - 0.5) * v_fov

    Once we have the depression angle and the drone's altitude, basic
    trigonometry gives the ground distance:

        ground_distance = altitude / tan(depression)

    Assumptions
    -----------
    * The ground below the target is approximately flat.
    * camera_tilt_deg is accurate — use the IMU-measured value when available.
    * GPS altitude has reasonable accuracy (< 1 m barometric drift is typical).
    * The target is not near the horizon (depression > ~1°).  Very shallow
      angles cause the tangent to approach zero and the estimate blows up.
      The function guards against this and returns -1.0 in that case.

    Parameters
    ----------
    camera_tilt_deg  : Camera tilt below horizontal in degrees.  Use
                       config.effective_camera_tilt_deg so the IMU-measured
                       value is used automatically when available.
    camera_fov_v_deg : Vertical field-of-view of the camera in degrees.
                       From config.camera_fov_v_deg.  D455 ≈ 58°.
    target_y_px      : Vertical pixel coordinate of the target's centre in the
                       frame (0 = top edge, frame_height = bottom edge).
    frame_height     : Camera frame height in pixels (e.g. 480).
    altitude_m       : Drone altitude above ground in metres.
                       Use drone.location.global_relative_frame.alt.

    Returns
    -------
    float
        Estimated horizontal ground distance to the target in metres.
        Returns -1.0 if the geometry is degenerate (altitude too low,
        depression angle too shallow, or target near or above the horizon).

    Example
    -------
        dist = pex03_utils.estimate_ground_distance_m(
            camera_tilt_deg  = config.effective_camera_tilt_deg,
            camera_fov_v_deg = config.camera_fov_v_deg,
            target_y_px      = center[1],
            frame_height     = obj_track.FRAME_HEIGHT,
            altitude_m       = location.alt,
        )
    """
    import math

    # Sanity guard: we need meaningful altitude to compute distance.
    if altitude_m < 0.5:
        return -1.0

    v_fov_rad  = math.radians(camera_fov_v_deg)
    tilt_rad   = math.radians(camera_tilt_deg)

    # Compute the fraction of the way down the frame the target sits.
    # 0.0 = top edge, 0.5 = centre, 1.0 = bottom edge.
    # Subtract 0.5 so targets above centre give a negative offset
    # (shallower depression) and targets below give a positive offset
    # (steeper depression, i.e. closer to the ground directly below).
    row_fraction = float(target_y_px) / max(1, int(frame_height))
    pixel_offset_rad = (row_fraction - 0.5) * v_fov_rad

    # Total depression angle below the horizontal plane.
    depression_rad = tilt_rad + pixel_offset_rad

    # Guard: if the depression angle is less than ~1°, the target is
    # near or above the horizon and tan(depression) → 0, making the
    # distance estimate meaninglessly large or negative.
    if depression_rad <= math.radians(1.0):
        return -1.0

    return altitude_m / math.tan(depression_rad)


def get_ground_distance(height, hypotenuse):
    import math

    # Assuming we know the distance to object from the air
    # (the hypotenuse), we can calculate the ground distance
    # by using the simple formula of:
    # d^2 = hypotenuse^2 - height^2

    return math.sqrt(hypotenuse ** 2 - height ** 2)


def calc_new_location(cur_lat, cur_lon, heading, meters):
    from geopy import distance
    from geopy import Point

    # given: cur_lat, cur_lon,
    #        heading = bearing in degrees,
    #        meters = distance in meters

    origin = Point(cur_lat, cur_lon)
    destination = distance.distance(
        kilometers=(meters * .001)).destination(origin, heading)

    return destination.latitude, destination.longitude


def get_avg_distance_to_obj(seconds, device, virtual_mode=False):
    if virtual_mode:
        return 35.0

    distance = device.rangefinder.distance
    i = 1

    if distance is None:
        return -1

    t_end = time.time() + seconds
    while time.time() < t_end:
        i += 1
        distance += device.rangefinder.distance

    return distance / i


def release_grip(drone, seconds=2):
    sec = 1

    while sec <= seconds:
        override_gripper_state(drone, state=GRIPPER_OPEN)
        time.sleep(1)
        sec += 1


def override_gripper_state(drone, state=GRIPPER_CLOSED, channel=7):
    drone.channels.overrides[f'{channel}'] = state


def backup_prev_experiment(path):
    if os.path.exists(path):
        if len(glob.glob(f'{path}/*')) > 0:
            time_stamp = time.time()
            shutil.move(os.path.normpath(path),
                        os.path.normpath(f'{path}_{time_stamp}'))

    Path(path).mkdir(parents=True, exist_ok=True)


def clear_path(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)
