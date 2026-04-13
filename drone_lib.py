import math
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil
import time
import traceback
import sys
import logging

# Some useful information on coordinates and reference frames can be found
# here: https://dronekit-python.readthedocs.io/en/latest/guide/copter/guided_mode.html#guided-mode-copter-velocity-control


def upload_new_flight_path(device, flight_data):
    """
    Adds new flight data to the device and prepends with a takeoff command (ignored if drone is already in flight).

    The function assumes vehicle.commands matches the vehicle mission state
    (you must have called download at least once in the session and after clearing the mission)
    """
    print("Flight path change!")
    cmds = device.commands

    change_device_mode(device, "GUIDED")

    print(" Clearing any existing commands...")
    cmds.clear()

    print(" Define/add new commands.")
    # Add new commands. The meaning/order of the parameters is documented in the Command class.

    # Add MAV_CMD_NAV_TAKEOFF command. This is ignored if the vehicle is already in the air.
    cmds.add(
        Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0,
                0, 0, 0, 0, 20))

    # Now, for each set of coordinates, add a new waypoint into the command set
    for waypoint in flight_data:
        print(f"alt: {waypoint[2]*.10}")
        cmds.add(
            Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 0, 0, 0,
                    0, 0, 0, waypoint[0], waypoint[1], waypoint[2]*.10))

    print(" Upload new commands to vehicle")
    cmds.upload()
    cmds.wait_ready()

    change_device_mode(device, "AUTO")
    

def display_rover_state(connection):
    print(f"Battery: {connection.battery}")
    print(f"Last heartbeat: {connection.last_heartbeat}")
    print(f"Attitude: {connection.attitude}")
    print(f"Heading: {connection.heading}")
    print(f"Ground speed:{connection.groundspeed}")
    print(f"Velocity: {connection.velocity}")
    print(f"Steering rc: {connection.channels[1]}")
    print(f"Throttle rc: {connection.channels[3]}")


def display_vehicle_state(connection):
    # vehicle is an instance of the Vehicle class
    print(f"Autopilot capabilities (supports ftp): {connection.capabilities.ftp}")
    print(f"Global location: {connection.location.global_frame}")
    print(f"Global location (relative altitude): {connection.location.global_relative_frame}")
    print(f"Local location: {connection.location.local_frame}")  # NED
    print(f"Attitude: {connection.attitude}")
    print(f"Velocity: {connection.velocity}")
    print(f"GPS: {connection.gps_0}")
    print(f"Ground speed:{connection.groundspeed}")
    print(f"Airspeed: {connection.airspeed}")
    print(f"Gimbal status: {connection.gimbal}")
    print(f"Battery: {connection.battery}")
    print(f"EKF OK?: {connection.ekf_ok}")
    print(f"Last heartbeat: {connection.last_heartbeat}")
    print(f"Rangefinder: {connection.rangefinder}")
    print(f"Rangefinder distance: {connection.rangefinder.distance}")
    print(f"Rangefinder voltage: {connection.rangefinder.voltage}")
    print(f"Heading: {connection.heading}")
    print(f"Is Armable?: {connection.is_armable}")
    print(f"System status: {connection.system_status.state}")
    print(f"Mode: {connection.mode.name}")  # settable
    print(f"Armed: {connection.armed}")  # settable
    

def get_short_distance_meters(location_1, location_2):
    """
    Returns the ground distance in meters between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    d_lat = location_2.lat - location_1.lat
    d_long = location_2.lon - location_1.lon
    return math.sqrt((d_lat * d_lat) + (d_long * d_long)) * 1.113195e5


def device_relative_distance_from_point(device, lat, lon, alt):
    """
    Gets distance in meters to the current waypoint.

    """

    cur_location = device.location.global_relative_frame
    target_location = LocationGlobalRelative(lat, lon, alt)
    distance = get_short_distance_meters(cur_location, target_location)

    return distance

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned Location has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon,original_location.alt)


def distance_to_current_waypoint(vehicle):
    """
    Gets distance in metres to the current waypoint. 
    It returns None for the first waypoint (Home location).
    """
    nextwaypoint = vehicle.commands.next
    if nextwaypoint==0:
        return None
    
    missionitem=vehicle.commands[nextwaypoint-1] #commands are zero indexed
    lat = missionitem.x
    lon = missionitem.y
    alt = missionitem.z
    targetWaypointLocation = LocationGlobalRelative(lat,lon,alt)
    distancetopoint = get_short_distance_meters(vehicle.location.global_frame, targetWaypointLocation)
    return distancetopoint


def log_activity(msg, log=None, level=logging.INFO):
    """
    Log a message via the provided logger, or the root logger if none given.

    The unconditional ``print()`` that was here previously was removed —
    it duplicated every log line to stdout regardless of the configured log
    level, interfering with any operator using a clean console.  Use
    ``--debug`` and a log handler with a StreamHandler if stdout output is
    needed.

    Parameters
    ----------
    msg   : Message string.
    log   : Optional logger instance.  Uses root logger when None.
    level : Log level (default: logging.INFO).
    """
    logger = log if log is not None else logging.getLogger(__name__)
    logger.log(level, msg)


def small_move_up(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone upwards by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """

    # Note: up is negative
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, 0, 0, -velocity, duration)
    move_local(device, 0, 0, velocity, duration, log, cube=cube)


def small_move_down(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone upwards by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """

    # Note: down is positive
    velocity = abs(velocity)

    # send_body_frame_velocities(device, 0, 0, velocity, duration)
    move_local(device, 0, 0, velocity, duration, log, cube=cube)


def small_move_forward(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone forward by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = abs(velocity)

    # send_body_frame_velocities(device, velocity, 0, 0.0, duration)
    move_local(device, velocity, 0, 0.0, duration, log, cube=cube)


def small_move_back(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone backward by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, velocity, 0, 0.0, duration)
    move_local(device, velocity, 0, 0.0, duration, log, cube=cube)


def small_move_right(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone right by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = abs(velocity)

    # send_body_frame_velocities(device, 0, velocity, 0.0, duration)
    move_local(device, 0, velocity, 0.0, duration, log, cube=cube)


def small_move_left(device, velocity=0.5, duration=1, log=None, cube=True):
    """
    Moves drone left by a small amount.
    To move by a larger amount, set velocity and/or duration to something bigger.
    """
    velocity = -abs(velocity)

    # send_body_frame_velocities(device, 0, velocity, 0.0, duration)
    move_local(device, 0, velocity, 0.0, duration, log, cube=cube)


def move_local(device, x, y, z, duration=1, log=None, cube=True):
    """
    Issue a body-frame velocity command and block for ``duration`` seconds.

    This is a convenience wrapper used by the ``small_move_*`` helpers.
    Because it calls ``send_body_frame_velocities()`` (the blocking plural
    form), it holds the calling thread for ``duration`` seconds — one second
    by default.

    For use in pex03.py's centering loop this is intentional: the drone
    physically moves for ``duration`` seconds before the loop re-evaluates
    the target's pixel position.  Students should account for this when
    choosing velocity magnitudes — at 0.5 m/s with duration=1 the drone
    moves ~0.5 m before the next camera frame is acted on.

    If you need a single fire-and-forget velocity command that does NOT
    block (for example, to send repeated commands at camera frame rate),
    use ``send_body_frame_velocity()`` (singular, non-blocking) directly.
    """
    log_activity(
        f"Local move with velocities {x},{y},{z} for {duration} seconds.",
        log,
        level=logging.DEBUG,
    )
    send_body_frame_velocities(device, x, y, z, duration)
    return

    '''if cube:
        send_body_frame_velocities(device, x, y, z, duration)
    else:
        send_global_frame_velocities(device, -y, -x, z, duration)'''


def condition_yaw(device, heading, relative=False, speed_dps=0, direction=1, log=None):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.

    By default the yaw of the vehicle will follow the direction of travel. After setting
    the yaw using this function there is no way to return to the default yaw "follow direction
    of travel" behavior (https://github.com/diydrones/ardupilot/issues/2427)

    For more information see:
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw

    Parameters
    ----------
    device     : DroneKit Vehicle instance.
    heading    : Angle in degrees.  For absolute mode this is the target bearing
                 (0 = North, CW).  For relative mode this is the magnitude of
                 the rotation (always positive; ``direction`` controls CW vs CCW).
    relative   : If True, ``heading`` is treated as a delta from the current yaw.
                 If False (default), it is an absolute compass bearing.
    speed_dps  : Angular rate hint in degrees per second.  0 lets the autopilot
                 choose its own rate (the original default behaviour).
    direction  : +1 for clockwise, -1 for counter-clockwise.
                 Only meaningful when ``relative=True``.  When ``relative=False``
                 the autopilot picks the shortest turn direction.
    log        : Optional logger instance passed to ``log_activity()``.
    """

    if log is not None:
        log.debug(
            "Yaw to %.1f degrees (relative=%s, speed=%.1f deg/s, dir=%+d).",
            heading, relative, speed_dps, direction,
        )

    if relative:
        is_relative = 1  # yaw relative to direction of travel
    else:
        is_relative = 0  # yaw is an absolute angle

    # create the CONDITION_YAW command using command_long_encode()
    msg = device.message_factory.command_long_encode(
        0, 0,                                              # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,             # command
        0,                                                 # confirmation
        abs(float(heading)),                               # param 1, yaw in degrees
        float(speed_dps),                                  # param 2, yaw speed deg/s
        int(direction),                                    # param 3, direction -1 ccw, 1 cw
        is_relative,                                       # param 4, relative offset 1, absolute angle 0
        0, 0, 0)                                           # param 5 ~ 7 not used

    # send command to vehicle
    device.send_mavlink(msg)


def send_global_frame_velocities(device, velocity_x, velocity_y, velocity_z, duration=2):
    """
    Send a NED (North-East-Down) global-frame velocity command, repeating it
    once per second for ``duration`` seconds.

    Uses MAV_FRAME_LOCAL_NED with set_position_target_local_ned_encode, which
    is the correct pairing for NED velocity commands.  The previous version
    used MAV_FRAME_BODY_NED (body-relative, deprecated) inside the global-int
    encoder — both choices were wrong for a world-frame NED velocity.

    .. deprecated::
        This function **blocks the calling thread** for ``duration`` seconds.
        It is retained for manual test scripts only — do not call from the
        control loop.  See ``send_body_frame_velocity`` for the non-blocking
        control-loop equivalent.

    Parameters
    ----------
    device     : DroneKit Vehicle instance.
    velocity_x : North velocity in m/s.
    velocity_y : East velocity in m/s.
    velocity_z : Down velocity in m/s (positive = descend).
    duration   : Seconds to repeat the command (one send per second).
    """
    msg = device.message_factory.set_position_target_local_ned_encode(
        0,                                         # time_boot_ms (not used)
        0, 0,                                      # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,       # world-frame NED axes
        0b001111000111,                            # type_mask (velocity only)
        0, 0, 0,                                   # x, y, z position (ignored)
        velocity_x,                                # vx — North m/s
        velocity_y,                                # vy — East m/s
        velocity_z,                                # vz — Down m/s
        0, 0, 0,                                   # ax, ay, az (ignored)
        0, 0,                                      # yaw, yaw_rate (ignored)
    )

    for _ in range(duration):
        device.send_mavlink(msg)
        time.sleep(1)


def send_body_frame_velocities(device, forward, right, velocity_z, duration=2):
    """
    Send a body-frame velocity command, repeating it once per second for
    ``duration`` seconds.

    .. deprecated::
        This function **blocks the calling thread** for ``duration`` seconds
        (one ``time.sleep(1)`` per iteration).  It must never be called from
        the sentinel perception-control loop or any other time-sensitive
        context.

        Use ``send_body_frame_velocity`` (singular, non-blocking) for all
        control-loop velocity commands.  This function is retained only for
        manual test scripts (e.g. ``drone_guided_test_console.py``) where
        blocking behaviour is acceptable.

    Parameters
    ----------
    device     : DroneKit Vehicle instance.
    forward    : Forward velocity in m/s (+X body frame).
    right      : Rightward velocity in m/s (+Y body frame).
    velocity_z : Downward velocity in m/s (+Z body frame, positive = descend).
    duration   : Number of seconds to repeat the command (one send per second).
    """

    msg = device.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # body-relative axes (confirmed on ArduCopter 4.0.4)
        0b001111000111,  # type_mask (only speeds enabled)
        0,  # x position (ignored)
        0,  # y position (ignored)
        0,  # z position (ignored)
        forward,    # X velocity in body frame in m/s
        right,      # Y velocity in body frame in m/s
        velocity_z, # Z velocity in body frame in m/s
        0, 0, 0,    # afx, afy, afz acceleration (ignored)
        0, 0)       # yaw, yaw_rate (ignored)

    # send command to vehicle on 1 Hz cycle
    for x in range(0, duration):
        device.send_mavlink(msg)
        time.sleep(1)


def send_body_frame_velocity(device, forward, right, down, log=None):
    """
    Send a single body-frame velocity command without blocking.

    Unlike ``send_body_frame_velocities`` (plural), this function transmits
    exactly **one** MAVLink message and returns immediately.  It is designed
    for use inside a high-frequency control loop (e.g. the Sentinel
    perception–control loop) where the caller is responsible for re-sending
    the velocity at its own update rate.

    The body frame axes are:
        forward : +X in body frame  (positive = nose direction)
        right   : +Y in body frame  (positive = starboard)
        down    : +Z in body frame  (positive = toward earth, NED convention)

    Parameters
    ----------
    device  : DroneKit Vehicle instance.
    forward : Forward velocity in m/s.  Positive = forward (nose direction).
    right   : Rightward velocity in m/s.  Positive = starboard.
    down    : Downward velocity in m/s.  Positive = toward earth.
              Set to 0.0 to hold current altitude.
    log     : Optional logger instance.  Logs at DEBUG level to avoid
              flooding the log at control-loop rates (~4 Hz per axis).

    Notes
    -----
    The type_mask (0b0000_1100_0011_1000_0111 = 0x0DC7) ignores position and
    acceleration components and suppresses yaw/yaw-rate so the autopilot
    does not fight with any concurrent yaw commands from condition_yaw().

    Uses MAV_FRAME_BODY_NED (8).  MAV_FRAME_BODY_FRD (12) was tested and
    caused follow control to stop working on ArduCopter 4.0.4.  Do not
    change this without re-testing on the target firmware version.
    """

    if log is not None:
        log.debug(
            "Body velocity: fwd=%.2f right=%.2f down=%.2f m/s (single shot).",
            forward, right, down,
        )

    # type_mask: ignore position (bits 0-2), USE velocity (bits 3-5 clear),
    # ignore acceleration (bits 6-8), ignore yaw/yaw_rate (bits 10-11).
    # = 1+2+4 + 64+128+256 + 1024+2048 = 3527 = 0x0DC7
    type_mask = 0x0DC7

    msg = device.message_factory.set_position_target_local_ned_encode(
        0,                                          # time_boot_ms (not used)
        0, 0,                                       # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,         # frame — body-relative axes
        type_mask,
        0.0, 0.0, 0.0,                             # x, y, z position (ignored)
        float(forward), float(right), float(down),  # vx, vy, vz velocity (body frame)
        0.0, 0.0, 0.0,                             # ax, ay, az (ignored)
        0.0, 0.0,                                   # yaw, yaw_rate (ignored)
    )
    device.send_mavlink(msg)


def connect_device(s_connection, baud=115200, log=None):
    log_activity(f"Connecting to device {s_connection} with baud rate {baud}...", log)
    device = connect(s_connection, wait_ready=True, baud=baud)
    log_activity("Device connected.", log)
    log_activity(f"Device version: {device.version}", log)
    return device


def arm_device(device, log=None, n_reps=10, mode="GUIDED", skip_monitor=False):
    log_activity("Arming device...", log)
    wait = 1

    # "GUIDED" mode sets drone to listen
    # for our commands that tell it what to do...
    device.mode = VehicleMode(mode)
    while device.mode.name != mode:
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            break

        log_activity(f"Switching to {mode} mode...", log)
        time.sleep(2)
        wait += 1
    wait = 1
    device.armed = True

    while not device.armed:
        if wait > n_reps:
            log_activity("arm timeout.", log)
            break
        log_activity("Waiting for arm...", log)
        time.sleep(2)

    log_activity(f"Device armed: {device.armed}.", log)

    return device.armed


def change_device_mode(device, mode, n_reps=10, log=None):
    wait = 0
    log_activity(f"Changing device mode from {device.mode} to {mode}...", log)

    device.mode = VehicleMode(mode)

    while device.mode.name != mode:
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            return False
        device.mode = VehicleMode(mode)
        time.sleep(.5)
        wait += 1

    log_activity(f"Device mode = {device.mode}.", log)


def device_takeoff(device, altitude, log=None, skip_monitor=False):
    log_activity("Device takeoff...", log)
    device.mode = VehicleMode("GUIDED")
    time.sleep(.5)
    device.simple_takeoff(altitude)
    device.airspeed = 3
    
    if skip_monitor:
        return
    
    while device.armed \
            and device.mode.name == "GUIDED":
        log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
        if device.location.global_relative_frame.alt >= (altitude * .90):
            break
        time.sleep(.5)

    time.sleep(2)


def device_land(device, log=None):
    log_activity("Device land...", log)
    device.mode = VehicleMode("LAND")
    time.sleep(.5)
    while device.armed \
            and device.mode.name == "LAND":
        log_activity(f"Current altitude: {device.location.global_relative_frame.alt}", log)
        if device.location.global_relative_frame.alt <= 1:
            log_activity("Device has landed.", log)
            break
        time.sleep(.1)

    # disarm the device
    time.sleep(4)
    device.armed = False


def execute_flight_plan(device, n_reps=10, wait=1, log=None):
    if device.commands.count == 0:
        log_activity("No flight plan to execute.", log)
        return False

    log_activity("Executing flight plan...", log)

    # Reset mission set to first (0) waypoint
    device.commands.next = 0

    # Set mode to AUTO to start mission
    device.mode = VehicleMode("AUTO")
    time.sleep(.5)
    while device.mode.name != "AUTO":
        if wait > n_reps:
            log_activity("mode change timeout.", log)
            return False

        log_activity("Switching to AUTO mode...", log)
        time.sleep(1)
        wait += 1
    return True


def goto_point(device, lat, lon, speed, alt, log=None, timeout_s=60):
    """
    Fly to the specified GPS coordinate and block until arrival or timeout.

    Arrival is determined by ground distance (< 1.5 m) and relative altitude
    matching (within 10 % of target).  The old implementation used
    lat/lon percentage ratios which broke for negative coordinates and for
    targets near zero — both common in real operations.

    This function is intentionally blocking — it does not return until the
    drone arrives or the timeout expires.  This makes the mission flow in
    pex03.py easy to read (go here, THEN do the next thing) without
    requiring the student to manage asynchronous navigation state.

    Parameters
    ----------
    device    : DroneKit Vehicle.
    lat       : Target latitude in decimal degrees.
    lon       : Target longitude in decimal degrees.
    speed     : Airspeed in m/s.
    alt       : Target altitude in metres (relative to home).
    log       : Optional logger.
    timeout_s : Maximum seconds to wait for arrival before giving up.
                Default is 60 seconds.  If the drone has not reached the
                target within this time (e.g. due to wind, GPS oscillation,
                or an unreachable point), the function returns False so the
                caller can decide how to handle the failure.
                Set to None to wait indefinitely (original behaviour —
                not recommended for flight).

    Returns
    -------
    bool
        True  — drone arrived within the distance/altitude tolerances.
        False — timeout expired before arrival, or drone was disarmed /
                mode-changed out of GUIDED before reaching the target.
    """
    log_activity(f"Goto point: {lat}, {lon}, {speed}, {alt}...", log)

    device.airspeed = speed
    point = LocationGlobalRelative(lat, lon, alt)
    device.simple_goto(point)

    deadline = (time.time() + timeout_s) if timeout_s is not None else None

    while device.armed and device.mode.name == "GUIDED":
        try:
            # Check timeout before doing anything else so we exit promptly
            # even if the GPS/telemetry poll below is slow.
            if deadline is not None and time.time() > deadline:
                log_activity(
                    f"goto_point: timeout ({timeout_s} s) reached before arrival "
                    f"at ({lat}, {lon}).  Returning False.",
                    log,
                )
                return False

            cur_alt = device.location.global_relative_frame.alt
            distance = device_relative_distance_from_point(device, lat, lon, alt)

            log_activity(
                f"Current alt: {cur_alt:.1f} m  Ground distance: {distance:.1f} m",
                log,
            )

            # Altitude match: within 10 % of target (avoids divide-by-zero
            # for alt=0 and works correctly for any altitude value).
            alt_ok = abs(cur_alt - alt) <= max(1.0, abs(alt) * 0.10)

            if alt_ok and distance < 1.5:
                log_activity(f"goto_point: arrived at ({lat}, {lon}).", log)
                return True  # close enough — may never hit exactly
            time.sleep(1)
        except Exception as e:
            log_activity(
                f"Error on goto: {traceback.format_exception(*sys.exc_info())}", log
            )
            raise

    # Loop exited because drone was disarmed or mode changed (e.g. GCS RTL).
    log_activity(
        f"goto_point: exited early — drone mode={device.mode.name}, "
        f"armed={device.armed}.",
        log,
    )
    return False


def goto_point2(device, lat, lon, speed, alt, log=None, wait_secs=None):
    """
    .. deprecated::
        Use ``goto_point()`` instead.  This function has a known bug in its
        altitude-match logic: when ``cur_alt > alt`` it computes
        ``alt / cur_alt`` (always ≤ 1.0) rather than an absolute difference,
        so the altitude-OK check never triggers correctly when descending to a
        lower target altitude.  ``goto_point()`` fixes this and adds a
        ``timeout_s`` parameter for convergence safety.
        This function is retained only to avoid breaking any existing test
        scripts that reference it by name.
    """
    log_activity(f"Goto point: {lat}, {lon}, {speed}, {alt}...", log)

    # set the default travel speed
    device.airspeed = speed

    point = LocationGlobalRelative(lat, lon, alt)

    device.simple_goto(point)

    while device.armed \
            and device.mode.name == "GUIDED":
        try:
            cur_alt = device.location.global_relative_frame.alt
            log_activity(f"Current altitude: {cur_alt}", log)
            log_activity(f"Current lat: {device.location.global_relative_frame.lat}", log)
            log_activity(f"Current lon: {device.location.global_relative_frame.lon}", log)

            if cur_alt <= alt:
                alt_percent = cur_alt / alt
            else:
                alt_percent = alt / cur_alt

            distance = device_relative_distance_from_point(device, lat, lon, alt)

            log_activity(f"Ground distance to destination: {distance}; diff in altitude: {alt_percent} ", log)

            if wait_secs is None:
                # Monitor our progress...
                if (.985 <= alt_percent <= 1.1) \
                        and distance < 1.2:  # (less than 1.2 meters in distance)
                    break  # close enough - may never be perfectly on the mark
            else:
                # Instead of monitoring progress towards destination,
                # just wait the allotted time and then get out.
                time.sleep(wait_secs)
                break

            time.sleep(1.0)
        except Exception as e:
            log_activity(f"Error on goto: {traceback.format_exception(*sys.exc_info())}", log)
            raise


def return_to_launch(device, log=None):
    """
    Command the drone to return to its launch point and land.

    Switches the autopilot to RTL mode and returns immediately — ArduPilot
    manages the full return-and-land sequence autonomously without any
    further commands from this application.  There is no need to monitor
    the descent here; the GCS and autopilot failsafes handle it.
    """
    log_activity("Device returning to launch...", log)
    device.mode = VehicleMode("RTL")
    time.sleep(.5)
