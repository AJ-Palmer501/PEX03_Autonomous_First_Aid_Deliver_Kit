# PEX 03 — Autonomous First Aid Kit Delivery ⛑️

> **CS 472: Autonomous Systems Integration | Spring 2026**
> *Programming Exercise 03 · 300 points · Due M40 at 10:30*

---

## Table of Contents

1. [Background and Context](#background-and-context)
2. [Project Description](#project-description)
3. [What You Will Learn](#what-you-will-learn)
4. [The Mission](#the-mission)
5. [The State Machine](#the-state-machine)
6. [Why the States Are Designed This Way](#why-the-states-are-designed-this-way)
7. [Repository Structure](#repository-structure)
8. [Program Architecture](#program-architecture)
9. [Configuration](#configuration)
10. [Provided Modules — API Reference](#provided-modules--api-reference)
11. [What You Must Build](#what-you-must-build)
12. [Engineering Challenges](#engineering-challenges)
13. [Development Roadmap](#development-roadmap)
14. [SITL Testing Guide](#sitl-testing-guide)
15. [Tunable Parameters](#tunable-parameters)
16. [Hardware Reference](#hardware-reference)
17. [Grading](#grading)

---

## Background and Context

### The Real-World Problem

Search and rescue operations in austere environments present one of the most difficult
logistics challenges in emergency response. When a person is injured in a remote
area — a downed pilot, a hiker with a broken leg, a soldier separated from their unit —
the time between injury and first medical intervention is often the difference between
survival and death. The "golden hour" is not a metaphor.

The fundamental problem is access. Terrain, enemy activity, weather, or simply distance
may prevent first responders from reaching the casualty quickly enough. A ground team
that takes 45 minutes to reach someone who needed a tourniquet in the first ten minutes
has arrived too late regardless of their skill.

Unmanned aerial systems (UAS) can fly directly to a GPS coordinate at speeds no ground
vehicle can match, over terrain no vehicle can navigate, without putting another person
at risk. A drone carrying a first-aid kit can reach a downed pilot in minutes and
deliver critical supplies — a tourniquet, a pressure dressing, a space blanket — that
buy time until a full rescue team arrives.

The catch is autonomy. An operator controlling a drone manually over a search area must
maintain visual or video contact, manage the flight, watch for obstacles, and identify
the casualty simultaneously. That cognitive load degrades performance rapidly and
scales poorly as the search area grows. A system that can autonomously search,
detect, confirm, and deliver — with a human only monitoring — multiplies the
operator's reach dramatically.

### Why This Exercise Exists

PEX 03 places you at the intersection of every major topic in this course. You are not
implementing a single algorithm or controlling a single subsystem. You are integrating
computer vision, autonomous flight control, sensor fusion, geospatial mathematics, and
state-machine design into one working system that must perform reliably under
real-world conditions.

This is also an exercise in engineering judgement. Many of the parameters you will
tune have no single correct answer — the right confidence threshold, the right velocity,
the right acceptance radius all depend on the specific hardware, environment, and
risk tolerance of the mission. You will be expected to derive your choices from first
principles and defend them.

### The Target

A dressed mannequin standing in an open field represents the downed pilot. Your drone
does not know the mannequin's location in advance. It must find them autonomously
during its search flight, confirm the sighting, maneuvere overhead, compute where
to drop the package, and lower it gently enough that the contents survive.

The payload contains eggs. This is intentional. Eggs break under the same forces that
would render a medical kit useless. Packaging a fragile payload, controlling descent
rates, and managing tether swing during flight are not incidental details — they are
core engineering requirements.

---

## Project Description

PEX 03 is the capstone programming exercise in CS 472: Autonomous Systems Integration.
It asks you to apply every skill developed throughout the course — computer vision,
MAVLink flight control, geospatial mathematics, IMU sensor fusion, and state-machine
design — and combine them into a single working autonomous system.

The exercise integrates five areas simultaneously:

- **Computer vision** — A YOLOv8 model trained on aerial VisDrone imagery detects
  people from altitude. An HSV color-histogram identity tracker then maintains a
  lock across frames without relying on pixel position — making it robust to the
  camera motion caused by the drone's own corrections.
- **Autonomous flight control** — DroneKit and MAVLink commands arm, navigate, and
  maneuvere the drone through every mission phase, including real-time body-frame
  velocity corrections during the centering loop.
- **Sensor fusion and geospatial mathematics** — The camera's field-of-view geometry
  combined with the drone's GPS altitude and the IMU-measured tilt angle are used to
  estimate the horizontal ground distance to the target without a rangefinder. That
  distance is then projected to a GPS drop coordinate via bearing projection.
- **IMU integration** — The camera's built-in accelerometer and gyroscope measure the
  actual camera tilt before takeoff. This replaces a manually configured constant with
  a measured value, improving distance estimation accuracy.
- **State machine design** — A five-state machine governs all mission logic, enforces
  correct transition ordering, and keeps the code readable and debuggable. All
  transitions are visible in one place.

Your primary deliverable is **`pex03.py`**, which contains the `DroneMission` class
and its `conduct_mission()` event loop. A skeleton with detailed comments and `# TODO`
markers is provided; you fill in the implementation.

---

## What You Will Learn

### Autonomous Systems Architecture

- How to design a state machine for a safety-critical autonomous system where an
  incorrect state transition has immediate physical consequences.
- Why all state-transition decisions belong in a single method (`conduct_mission()`)
  rather than scattered across helper functions — and what goes wrong when they are not.
- When a blocking sequential subroutine is the correct architectural choice over a
  non-blocking event loop, and how to reason about what the drone is doing during
  a blocking call.
- How a clean separation between *what to do next* (state machine) and *how to do it*
  (helper methods) makes a complex system testable and maintainable.

### Computer Vision in a Real Deployment

- Why appearance-based identity tracking (HSV color histograms) is fundamentally more
  robust under camera ego-motion than position-based tracking (CSRT, Kalman filters)
  — and what "ego-motion" means in the context of a drone correction loop.
- How a two-phase detect-then-track pipeline works: heavyweight YOLO detection to
  acquire an initial lock, followed by lightweight histogram matching to maintain it.
- The difference between a non-committing detection scan (`check_for_initial_target`)
  and committing the tracker to a target (`set_object_to_track`) — and why the drone
  must fly back to the sighting location before doing the latter.
- Real-world model limitations: confidence score distributions, orientation sensitivity,
  altitude-dependent scale, and the cost of false positives in terms of flight time.

### Sensor Fusion and Geospatial Reasoning

- How to combine camera geometry (field of view, tilt angle, pixel position of target)
  with GPS altitude to estimate the horizontal ground distance to a target — without
  any dedicated rangefinder hardware.
- Why IMU-measured tilt is more accurate than a manually configured constant, and how
  to obtain it before motors are running.
- Why time-averaged measurements substantially outperform single samples for noisy
  sensors, and how to design a sampling strategy around a blocking delay.

### Drone Control via DroneKit and MAVLink

- The difference between AUTO mode (autopilot executes its pre-loaded waypoint plan)
  and GUIDED mode (your code issues navigation commands directly), and what each
  implies for the flight plan state.
- Why `MIS_RESTART = 0` on the autopilot is a critical pre-flight configuration — and
  what happens to the waypoint mission if it is not set correctly.
- Body-frame velocity commands: the coordinate system, sign conventions, and why the
  blocking duration of `small_move_*` functions directly affects centering loop
  performance.
- Why `goto_point()` needs a timeout and what the correct response to a timeout is
  (hint: not "abort everything").

### Engineering Judgement and Quantitative Design

- How to derive pixel-level thresholds from physical requirements: given camera FOV,
  frame resolution, and mission altitude, how many pixels correspond to 1 meter of
  horizontal error?
- How to reason about a proportional velocity controller without a PID: why does 1 Hz
  effective control rate (caused by 1-second blocking calls) constrain the maximum
  safe correction velocity?
- How to justify a "safe delivery radius" from first principles: package mass, descent
  velocity, tether length, and acceptable landing error all factor in.
- How to decide what to do when a subsystem returns an out-of-range value or a timeout
  occurs — and why "continue with a logged warning" is sometimes the right answer.

### Testing and Debugging Embedded Systems

- Incremental integration: camera alone → SITL → integrated → live. Each stage
  validates the foundation for the next.
- How to use annotated on-disk frame logs to reconstruct what the drone was seeing at
  each decision point after a mission.
- Recognising which parts of an autonomous system can and cannot be simulated, and
  designing your development strategy accordingly.

---

## The Mission

A downed pilot is located somewhere within a defined search airspace. They are injured
and cannot self-extract. Your drone carries a first-aid package suspended on a 3-meter
(10-foot) tether cable with a servo-controlled release latch.

**The following must happen autonomously and in order:**

1. Take off and begin the pre-loaded search flight plan in AUTO mode.
2. Scan each camera frame for a person on the ground using YOLOv8.
3. When a high-confidence detection occurs, record the drone's exact GPS position,
   altitude, and heading. Pause the flight plan.
4. Fly back to those coordinates. Restore the original heading. Wait for the drone
   to stabilize. Run detection again to confirm the sighting.
5. If confirmed: lock the color-histogram tracker onto the target's bounding box.
   Issue proportional velocity corrections until the target is centered in the frame.
6. Estimate the horizontal distance to the target using camera geometry and the IMU-
   measured tilt angle. Compute the GPS drop coordinate. Fly there.
7. Fast descent to an intermediate altitude, then slow/fine descent as the tether
   approaches full extension. Open the servo latch to release the package. Return home.

**The payload contains eggs. They must not break.**

---

## The State Machine

The mission is governed by five states. The machine always starts in SEEK and always
ends in RTL. The current state is logged at the front of every log line and displayed
on the HUD.

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     MISSION STATE MACHINE                            │
  │                                                                      │
  │  ┌────────┐  candidate    ┌─────────┐  confirmed   ┌────────────┐   │
  │  │  SEEK  │ ────────────► │ CONFIRM │ ───────────► │   TARGET   │   │
  │  │(AUTO)  │ ◄──────────── │(GUIDED) │              │  (GUIDED)  │   │
  │  └────────┘  not          └─────────┘              └─────┬──────┘   │
  │              confirmed                                    │          │
  │                                                    centered on target │
  │                                                           ▼          │
  │                                                    ┌────────────┐   │
  │                                                    │  DELIVER   │   │
  │                                                    │  (GUIDED)  │   │
  │                                                    └─────┬──────┘   │
  │                                                          │          │
  │                                                   package released  │
  │                                                          ▼          │
  │                                                    ┌────────────┐   │
  │                                                    │    RTL     │   │
  │                                                    │(return home)   │
  │                                                    └────────────┘   │
  └──────────────────────────────────────────────────────────────────────┘
```

| State | Mode | What runs | How you exit |
|---|---|---|---|
| **SEEK** | AUTO | `check_for_initial_target()` every N frames | confidence > threshold → CONFIRM |
| **CONFIRM** | GUIDED | Fly back → stabilize → detect again | tracker confirms → TARGET; attempts exhausted → SEEK |
| **TARGET** | GUIDED | `track_with_confirm()` + `adjust_to_target_center()` | centered → DELIVER; tracker lost → SEEK |
| **DELIVER** | GUIDED | Geometry → navigate → descend → release | package released → RTL |
| **RTL** | RTL | Main loop exits | Drone lands autonomously |

### Design Rules

**`conduct_mission()` owns all state transitions.** Every `self.mission_mode =` assignment
lives in this one method. No transitions are hidden inside helper methods. You can read
the mission flow from top to bottom without leaving that function.

**Helper methods do work and return results.** `adjust_to_target_center()` returns `True`
when centered, `False` while still adjusting. `deliver_package()` returns when the
delivery sequence is complete. The *transition* decision is always made by the caller.

**CONFIRM and DELIVER are intentionally blocking.** While they run, the camera loop is
suspended. This is the right architecture for a sequential physical process where each
step must complete before the next begins.

---

## Why the States Are Designed This Way

### Why CONFIRM must physically fly back

By the time YOLO detects a person and Python processes the result, the drone may have
travelled 20–50 m past the target. The detection came from a camera frame captured at
the drone's *old* position. Confirming from the current drone position would fail because
the target is now behind or outside the field of view.

The drone returns to the exact lat/lon/alt of the original sighting and restores its
original heading. The camera's field of view is fixed to the drone body — the wrong
heading means the camera is pointing the wrong direction even at the correct GPS
coordinates.

### Why the histogram tracker does not initialize at first sight

`set_object_to_track()` builds a color signature (HSV histogram) from the target's
bounding box crop. That signature needs:
- A stationary drone (no motion blur, no camera shake)
- The target clearly visible in the current frame
- An accurate, confirmed bounding box from a fresh YOLO pass

At first sight in SEEK the drone is cruising at speed and has likely already passed the
target. None of these conditions hold. Initialising the tracker in CONFIRM, after the
drone has returned to the sighting location and stabilized, gives the tracker the
ideal starting conditions and the most accurate initial signature.

### Why HSV histogram tracking instead of CSRT

The previous version of this project used an OpenCV CSRT tracker. CSRT predicts each
object's next position by assuming the camera is roughly stationary. When the drone
yaws or translates to centre on the target, every pixel in the frame shifts
simultaneously — CSRT's predictor saw this as the target disappearing, so tracking
failed exactly when the drone was trying to work. This is not a tuning problem; it is
a fundamental design mismatch between a static-camera tracker and a moving drone.

The histogram tracker matches targets by their clothing color, not their pixel
position. A drone pan that shifts every bounding box by 30 pixels does not change
a person's blue jacket. Histogram matching is inherently immune to camera ego-motion.

### Why camera geometry instead of a rangefinder

An Intel RealSense in depth mode provides the slant distance to the target, which
combined with altitude gives ground distance via Pythagoras. However, the depth sensor
requires USB 3.0 bandwidth and adds a hardware dependency that can fail.

Camera geometry uses the target's *pixel row* in the frame, the camera's vertical
field of view, the IMU-measured tilt angle, and the drone's GPS altitude to estimate
ground distance with no additional sensor. At the mission altitudes used in this
exercise, the geometric estimate is accurate to within 1–2 meters — sufficient for
a delivery within the 3-meter safe radius.

The configuration file (`pex03_config.json`) lets you switch between both methods with
one setting. Teams with a working rangefinder may use it; teams without can rely on
the geometric estimate.

---

## Repository Structure

```
student_pex03_oop/
│
├── pex03.py                        ← YOUR FILE — fill in every # TODO
├── pex03_config.json               ← Configuration file — change values here,
│                                       not in Python source code
│
├── mission_config.py               ← PROVIDED — loads pex03_config.json into
│                                       a typed MissionConfig object
│
├── object_tracking_y8_histo.py     ← PROVIDED — YOLOv8 detection, HSV histogram
│                                       identity tracking, camera stream interface
│
├── cam_handler.py                  ← PROVIDED — RealSense camera pipeline with
│                                       background reader thread; depth + IMU streams
│
├── imu.py                          ← PROVIDED — RealSense IMU reader and
│                                       complementary filter for roll/pitch estimation
│
├── drone_lib.py                    ← PROVIDED — DroneKit / MAVLink flight control
│                                       abstraction layer
│
├── pex03_utils.py                  ← PROVIDED — Geometry helpers (ground distance,
│                                       GPS projection, estimated distance from camera),
│                                       gripper control, frame I/O
│
└── yolo_visdrone/
    ├── yolov8s_visdrone.engine     ← TensorRT-compiled detection model
    ├── yolov4-tiny-custom.cfg      ← Legacy config (reference only)
    └── custom.names                ← Class names (pedestrian, people, ...)
```

---

## Program Architecture

Understanding how the modules relate to each other will help you debug problems and
understand why each layer exists.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          pex03.py                                   │
│   DroneMission — mission state machine, all flight decisions        │
│   conduct_mission() — the single loop that owns all transitions     │
└────────┬──────────────┬──────────────────┬──────────────────────────┘
         │              │                  │
         ▼              ▼                  ▼
┌──────────────┐  ┌────────────┐  ┌────────────────────────────────┐
│  drone_lib   │  │pex03_utils │  │  object_tracking_y8_histo      │
│              │  │            │  │                                │
│ DroneKit /   │  │ Geometry:  │  │ check_for_initial_target()     │
│ MAVLink      │  │ ground     │  │ set_object_to_track()          │
│ abstraction  │  │ distance,  │  │ track_with_confirm()           │
│              │  │ GPS proj., │  │ start_camera_stream()          │
│ goto_point() │  │ estimated  │  │ get_cur_frame()                │
│ small_move_* │  │ distance,  │  │ load_model()                   │
│ condition_   │  │ gripper,   │  └───────────────┬────────────────┘
│ yaw()        │  │ frame I/O  │                  │
└──────────────┘  └────────────┘                  ▼
                                       ┌────────────────────┐
                                       │    cam_handler     │
                                       │                    │
                                       │ RealSense pipeline │
                                       │ Background thread  │
                                       │ Depth + IMU cache  │
                                       └────────────────────┘
                                                  ▲
                                       ┌──────────┴─────────┐
                                       │       imu.py       │
                                       │                    │
                                       │ ImuReader class    │
                                       │ Complementary      │
                                       │ filter             │
                                       │ measure_tilt()     │
                                       └────────────────────┘
```

### Layering principle

`pex03.py` never talks to hardware directly. It calls `drone_lib` for flight,
`object_tracking_y8_histo` for perception, and `pex03_utils` for geometry. Those
modules call `cam_handler` for camera frames and `imu.py` for attitude data.
`pex03.py` has no knowledge of RealSense, MAVLink packets, or histogram computation.

This layering means you can swap out the tracker, the camera, or the flight controller
without rewriting mission logic — and it means you can test each layer independently.

---

## Configuration

**All tunable parameters live in `pex03_config.json`.** Do not hardcode values in
`pex03.py`. When you want to change a threshold, confidence level, path, or velocity,
edit the JSON file.

`mission_config.py` loads the JSON file, validates it, and exposes all settings as
typed attributes of a `MissionConfig` object. `DroneMission.__init__` receives one
`config` argument — everything else flows from there.

### Key settings to check before every run

```jsonc
{
  "virtual_mode": true,            // ← FALSE for live flight
  "connect_string": "127.0.0.1:14550",  // ← /dev/ttyACM0 for real hardware
  "mission_log_path": "/media/usafa/data/pex03_mission/cam",  // ← your team folder
  "weights_path": "/home/usafa/.../yolov8s_visdrone.engine",  // ← your model path
  "use_estimated_distance": true,  // ← false if using rangefinder
  "camera_tilt_from_imu": true     // ← false if camera has no IMU
}
```

### Config file search order

`MissionConfig.load()` checks these locations in order, using the first match:

1. `./pex03_config.json` (current working directory)
2. `student/pex03_config.json`
3. Alongside `mission_config.py` in the package directory

---

## Provided Modules — API Reference

### `object_tracking_y8_histo.py`

| Function | Signature | Notes |
|---|---|---|
| `start_camera_stream` | `(use_distance, resolution_width, ...)` | Call once at startup. Args come from config. |
| `load_model` | `(engine_path, imgsz, conf)` | Load YOLOv8. Call once after camera starts. |
| `get_cur_frame` | `(attempts=5, flip_v=False)` | Returns numpy array or **`None`** — always check. |
| `check_for_initial_target` | `(img, img_write, show_img, in_debug)` | Non-committing YOLO scan. Returns `(center, confidence, corner, radius, frame, bbox)`. `confidence=None` if nothing found. |
| `set_object_to_track` | `(frame, bbox, **kwargs)` | Lock tracker — call only after CONFIRM on a confirmed bbox. Builds the histogram signature. |
| `track_with_confirm` | `(img, img_write, show_img, **kwargs)` | Same 6-tuple. `confidence=None` and `_target_track_id=None` on total failure (budget exhausted). |

Module-level globals you may read:

```python
obj_track.FRAME_HORIZONTAL_CENTER   # int — frame centre x
obj_track.FRAME_VERTICAL_CENTER     # int — frame centre y
obj_track.FRAME_HEIGHT              # int — frame height in pixels
obj_track.FRAME_WIDTH               # int — frame width in pixels
obj_track._target_track_id          # None when tracker is not active
obj_track.TRACKER_MISSES_MAX        # set from config at startup
obj_track.HISTO_MATCH_THRESHOLD     # set from config at startup
```

### `pex03_utils.py`

| Function | What it does |
|---|---|
| `estimate_ground_distance_m(*, camera_tilt_deg, camera_fov_v_deg, target_y_px, frame_height, altitude_m)` | Camera-geometry ground distance estimate. Returns -1.0 if geometry is degenerate. |
| `get_ground_distance(height, hypotenuse)` | Pythagorean formula: `sqrt(hyp² - alt²)`. For rangefinder path. |
| `calc_new_location(lat, lon, heading, meters)` | Project GPS coordinate along a bearing. |
| `release_grip(drone, seconds=2)` | PWM 1087 on channel 7 — opens latch. |
| `get_avg_distance_to_obj(seconds, device, virtual_mode)` | Average rangefinder over N seconds. Returns 35.0 in virtual mode. |
| `write_frame(frame_num, frame, path)` | Save annotated PNG to mission log path. |

### `drone_lib.py`

| Function | Notes |
|---|---|
| `connect_device(address, baud)` | SITL: `"127.0.0.1:14550"` |
| `change_device_mode(drone, mode)` | `"AUTO"`, `"GUIDED"`, `"RTL"` |
| `goto_point(drone, lat, lon, speed, alt, timeout_s=60)` | **Returns True on arrival, False on timeout.** Handle both. |
| `condition_yaw(drone, heading)` | Rotate to bearing (0 = North) |
| `small_move_forward/back/left/right(drone, velocity)` | Blocks for 1 second. Drone moves ~velocity meters. |
| `small_move_down(drone, velocity)` | Positive = descend. Blocks for 1 second. |
| `return_to_launch(drone)` | Sets RTL mode. Returns immediately. |

> **Important:** `goto_point()` now returns `True` on arrival and `False` on timeout.
> Always capture the return value and decide what to do. During delivery, a timeout
> is not necessarily a reason to abort — deliver from the current position and log
> a warning.

### `imu.py`

| Class / Function | Notes |
|---|---|
| `ImuReader(alpha)` | Instantiate one before arming. |
| `.start()` | Starts the background IMU polling thread. |
| `.measure_tilt(duration_s)` | Blocks for `duration_s` seconds, returns measured tilt in degrees (positive = below horizontal), or `None` on failure. |
| `.stop()` | Call on clean exit or exception. |

### `mission_config.py`

| Attribute / Property | Type | Description |
|---|---|---|
| `config.effective_camera_tilt_deg` | `float` | IMU-measured tilt if available, else config fallback. Always use this. |
| `config.tilt_source` | `str` | `"imu"` or `"config"` — for logging. |
| `config.virtual_mode` | `bool` | Sim vs. live hardware. |
| `config.use_estimated_distance` | `bool` | Camera geometry vs. rangefinder. |
| All other fields | see file | Documented in `pex03_config.json` comments. |

---

## What You Must Build

Search `# TODO` in `pex03.py` to find every location with instructions.

| Location | What to implement |
|---|---|
| `conduct_mission()` | `frame = obj_track.get_cur_frame()` each loop iteration |
| `conduct_mission()` | GPS snapshot: `self.last_lat_pos`, `self.last_lon_pos`, `self.last_alt_pos` |
| `conduct_mission()` — SEEK state | Set `conf_level`. Justify your value. |
| `conduct_mission()` — SEEK state | `obj_track.set_object_to_track(frame, bbox)` |
| `conduct_mission()` — DELIVER state | `drone_lib.return_to_launch(self.drone)` |
| `adjust_to_target_center()` | Set `pixel_forgiveness`. Derive from camera geometry. |
| `adjust_to_target_center()` | Set `pixel_distance_threshold`. Justify from physical distance. |
| `adjust_to_target_center()` | Four velocity variables (`xv`, `yv`) for fast and slow correction |
| `adjust_to_target_center()` | Four `drone_lib.small_move_*` calls (fwd/back/left/right) |
| `deliver_package()` — rangefinder path | `pex03_utils.get_ground_distance(alt, slant_dist)` |
| `deliver_package()` | `pex03_utils.calc_new_location(lat, lon, heading, ground_dist)` |
| `deliver_package()` | `drone_lib.goto_point(...)` to fly to drop coordinate |
| `deliver_package()` | Set `alt_thresh`. Justify from tether length. |
| `deliver_package()` | `drone_lib.small_move_down(...)` in both descent loops |
| `deliver_package()` | `pex03_utils.release_grip(self.drone, seconds=2)` |

---

## Engineering Challenges

### Confidence threshold selection

`conf_level` controls when a detection is worth pausing the mission for. Too high:
the drone flies the entire search pattern without ever committing to a sighting. Too
low: every shadow triggers a 30-second CONFIRM cycle that wastes flight time and
battery. The right value depends on your YOLO weights, lighting, and search altitude.
You must reason about the tradeoff and justify your choice quantitatively.

### Pixel-to-meter conversion

Your centering thresholds are in pixels but the physical requirement (deliver within 3 m)
is in meters. You need to derive the relationship between pixels and meters at your
mission altitude. This requires camera FOV, frame resolution, and altitude. Do the
calculation and use it to justify every pixel-based threshold you set.

### Proportional control and the 1-second block

Each `small_move_*` call blocks for 1 second. The drone physically moves for that
second before the next camera frame is evaluated. This makes the effective correction
rate approximately 1 Hz. At 0.5 m/s, each correction moves the drone 0.5 m. Design
your velocity values knowing that every command executes for a full second before you
can re-evaluate — too fast and you overshoot; too slow and convergence takes minutes.

### Camera geometry distance estimation

`estimate_ground_distance_m()` depends on three measured or configured values:
camera tilt, vertical FOV, and GPS altitude. An error in any one of them shifts the
computed drop point. Understand the formula. Know which parameter has the most
influence on accuracy and how to verify your IMU tilt measurement is plausible.

### goto_point convergence

`goto_point()` returns `False` if the drone has not reached the target within
`timeout_s` seconds. This can happen due to GPS oscillation near the target, wind
opposing the approach, or a waypoint in a geofenced area. Your code must handle the
`False` return at every call site. "Handle" does not always mean "abort" — think
carefully about the right response in each context.

### Tether physics and the safe delivery radius

The package hangs on a 3-meter tether below the drone. When the drone moves laterally,
the tether swings. Aggressive corrections in TARGET mode create pendulum oscillations
that can cause early latch release or damage the payload. Slow, smooth corrections
are engineering requirements, not just style preferences.

When computing the drop point, you are placing the drone such that the package lands
approximately `SAFE_DELIVERY_RADIUS` meters from the target. Think about what that
radius should be: close enough to be useful to an injured person, far enough that
the descending package and rotor wash do not injure them. Derive it, don't guess it.

---

## Development Roadmap

Work through these stages in order. Do not attempt a live flight without completing
every earlier stage.

**Stage 1 — Camera and detection (no drone)**
Run `python object_tracking_y8_histo.py` standalone. Point the RealSense down at a
person from a height. Verify YOLOv8 detections appear, the tracker locks and shows
a histogram match score, FPS stays above 10. Use `in_debug=True` to detect cars
indoors during lab testing.

**Stage 2 — Configuration and startup**
Set `virtual_mode: true` in `pex03_config.json`. Run `python pex03.py`. Verify the
config loads and logs the summary table. Check that the IMU measurement runs (even
if it fails with no camera connected in sim).

**Stage 3 — SITL connectivity**
Start SITL, upload a waypoint search pattern mission. Verify the drone arms, takes
off, enters AUTO, and logs `[SEEK] Scanning...` each iteration.

**Stage 4 — Centering loop (TARGET mode)**
Implement `adjust_to_target_center()`. Force `mission_mode = MISSION_MODE_TARGET`
and a hardcoded off-centre `center` tuple. Verify the drone moves in the correct
direction and that the velocity values you have chosen feel physically reasonable in
SITL before flying.

**Stage 5 — CONFIRM sequence**
Implement the `set_object_to_track()` call. Force a confidence above your threshold.
Trace SEEK → CONFIRM → TARGET in the logs. Verify: GUIDED mode, `goto_point` called
and return value checked, heading restored, tracker locked only after second detection.

**Stage 6 — Delivery geometry (unit test)**
```python
from student.pex03_utils import estimate_ground_distance_m, calc_new_location

# At 15m altitude, 37.5° tilt, target at frame centre → ~19.6m forward
dist = estimate_ground_distance_m(
    camera_tilt_deg=37.5, camera_fov_v_deg=58.0,
    target_y_px=240, frame_height=480, altitude_m=15.0)
print(f"Ground dist: {dist:.2f} m")   # expect ~19.6 m

lat, lon = calc_new_location(39.0, -104.0, 0, dist - 3.0)  # drop 3m short
print(f"Drop: {lat:.6f}, {lon:.6f}")
```
Then implement the full delivery sequence in SITL and verify altitude decrements
through both thresholds in the log.

**Stage 7 — Full integration (real camera, SITL)**
Real RealSense + SITL. Run the IMU tilt measurement. Force a detection and run the
full state sequence. Review annotated PNGs in `mission_log_path`.

**Stage 8 — Live flight**
Set `virtual_mode: false` in `pex03_config.json`. Conduct a bound tethered flight
before the evaluated mission. Verify the gripper opens on command.

---

## SITL Testing Guide

```bash
# From your ArduPilot directory:
sim_vehicle.py -v ArduCopter --console --map
```

SITL listens on `127.0.0.1:14550`. Set `connect_string: "127.0.0.1:14550"` and
`virtual_mode: true` in `pex03_config.json`.

| ✅ Can test in SITL | ❌ Cannot test in SITL |
|---|---|
| All five state transitions | YOLOv8 detection (test with `object_tracking_y8_histo.py` standalone) |
| AUTO ↔ GUIDED ↔ RTL mode switches | Depth sensor (`virtual_mode=true` returns fixed 35.0 m) |
| `goto_point` including timeout handling | Camera tilt IMU measurement (no camera connected) |
| `condition_yaw` heading restoration | Gripper servo (bench test separately) |
| Altitude-based descent loops | Wind, vibration, tether swing |
| GPS drop coordinate calculation | |
| Config loading and log output | |
| Frame snapshot writes to disk | |

---

## Tunable Parameters

All parameters are set in `pex03_config.json`. Do not hardcode these in `pex03.py`.

| Parameter | Config key | Default | Effect of increasing |
|---|---|---|---|
| Detection confidence | `detect_confidence` | 0.25 | Fewer weak detections passed to state machine |
| Mission confidence threshold | Set in `pex03.py` code | — | Fewer false CONFIRM cycles; risk of missing target |
| Max confirm attempts | `max_confirm_attempts` | 8 | More retries before returning to SEEK |
| Acceptance radius | `target_acceptance_radius_px` | 10 px | Looser centering required before delivery |
| Tracker miss budget | `tracker_misses_max` | 20 | More tolerant of brief occlusions |
| Histogram match threshold | `histo_match_threshold` | 0.70 | Stricter identity matching; reduce false swaps |
| Update rate | `update_rate` | 1 | Scan every N frames during SEEK (CPU tradeoff) |
| Image log rate | `image_log_rate` | 4 | Save every N frames to disk |
| Camera tilt | `camera_tilt_deg` | 37.5° | Affects all distance estimates (use IMU measurement) |
| Use estimated distance | `use_estimated_distance` | true | true = camera geometry; false = rangefinder |

---

## Hardware Reference

| Component | Details |
|---|---|
| Drone | Quadrotor with ArduPilot autopilot (Pixhawk or equivalent) |
| Companion computer | NVIDIA Jetson; runs all Python code |
| Camera | Intel RealSense D455 — 640×480 BGR + Z16 depth at 60 fps; built-in IMU |
| Autopilot link | `/dev/ttyACM0` at 115200 baud |
| Gripper servo | RC channel 7; PWM 1087 = open, PWM 1940 = closed |
| Tether | 3.0 m (≈10 feet) from drone latch to care package |
| Payload | First-aid box containing eggs |

**Pre-flight checklist:**
1. `pex03_config.json` — `virtual_mode: false`, correct `connect_string`, correct paths.
2. `python object_tracking_y8_histo.py` — camera streams, YOLOv8 detects, histogram tracker locks.
3. Gripper bench test — latch opens cleanly on `release_grip()` command.
4. Mission uploaded and verified: `drone.commands.count > 0`.
5. `MIS_RESTART = 0` on the autopilot — mission resumes at current waypoint, not from the beginning.
6. IMU tilt measurement runs at startup — verify the logged tilt angle is physically plausible.

---

## Grading

Teams are ranked against each other on mission day. All grading is competitive —
the score awarded depends on your rank relative to the other teams, not on an
absolute standard.

| Place | Score | Criterion |
|---|---|---|
| **1st** | **100%** | Fastest execution AND most accurate delivery |
| **2nd** | **95%** | Strong performance — target identified, package within 10 ft |
| **3rd and below** | **90%** | Mission completed successfully |
| **Failed to execute** | **80% starting point** | Score adjusted down based on code review and team interviews |

**Scoring dimensions (all teams that execute):**
1. **Target identification** — Was the correct person detected and confirmed?
2. **Delivery proximity** — How close did the package land to the target (10-foot requirement)?
3. **Payload integrity** — Are the eggs unbroken?
4. **Mission speed** — Total elapsed time from takeoff to RTL.
5. **Execution smoothness** — Excessive oscillation in TARGET mode, long pauses between
   states, and erratic corrections are penalised here.

**Teams that fail to execute** will be graded starting at 80%. The final score may be
lower than 80% based on a combination of:
- Code review: Is the implementation logically sound? Are all TODOs addressed?
- Team interviews: Can each team member explain their design decisions and parameter choices?
- Partial credit for states that functioned correctly before failure.

**Documentation requirement:** A project log is required alongside your code submission.
It must include the main technical challenges you encountered, how you resolved each one,
and each team member's specific contributions. A missing or vague project log results in
a **12% deduction** from the final project score.
