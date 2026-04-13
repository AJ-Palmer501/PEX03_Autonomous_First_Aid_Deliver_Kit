"""
sentinel/cam_handler.py
=======================
Single-owner RealSense camera handler for Sentinel.

This module owns the RealSense pipeline, a single background reader thread,
and the cached outputs derived from that one stream-consumption path:

* latest color frame
* latest depth frame (when enabled)
* latest accel / gyro motion samples
* frame geometry and timestamps
* restart / reconnect bookkeeping

Design intent
-------------
The rest of Sentinel should not compete for the pipeline.  Mission code reads
frames and timestamps from here, the IMU helper consumes cached motion samples
from here, and the tracker stays focused on detection / tracking rather than
camera lifecycle.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
    if not hasattr(rs, 'stream'):
        import pyrealsense2.pyrealsense2 as rs  # type: ignore
except Exception:
    import pyrealsense2.pyrealsense2 as rs  # type: ignore

LOG = logging.getLogger('sentinel_drone.cam_handler')


@dataclass
class CameraState:
    frame_width: int = 640
    frame_height: int = 480
    frame_horizontal_center: int = 320
    frame_vertical_center: int = 240
    streaming_distance: bool = False
    last_frame_timestamp: float = 0.0
    last_motion_timestamp: float = 0.0
    latest_frame: np.ndarray | None = None
    latest_depth_image: np.ndarray | None = None
    depth_scale_m_per_unit: float = 0.0
    latest_accel_data: object | None = None
    latest_gyro_data: object | None = None
    last_wait_error: str = ''
    consecutive_wait_failures: int = 0
    started: bool = False
    reader_running: bool = False


@dataclass
class StreamRequest:
    use_distance: bool = False
    resolution_width: int = 640
    resolution_height: int = 480
    color_rate: int = 60
    depth_rate: int = 30
    enable_imu: bool = False
    imu_accel_rate: int = 200
    imu_gyro_rate: int = 400


_pipeline = rs.pipeline()
_config = rs.config()
_state = CameraState()
_last_request = StreamRequest()
_lock = threading.RLock()
_reader_thread: threading.Thread | None = None
_reader_stop_event: threading.Event | None = None


def _build_request(use_distance: bool = False,
                   resolution_width: int = 640,
                   resolution_height: int = 480,
                   color_rate: int = 60,
                   depth_rate: int = 30,
                   enable_imu: bool = False,
                   imu_accel_rate: int = 200,
                   imu_gyro_rate: int = 400) -> StreamRequest:
    
    return StreamRequest(
        use_distance=bool(use_distance),
        resolution_width=int(resolution_width),
        resolution_height=int(resolution_height),
        color_rate=int(color_rate),
        depth_rate=int(depth_rate),
        enable_imu=bool(enable_imu),
        imu_accel_rate=int(imu_accel_rate),
        imu_gyro_rate=int(imu_gyro_rate),
    )


def _reader_loop(pipeline, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=250)
            except TypeError:
                frames = pipeline.wait_for_frames()

            rgb_frame = frames.get_color_frame()
            if not rgb_frame:
                with _lock:
                    _state.consecutive_wait_failures += 1
                continue

            frame = np.asanyarray(rgb_frame.get_data()).copy()
            frame_ts: float
            try:
                frame_ts = float(rgb_frame.get_timestamp())
            except Exception:
                frame_ts = time.time()

            depth_image = None
            if _state.streaming_distance:
                try:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data()).copy()
                except Exception:
                    depth_image = None

            accel = None
            gyro = None
            motion_ts = frame_ts
            try:
                accel_frame = frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    accel = accel_frame.as_motion_frame().get_motion_data()
                    try:
                        motion_ts = float(accel_frame.get_timestamp())
                    except Exception:
                        pass
            except Exception:
                accel = None

            try:
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if gyro_frame:
                    gyro = gyro_frame.as_motion_frame().get_motion_data()
                    try:
                        motion_ts = max(motion_ts, float(gyro_frame.get_timestamp()))
                    except Exception:
                        pass
            except Exception:
                gyro = None

            with _lock:
                _state.latest_frame = frame
                _state.last_frame_timestamp = frame_ts
                _state.frame_height = int(frame.shape[0])
                _state.frame_width = int(frame.shape[1])
                _state.frame_horizontal_center = _state.frame_width // 2
                _state.frame_vertical_center = _state.frame_height // 2
                if depth_image is not None:
                    _state.latest_depth_image = depth_image
                if accel is not None:
                    _state.latest_accel_data = accel
                if gyro is not None:
                    _state.latest_gyro_data = gyro
                if accel is not None or gyro is not None:
                    _state.last_motion_timestamp = motion_ts
                _state.consecutive_wait_failures = 0
                _state.last_wait_error = ''
        except Exception as exc:
            if stop_event.is_set():
                break
            with _lock:
                _state.consecutive_wait_failures += 1
                _state.last_wait_error = str(exc)
            time.sleep(0.01)

    with _lock:
        _state.reader_running = False


def start_camera_stream(use_distance: bool = False,
                        resolution_width: int = 640,
                        resolution_height: int = 480,
                        color_rate: int = 60,
                        depth_rate: int = 30,
                        enable_imu: bool = False,
                        imu_accel_rate: int = 200,
                        imu_gyro_rate: int = 400) -> None:
    global _pipeline, _config, _last_request, _reader_thread, _reader_stop_event
    request = _build_request(
        use_distance=use_distance,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        color_rate=color_rate,
        depth_rate=depth_rate,
        enable_imu=enable_imu,
        imu_accel_rate=imu_accel_rate,
        imu_gyro_rate=imu_gyro_rate,
    )

    stop_camera_stream()

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        request.resolution_width,
        request.resolution_height,
        rs.format.bgr8,
        color_rate,
    )

    if request.use_distance:
        config.enable_stream(
            rs.stream.depth,
            request.resolution_width,
            request.resolution_height,
            rs.format.z16,
            request.depth_rate,
        )

    if request.enable_imu:
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)
    time.sleep(1.00)
    depth_scale = 0.0
    if request.use_distance:
        try:
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        except Exception as exc:
            LOG.warning('Could not query depth scale: %s', exc)
            depth_scale = 0.0

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_reader_loop,
        args=(pipeline, stop_event),
        name='sentinel-camera',
        daemon=True,
    )

    with _lock:
        _pipeline = pipeline
        _config = config
        _last_request = request
        _reader_stop_event = stop_event
        _reader_thread = thread
        _state.frame_width = int(request.resolution_width)
        _state.frame_height = int(request.resolution_height)
        _state.frame_horizontal_center = _state.frame_width // 2
        _state.frame_vertical_center = _state.frame_height // 2
        _state.streaming_distance = request.use_distance
        _state.last_frame_timestamp = 0.0
        _state.last_motion_timestamp = 0.0
        _state.latest_frame = None
        _state.latest_depth_image = None
        _state.depth_scale_m_per_unit = depth_scale
        _state.latest_accel_data = None
        _state.latest_gyro_data = None
        _state.last_wait_error = ''
        _state.consecutive_wait_failures = 0
        _state.started = True
        _state.reader_running = True

    thread.start()
    
    LOG.info(
        'Camera stream started: %dx%d @ %d fps  depth=%s  imu=%s  accel=%d  gyro=%d',
        request.resolution_width, request.resolution_height, request.color_rate,
        request.use_distance, request.enable_imu,
        request.imu_accel_rate, request.imu_gyro_rate,
    )


def stop_camera_stream() -> None:
    global _pipeline, _config, _reader_thread, _reader_stop_event

    with _lock:
        thread = _reader_thread
        stop_event = _reader_stop_event
        pipeline = _pipeline
        _reader_thread = None
        _reader_stop_event = None

    if stop_event is not None:
        stop_event.set()

    try:
        pipeline.stop()
    except Exception:
        pass

    if thread is not None:
        thread.join(timeout=2.0)

    with _lock:
        _pipeline = rs.pipeline()
        _config = rs.config()
        _state.streaming_distance = False
        _state.last_frame_timestamp = 0.0
        _state.last_motion_timestamp = 0.0
        _state.latest_frame = None
        _state.latest_depth_image = None
        _state.depth_scale_m_per_unit = 0.0
        _state.latest_accel_data = None
        _state.latest_gyro_data = None
        _state.last_wait_error = ''
        _state.consecutive_wait_failures = 0
        _state.started = False
        _state.reader_running = False


def restart_camera_stream(**kwargs) -> None:
    with _lock:
        request = _last_request

    if kwargs:
        request = _build_request(
            use_distance=kwargs.get('use_distance', request.use_distance),
            resolution_width=kwargs.get('resolution_width', request.resolution_width),
            resolution_height=kwargs.get('resolution_height', request.resolution_height),
            color_rate=kwargs.get('color_rate', request.color_rate),
            depth_rate=kwargs.get('depth_rate', request.depth_rate),
            enable_imu=kwargs.get('enable_imu', request.enable_imu),
            imu_accel_rate=kwargs.get('imu_accel_rate', request.imu_accel_rate),
            imu_gyro_rate=kwargs.get('imu_gyro_rate', request.imu_gyro_rate),
        )

    LOG.warning('Restarting camera stream through cam_handler.')
    start_camera_stream(
        use_distance=request.use_distance,
        resolution_width=request.resolution_width,
        resolution_height=request.resolution_height,
        color_rate=request.color_rate,
        depth_rate=request.depth_rate,
        enable_imu=request.enable_imu,
        imu_accel_rate=request.imu_accel_rate,
        imu_gyro_rate=request.imu_gyro_rate,
    )


def get_cur_frame(attempts: int = 5, flip_v: bool = False) -> Optional[np.ndarray]:
    deadline = time.time() + max(1, int(attempts)) * 0.01
    while time.time() < deadline:
        with _lock:
            frame = None if _state.latest_frame is None else _state.latest_frame.copy()
        if frame is not None:
            if flip_v:
                frame = cv2.flip(frame, 0)
            return frame
        time.sleep(0.005)
    return None


def get_distance_using_center() -> float:
    with _lock:
        depth = None if _state.latest_depth_image is None else _state.latest_depth_image
        scale = float(_state.depth_scale_m_per_unit)
        cx = int(_state.frame_horizontal_center)
        cy = int(_state.frame_vertical_center)
    if depth is None or scale <= 0.0:
        return 0.0
    h, w = depth.shape[:2]
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return 0.0
    return float(depth[cy, cx]) * scale


def get_avg_distance_to_obj(seconds: float, virtual_mode: bool = False) -> float:
    if virtual_mode:
        return 25.0
    total = 0.0
    count = 0
    t_end = time.time() + seconds
    while time.time() < t_end:
        d = get_distance_using_center()
        if d > 0.0:
            total += d
            count += 1
        time.sleep(0.01)
    return total / count if count > 0 else 0.0


def get_pipeline():
    return _pipeline


def get_latest_motion_data() -> Tuple[object | None, object | None, float]:
    with _lock:
        return _state.latest_accel_data, _state.latest_gyro_data, _state.last_motion_timestamp


def get_latest_accel_data():
    with _lock:
        return _state.latest_accel_data


def get_latest_gyro_data():
    with _lock:
        return _state.latest_gyro_data


def get_last_frame_timestamp() -> float:
    with _lock:
        return _state.last_frame_timestamp


def get_latest_frame(copy: bool = True) -> Optional[np.ndarray]:
    with _lock:
        if _state.latest_frame is None:
            return None
        return _state.latest_frame.copy() if copy else _state.latest_frame


def get_frame_geometry() -> Tuple[int, int, int, int]:
    with _lock:
        return (
            _state.frame_width,
            _state.frame_height,
            _state.frame_horizontal_center,
            _state.frame_vertical_center,
        )


def is_streaming_distance() -> bool:
    with _lock:
        return _state.streaming_distance


def get_last_request() -> dict:
    with _lock:
        return {
            'use_distance': _last_request.use_distance,
            'resolution_width': _last_request.resolution_width,
            'resolution_height': _last_request.resolution_height,
            'color_rate': _last_request.color_rate,
            'depth_rate': _last_request.depth_rate,
            'enable_imu': _last_request.enable_imu,
            'imu_accel_rate': _last_request.imu_accel_rate,
            'imu_gyro_rate': _last_request.imu_gyro_rate,
        }


def get_state_snapshot() -> dict:
    with _lock:
        return {
            'frame_width': _state.frame_width,
            'frame_height': _state.frame_height,
            'frame_horizontal_center': _state.frame_horizontal_center,
            'frame_vertical_center': _state.frame_vertical_center,
            'streaming_distance': _state.streaming_distance,
            'last_frame_timestamp': _state.last_frame_timestamp,
            'last_motion_timestamp': _state.last_motion_timestamp,
            'has_latest_frame': _state.latest_frame is not None,
            'has_latest_depth': _state.latest_depth_image is not None,
            'latest_accel_data': _state.latest_accel_data,
            'latest_gyro_data': _state.latest_gyro_data,
            'last_wait_error': _state.last_wait_error,
            'consecutive_wait_failures': _state.consecutive_wait_failures,
            'started': _state.started,
            'reader_running': _state.reader_running,
        }
