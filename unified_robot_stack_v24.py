#!/usr/bin/env python3
from __future__ import annotations
import math
import struct
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath, OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan, NavSatFix
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
from geometry_msgs.msg import TransformStamped
import serial
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # General quaternion -> yaw conversion (ZYX)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def geodetic_to_local_xy(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    r = 6378137.0
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    return r * dlon * math.cos(math.radians((lat + lat0) * 0.5)), r * dlat


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


class VisionModel:
    def __init__(self, backend: str, model_path: str):
        self.backend, self.ready = backend, False
        p = Path(model_path)
        if not p.exists():
            return
        if backend == 'tflite':
            try:
                import tflite_runtime.interpreter as tflite
            except Exception:
                import tensorflow.lite as tflite  # type: ignore
            self.i = tflite.Interpreter(model_path=str(p))
            self.i.allocate_tensors()
            self.inp = self.i.get_input_details()[0]['index']
            self.out = self.i.get_output_details()[0]['index']
            self.ready = True
        elif backend == 'torch':
            import torch
            self.torch = torch
            self.m = torch.jit.load(str(p), map_location='cpu')
            self.m.eval()
            self.ready = True

    def infer(self, frame: np.ndarray) -> tuple[float, float]:
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.backend == 'tflite':
            x = np.expand_dims(rgb, 0)
            self.i.set_tensor(self.inp, x)
            self.i.invoke()
            out = self.i.get_tensor(self.out).reshape(-1).astype(np.float64)
        elif self.backend == 'torch':
            t = self.torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
            with self.torch.no_grad():
                out = self.m(t).reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            out = np.array([0.0], dtype=np.float64)

        if out.size <= 1:
            s = float(clamp(float(out[0]), 0.0, 1.0))
            c = float(clamp(abs(s - 0.5) * 2.0, 0.0, 1.0))
            return s, c

        logits = np.clip(out - np.max(out), -60.0, 60.0)
        exp_logits = np.exp(logits)
        probs = exp_logits / max(float(np.sum(exp_logits)), 1e-12)
        s = float(clamp(float(np.max(probs)), 0.0, 1.0))
        ent = float(-np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))))
        ent_norm = ent / max(math.log(float(out.size)), 1e-12)
        c = float(clamp(1.0 - ent_norm, 0.0, 1.0))
        return s, c


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.declare_parameter('publish_hz', 20.0)
        self.declare_parameter('scan_front_half_angle_deg', 25.0)
        self.declare_parameter('model_backend', 'tflite')
        self.declare_parameter('model_path', '/home/pi/models/obstacle_v22.tflite')
        self.declare_parameter('semantic_threshold', 0.65)
        self.declare_parameter('dynamic_speed_threshold', 0.03)
        self.declare_parameter('class_pedestrian_score', 0.72)
        self.declare_parameter('class_box_score', 0.58)
        self.hz = float(self.get_parameter('publish_hz').value)
        self.half = math.radians(float(self.get_parameter('scan_front_half_angle_deg').value))
        self.model = VisionModel(str(self.get_parameter('model_backend').value), str(self.get_parameter('model_path').value))

        self.bridge = CvBridge()
        self.last_frame: np.ndarray | None = None
        self.score, self.conf, self.front = 0.0, 0.0, 99.0
        self.semantic = False
        self.track_state = np.zeros(4, dtype=np.float64)  # x,y,vx,vy normalized image coords
        self.track_cov = np.eye(4, dtype=np.float64) * 0.2
        self.last_track_ts = time.monotonic()
        self.sem_th = float(self.get_parameter('semantic_threshold').value)
        self.dynamic_th = float(self.get_parameter('dynamic_speed_threshold').value)
        self.obj_class = 'Diger'
        self.infer_ema, self.infer_max = 0.0, 0.0
        self._frame_lock = threading.Lock()
        self._run_worker = True
        self._last_processed_ts = 0.0

        self.create_subscription(Image, '/camera/image_raw', self.on_image, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 30)
        self.pub_score = self.create_publisher(Float32, '/ai/vision_obstacle_score', 10)
        self.pub_conf = self.create_publisher(Float32, '/ai/vision_confidence', 10)
        self.pub_front = self.create_publisher(Float32, '/ai/front_distance_m', 10)
        self.pub_inf = self.create_publisher(Float32, '/ai/inference_latency_ms', 10)
        self.pub_inf_max = self.create_publisher(Float32, '/ai/inference_latency_max_ms', 10)
        self.pub_sem = self.create_publisher(Bool, '/ai/semantic_obstacle', 10)
        self.pub_track = self.create_publisher(Float32MultiArray, '/ai/object_track', 10)
        self.pub_obj_class = self.create_publisher(String, '/ai/object_class', 10)
        self.pub_hb = self.create_publisher(Float32, '/heartbeat/vision', 10)
        self.worker = threading.Thread(target=self.vision_worker, daemon=True)
        self.worker.start()
        self.create_timer(1.0 / self.hz, self.on_timer)
        self.create_timer(0.2, lambda: self.pub_hb.publish(Float32(data=time.monotonic())))

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._frame_lock:
            self.last_frame = frame
            self._last_frame_ts = time.monotonic()

    def on_scan(self, msg: LaserScan):
        a, vals = msg.angle_min, []
        for r in msg.ranges:
            if -self.half <= a <= self.half and math.isfinite(r) and msg.range_min <= r <= msg.range_max:
                vals.append(r)
            a += msg.angle_increment
        self.front = float(min(vals)) if vals else 99.0

    def process_frame(self, frame: np.ndarray | None = None):
        if frame is None:
            with self._frame_lock:
                frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return
        t0 = time.monotonic()
        if self.model.ready:
            self.score, self.conf = self.model.infer(frame)
        else:
            h, w, _ = frame.shape
            roi = frame[int(0.6 * h):h, int(0.1 * w):int(0.9 * w)]
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 3)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            self.score, self.conf = float(np.count_nonzero(th) / (th.size + 1)), 0.25
        self.semantic = (self.score * self.conf) > self.sem_th
        self.update_object_tracker(frame)
        self.classify_object()
        dt = (time.monotonic() - t0) * 1000.0
        if self.infer_ema <= 1e-6:
            self.infer_ema = dt
        else:
            self.infer_ema = 0.9 * self.infer_ema + 0.1 * dt
        self.infer_max = max(self.infer_max * 0.995, dt)


    def update_object_tracker(self, frame: np.ndarray):
        h, w, _ = frame.shape
        roi = frame[int(0.45 * h):h, :]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        now = time.monotonic()
        dt = clamp(now - self.last_track_ts, 1e-3, 0.2)
        self.last_track_ts = now

        # predict
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Q = np.diag([1e-3, 1e-3, 2e-2, 2e-2])
        self.track_state = F @ self.track_state
        self.track_cov = F @ self.track_cov @ F.T + Q

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(c)
            if a > 40:
                m = cv2.moments(c)
                if m['m00'] > 1e-6:
                    mx = (m['m10'] / m['m00']) / max(float(w), 1.0)
                    my = ((m['m01'] / m['m00']) + 0.45 * h) / max(float(h), 1.0)
                    z = np.array([mx, my], dtype=np.float64)
                    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
                    R = np.diag([2e-3, 2e-3])
                    y = z - H @ self.track_state
                    S = H @ self.track_cov @ H.T + R
                    K = self.track_cov @ H.T @ np.linalg.inv(S)
                    self.track_state = self.track_state + K @ y
                    I = np.eye(4)
                    self.track_cov = (I - K @ H) @ self.track_cov

    def tracker_msg(self) -> Float32MultiArray:
        spd = float(math.hypot(self.track_state[2], self.track_state[3]))
        is_dyn = 1.0 if spd > self.dynamic_th else 0.0
        return Float32MultiArray(data=[float(self.track_state[0]), float(self.track_state[1]), float(self.track_state[2]), float(self.track_state[3]), is_dyn])




    def classify_object(self):
        dyn = math.hypot(self.track_state[2], self.track_state[3]) > self.dynamic_th
        if self.semantic and dyn and (self.score > float(self.get_parameter('class_pedestrian_score').value)):
            self.obj_class = 'Yaya'
        elif self.semantic and (self.score > float(self.get_parameter('class_box_score').value)):
            self.obj_class = 'Kutu' if not dyn else 'Diger Robot'
        else:
            self.obj_class = 'Diger'


    def vision_worker(self):
        period = 1.0 / max(self.hz, 1.0)
        while self._run_worker:
            frame = None
            ts = 0.0
            with self._frame_lock:
                if self.last_frame is not None:
                    ts = getattr(self, '_last_frame_ts', 0.0)
                    if ts > self._last_processed_ts:
                        frame = self.last_frame.copy()
            if frame is not None:
                self.process_frame(frame)
                self._last_processed_ts = ts
            time.sleep(period * 0.7)

    def destroy_node(self):
        self._run_worker = False
        if hasattr(self, "worker") and self.worker.is_alive():
            self.worker.join(timeout=1.0)
        return super().destroy_node()

    def on_timer(self):
        self.pub_score.publish(Float32(data=self.score))
        self.pub_conf.publish(Float32(data=self.conf))
        self.pub_front.publish(Float32(data=self.front))
        self.pub_inf.publish(Float32(data=self.infer_ema))
        self.pub_inf_max.publish(Float32(data=self.infer_max))
        self.pub_sem.publish(Bool(data=self.semantic))
        self.pub_track.publish(self.tracker_msg())
        self.pub_obj_class.publish(String(data=self.obj_class))


class ScanMatcherNode(Node):
    def __init__(self):
        super().__init__('scan_matcher_node')
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('icp_gain', 0.25)
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.icp_gain = float(self.get_parameter('icp_gain').value)
        self.prev_pts: np.ndarray | None = None

        self.pub = self.create_publisher(Float32MultiArray, '/scan_match/delta', 20)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 20)


    def on_scan(self, m: LaserScan):
        pts = []
        a = m.angle_min
        for r in m.ranges:
            if math.isfinite(r) and m.range_min <= r <= min(m.range_max, self.max_range):
                pts.append([r * math.cos(a), r * math.sin(a)])
            a += m.angle_increment
        if len(pts) < 40:
            return
        cur = np.asarray(pts, dtype=np.float64)
        # downsample for deterministic runtime
        cur = cur[::max(1, len(cur) // 250)]

        dx = dy = dyaw = 0.0
        score = 0.0
        if self.prev_pts is not None and len(self.prev_pts) >= 40:
            ref = self.prev_pts
            if len(ref) > len(cur):
                ref = ref[::max(1, len(ref) // max(1, len(cur)))]
            n = min(len(ref), len(cur))
            A = ref[:n]
            B = cur[:n]
            ca = np.mean(A, axis=0)
            cb = np.mean(B, axis=0)
            AA = A - ca
            BB = B - cb
            H = AA.T @ BB
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[1, :] *= -1
                R = Vt.T @ U.T
            t = cb - R @ ca
            dyaw = float(math.atan2(R[1, 0], R[0, 0]) * self.icp_gain)
            dx = float(t[0] * self.icp_gain)
            dy = float(t[1] * self.icp_gain)
            score = float(min(1.0, np.mean(np.linalg.norm((AA @ R.T + t) - BB, axis=1))))

        self.prev_pts = cur
        self.pub.publish(Float32MultiArray(data=[dx, dy, dyaw, score]))


class LocalCostmapNode(Node):
    def __init__(self):
        super().__init__('local_costmap_node')
        self.declare_parameter('size_m', 6.0)
        self.declare_parameter('resolution_m', 0.1)
        self.declare_parameter('inflation_m', 0.35)
        self.declare_parameter('inflation_yaya_m', 0.90)
        self.declare_parameter('inflation_kutu_m', 0.55)
        self.declare_parameter('inflation_diger_robot_m', 0.75)
        self.declare_parameter('inflation_diger_m', 0.45)
        self.size_m = float(self.get_parameter('size_m').value)
        self.res = float(self.get_parameter('resolution_m').value)
        self.infl = float(self.get_parameter('inflation_m').value)
        self.class_inflation = {
            'Yaya': float(self.get_parameter('inflation_yaya_m').value),
            'Kutu': float(self.get_parameter('inflation_kutu_m').value),
            'Diger Robot': float(self.get_parameter('inflation_diger_robot_m').value),
            'Diger': float(self.get_parameter('inflation_diger_m').value),
        }
        self.width = int(self.size_m / self.res)
        self.height = int(self.size_m / self.res)

        self.pub_map = self.create_publisher(OccupancyGrid, '/local_costmap', 10)
        self.pub_scale = self.create_publisher(Float32, '/local_cost_scale', 10)
        self.pub_obstacle = self.create_publisher(Float32MultiArray, '/local/obstacle_hint', 10)
        self.semantic_obstacle = False
        self.object_track = np.zeros(5, dtype=np.float64)
        self.object_class = 'Diger'
        self.map_cost = None
        self.create_subscription(LaserScan, '/scan', self.on_scan, 20)
        self.create_subscription(Bool, '/ai/semantic_obstacle', lambda m: setattr(self, 'semantic_obstacle', bool(m.data)), 20)
        self.create_subscription(Float32MultiArray, '/ai/object_track', self.on_track, 20)
        self.create_subscription(String, '/ai/object_class', self.on_class, 20)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, 2)



    def on_track(self, m: Float32MultiArray):
        if len(m.data) >= 5:
            self.object_track = np.array(m.data[:5], dtype=np.float64)


    def on_class(self, m: String):
        name = str(m.data).strip()
        self.object_class = name if name in self.class_inflation else 'Diger'

    def current_inflation(self) -> float:
        return self.class_inflation.get(self.object_class, self.infl)

    def on_map(self, m: OccupancyGrid):
        self.map_cost = m

    def mark_semantic_zone(self, grid: np.ndarray, cx: int, cy: int):
        if not self.semantic_obstacle:
            return
        # S-maneuver trigger: block center corridor and leave side corridors lower cost
        extra = max(2, int(self.current_inflation() / max(self.res, 1e-3)))
        sx0 = max(0, cx - extra)
        sx1 = min(self.width, cx + extra + 1)
        sy0 = max(0, cy - (8 + extra))
        sy1 = min(self.height, cy + (6 + extra))
        grid[sy0:sy1, sx0:sx1] = 100

    def overlay_global_map(self, grid: np.ndarray):
        if self.map_cost is None or self.map_cost.info.width == 0:
            return
        # lightweight map_server read: sample center patch from /map into local grid
        mw = int(self.map_cost.info.width)
        mh = int(self.map_cost.info.height)
        data = np.array(self.map_cost.data, dtype=np.int16)
        if data.size != mw * mh:
            return
        mg = data.reshape((mh, mw))
        cy = mh // 2
        cx = mw // 2
        patch = mg[max(0, cy - 15):min(mh, cy + 15), max(0, cx - 15):min(mw, cx + 15)]
        ph, pw = patch.shape
        gy0 = max(0, self.height // 2 - ph // 2)
        gx0 = max(0, self.width // 2 - pw // 2)
        gview = grid[gy0:gy0 + ph, gx0:gx0 + pw]
        gview[:] = np.maximum(gview, np.where(patch > 70, 90, 0).astype(np.uint8))

    def on_scan(self, m: LaserScan):
        grid = np.zeros((self.height, self.width), dtype=np.uint8)
        cx = self.width // 2
        cy = self.height // 2
        infl_m = self.current_inflation() if self.semantic_obstacle else self.infl
        infl_cells = max(1, int(infl_m / self.res))
        a = m.angle_min
        near_count = 0
        for r in m.ranges:
            if math.isfinite(r) and m.range_min <= r <= m.range_max and r <= self.size_m * 0.7:
                x = r * math.cos(a)
                y = r * math.sin(a)
                ix = int(cx + x / self.res)
                iy = int(cy + y / self.res)
                if 0 <= ix < self.width and 0 <= iy < self.height:
                    near_count += 1 if r < 1.2 else 0
                    x0 = max(0, ix - infl_cells); x1 = min(self.width, ix + infl_cells + 1)
                    y0 = max(0, iy - infl_cells); y1 = min(self.height, iy + infl_cells + 1)
                    grid[y0:y1, x0:x1] = np.maximum(grid[y0:y1, x0:x1], 80)
                    grid[iy, ix] = 100
            a += m.angle_increment

        self.mark_semantic_zone(grid, cx, cy)

        # dynamic object predicted point from tracker
        ox = float(self.object_track[0] + 0.6 * self.object_track[2])
        oy = float(self.object_track[1] + 0.6 * self.object_track[3])
        if self.semantic_obstacle:
            ix = int(cx + (ox - 0.5) * self.width * 0.6)
            iy = int(cy + (oy - 0.75) * self.height * 0.8)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                grid[max(0, iy - 2):min(self.height, iy + 3), max(0, ix - 2):min(self.width, ix + 3)] = 100

        self.overlay_global_map(grid)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.info.resolution = float(self.res)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = -0.5 * self.size_m
        msg.info.origin.position.y = -0.5 * self.size_m
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().astype(np.int8).tolist()
        self.pub_map.publish(msg)

        occ_ratio = float(np.count_nonzero(grid >= 80)) / float(self.width * self.height)
        scale = clamp(1.0 - 2.2 * occ_ratio - 0.02 * near_count, 0.25, 1.0)
        self.pub_scale.publish(Float32(data=scale))
        side = -1.0 if ox < 0.5 else 1.0
        lateral_bias = side * min(1.0, max(0.25, infl_m))
        class_id = {'Yaya': 1.0, 'Kutu': 2.0, 'Diger Robot': 3.0, 'Diger': 0.0}.get(self.object_class, 0.0)
        self.pub_obstacle.publish(Float32MultiArray(data=[ox, oy, float(1.0 if self.semantic_obstacle else 0.0), lateral_bias, infl_m, class_id]))


class SlamNav2BridgeNode(Node):
    """Expose SLAM/map readiness for Nav2 integration visibility."""
    def __init__(self):
        super().__init__('slam_nav2_bridge')
        self.declare_parameter('map_timeout_s', 1.0)
        self.declare_parameter('publish_hz', 5.0)
        self.map_timeout = float(self.get_parameter('map_timeout_s').value)
        self.hz = float(self.get_parameter('publish_hz').value)

        self.last_map_mono = 0.0
        self.map_res = 0.0
        self.map_w = 0
        self.map_h = 0

        self.create_subscription(OccupancyGrid, '/map', self.on_map, 5)
        self.create_subscription(String, '/nav2/lifecycle_state', self.on_nav_state, 10)
        self.nav_state = 'unknown'

        self.pub_ready = self.create_publisher(Bool, '/slam/ready', 10)
        self.pub_hb = self.create_publisher(Float32, '/heartbeat/slam', 10)
        self.pub_status = self.create_publisher(String, '/slam/status', 10)
        self.create_timer(1.0 / max(1.0, self.hz), self.on_timer)

    def on_map(self, m: OccupancyGrid):
        self.last_map_mono = time.monotonic()
        self.map_res = float(m.info.resolution)
        self.map_w = int(m.info.width)
        self.map_h = int(m.info.height)

    def on_nav_state(self, m: String):
        self.nav_state = str(m.data).lower()

    def on_timer(self):
        now = time.monotonic()
        age = now - self.last_map_mono if self.last_map_mono > 0.0 else 1e9
        map_ok = age <= self.map_timeout and self.map_w > 0 and self.map_h > 0 and self.map_res > 0.0
        nav_hint = self.nav_state in ('active', 'running', 'unknown')
        ready = map_ok and nav_hint
        self.pub_ready.publish(Bool(data=ready))
        self.pub_hb.publish(Float32(data=now))
        self.pub_status.publish(String(data=f'map_ok={map_ok},age_s={age:.3f},map={self.map_w}x{self.map_h}@{self.map_res:.3f},nav_state={self.nav_state}'))


class Nav2LifecycleGuardNode(Node):
    def __init__(self):
        super().__init__('nav2_lifecycle_guard')
        self.declare_parameter('plan_timeout_s', 1.0)
        self.plan_timeout = float(self.get_parameter('plan_timeout_s').value)
        self.last_plan = 0.0
        self.nav_state = 'unknown'
        self.slam_ready = True
        self.pub_ready = self.create_publisher(Bool, '/nav2/ready', 10)
        self.pub_hb = self.create_publisher(Float32, '/heartbeat/nav2', 10)
        self.create_subscription(NavPath, '/plan', self.on_plan, 10)
        self.create_subscription(String, '/nav2/lifecycle_state', self.on_state, 10)
        self.create_subscription(Bool, '/slam/ready', lambda m: setattr(self, 'slam_ready', bool(m.data)), 10)
        self.create_timer(0.2, self.on_timer)

    def on_plan(self, _: NavPath):
        self.last_plan = time.monotonic()

    def on_state(self, m: String):
        self.nav_state = str(m.data).lower()
    def on_timer(self):
        age_ok = (time.monotonic() - self.last_plan) < self.plan_timeout
        state_ok = (self.nav_state in ('active', 'running', 'unknown'))
        ready = age_ok and state_ok and self.slam_ready
        self.pub_ready.publish(Bool(data=ready))
        self.pub_hb.publish(Float32(data=time.monotonic()))


class FailSafeWatchdogNode(Node):
    def __init__(self):
        super().__init__('failsafe_watchdog')
        self.declare_parameter('hb_timeout_s', 0.8)
        self.timeout = float(self.get_parameter('hb_timeout_s').value)
        self.hb = {'vision': 0.0, 'risk': 0.0, 'bridge': 0.0, 'nav2': 0.0, 'slam': 0.0}
        self.nav2_ready = False
        self.manual_estop = False

        self.create_subscription(Float32, '/heartbeat/vision', lambda m: self._beat('vision', m), 20)
        self.create_subscription(Float32, '/heartbeat/risk', lambda m: self._beat('risk', m), 20)
        self.create_subscription(Float32, '/heartbeat/bridge', lambda m: self._beat('bridge', m), 20)
        self.create_subscription(Float32, '/heartbeat/nav2', lambda m: self._beat('nav2', m), 20)
        self.create_subscription(Float32, '/heartbeat/slam', lambda m: self._beat('slam', m), 20)
        self.create_subscription(Bool, '/nav2/ready', lambda m: setattr(self, 'nav2_ready', bool(m.data)), 20)
        self.create_subscription(Bool, '/manual_estop', lambda m: setattr(self, 'manual_estop', bool(m.data)), 20)

        self.pub_fs = self.create_publisher(Bool, '/failsafe/estop', 20)
        self.pub_brake = self.create_publisher(Bool, '/brake_override', 20)
        self.pub_mode = self.create_publisher(String, '/failsafe/mode', 10)
        self.create_timer(0.05, self.on_timer)

    def _beat(self, key: str, m: Float32):
        self.hb[key] = float(m.data)
    def on_timer(self):
        now = time.monotonic()
        stale = [k for k, t in self.hb.items() if (now - t) > self.timeout]
        tripped = self.manual_estop or (not self.nav2_ready) or (len(stale) > 0)
        self.pub_fs.publish(Bool(data=tripped))
        self.pub_brake.publish(Bool(data=tripped))
        mode = 'FAILSAFE' if tripped else 'RUN'
        if stale:
            mode += ':' + '|'.join(stale)
        self.pub_mode.publish(String(data=mode))




class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor')
        self.declare_parameter('report_hz', 2.0)
        self.hz = float(self.get_parameter('report_hz').value)

        self.last = {'vision': 0.0, 'risk': 0.0, 'bridge': 0.0, 'nav2': 0.0, 'slam': 0.0}
        self.loop_dt = {'ekf': 0.0, 'mpc': 0.0}
        self.solve_dt = {'mpc': 0.0}

        self.create_subscription(Float32, '/heartbeat/vision', lambda m: self._hb('vision', m), 20)
        self.create_subscription(Float32, '/heartbeat/risk', lambda m: self._hb('risk', m), 20)
        self.create_subscription(Float32, '/heartbeat/bridge', lambda m: self._hb('bridge', m), 20)
        self.create_subscription(Float32, '/heartbeat/nav2', lambda m: self._hb('nav2', m), 20)
        self.create_subscription(Float32, '/heartbeat/slam', lambda m: self._hb('slam', m), 20)
        self.create_subscription(Float32, '/perf/ekf_loop_dt_ms', lambda m: self._dt('ekf', m), 20)
        self.create_subscription(Float32, '/perf/mpc_loop_dt_ms', lambda m: self._dt('mpc', m), 20)
        self.create_subscription(Float32, '/perf/mpc_solve_ms', lambda m: self._solve('mpc', m), 20)

        self.pub = self.create_publisher(String, '/perf/summary', 10)
        self.create_timer(1.0 / max(0.5, self.hz), self.on_timer)

    def _hb(self, key: str, m: Float32):
        self.last[key] = float(m.data)

    def _dt(self, key: str, m: Float32):
        self.loop_dt[key] = float(m.data)

    def _solve(self, key: str, m: Float32):
        self.solve_dt[key] = float(m.data)

    def on_timer(self):
        now = time.monotonic()
        ages = {k: now - t for k, t in self.last.items()}
        msg = f"hb_age={ages}, loop_dt_ms={self.loop_dt}, solve_dt_ms={self.solve_dt}"
        self.pub.publish(String(data=msg))


class SafetySupervisorNode(Node):
    """Independent safety layer: merges risk/watchdog/bms/manual into final estop."""
    def __init__(self):
        super().__init__('safety_supervisor')
        self.risk_estop = False
        self.failsafe_estop = False
        self.manual_estop = False
        self.bms_critical = False

        self.create_subscription(Bool, '/emergency_stop', lambda m: setattr(self, 'risk_estop', bool(m.data)), 20)
        self.create_subscription(Bool, '/failsafe/estop', lambda m: setattr(self, 'failsafe_estop', bool(m.data)), 20)
        self.create_subscription(Bool, '/manual_estop', lambda m: setattr(self, 'manual_estop', bool(m.data)), 20)
        self.create_subscription(String, '/battery/status', self.on_bms, 10)

        self.pub_estop = self.create_publisher(Bool, '/safety/estop', 20)
        self.pub_brake = self.create_publisher(Bool, '/safety/brake_override', 20)
        self.pub_mode = self.create_publisher(String, '/safety/mode', 10)
        self.create_timer(0.02, self.on_timer)

    def on_bms(self, m: String):
        self.bms_critical = str(m.data).upper() == 'CRITICAL'

    def on_timer(self):
        estop = self.risk_estop or self.failsafe_estop or self.manual_estop or self.bms_critical
        self.pub_estop.publish(Bool(data=estop))
        self.pub_brake.publish(Bool(data=estop))
        self.pub_mode.publish(String(data='ESTOP' if estop else 'RUN'))




class RiskNode(Node):
    def __init__(self):
        super().__init__('risk_node')
        self.declare_parameter('ttc_slow_s', 1.2)
        self.declare_parameter('ttc_estop_s', 0.6)
        self.declare_parameter('curv_max', 1.5)
        self.declare_parameter('w_ttc', 0.5)
        self.declare_parameter('w_slip', 0.3)
        self.declare_parameter('w_curv', 0.2)
        self.declare_parameter('risk_beta', 0.25)
        self.declare_parameter('risk_beta_ttc_curv', 0.20)
        self.declare_parameter('hb_timeout_s', 0.7)
        self.declare_parameter('d_safe_m', 0.25)
        self.declare_parameter('slip_deadband_speed_mps', 0.30)

        self.ttc_slow = float(self.get_parameter('ttc_slow_s').value)
        self.ttc_estop = float(self.get_parameter('ttc_estop_s').value)
        self.curv_max = float(self.get_parameter('curv_max').value)
        self.w_ttc = float(self.get_parameter('w_ttc').value)
        self.w_slip = float(self.get_parameter('w_slip').value)
        self.w_curv = float(self.get_parameter('w_curv').value)
        self.risk_beta = float(self.get_parameter('risk_beta').value)
        self.risk_beta_ttc_curv = float(self.get_parameter('risk_beta_ttc_curv').value)
        self.hb_timeout = float(self.get_parameter('hb_timeout_s').value)
        self.d_safe = float(self.get_parameter('d_safe_m').value)
        self.slip_deadband_speed = float(self.get_parameter('slip_deadband_speed_mps').value)

        self.front, self.score, self.conf, self.v = 99.0, 0.0, 0.0, 0.0
        self.slip_l_f = self.slip_r_f = 0.0
        self.curv = 0.0
        self.risk_slow = False
        self.estop_latched = False
        self.last_vision_hb = 0.0

        self.create_subscription(Float32, '/ai/front_distance_m', lambda m: setattr(self, 'front', float(m.data)), 20)
        self.create_subscription(Float32, '/ai/vision_obstacle_score', lambda m: setattr(self, 'score', float(m.data)), 20)
        self.create_subscription(Float32, '/ai/vision_confidence', lambda m: setattr(self, 'conf', float(m.data)), 20)
        self.create_subscription(Float32, '/heartbeat/vision', lambda m: setattr(self, 'last_vision_hb', float(m.data)), 20)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'v', float(m.twist.twist.linear.x)), 20)
        self.create_subscription(Float32MultiArray, '/diag/slip', self.on_slip_diag, 20)
        self.create_subscription(Float32MultiArray, '/wheel_cmd', self.on_cmd, 20)

        self.pub_scale = self.create_publisher(Float32, '/ai_speed_scale', 20)
        self.pub_estop = self.create_publisher(Bool, '/emergency_stop', 20)
        self.pub_risk = self.create_publisher(Float32, '/ai/risk_index', 20)
        self.pub_hb = self.create_publisher(Float32, '/heartbeat/risk', 10)
        self.create_timer(0.05, self.on_timer)
        self.create_timer(0.2, lambda: self.pub_hb.publish(Float32(data=time.monotonic())))

    def on_slip_diag(self, m: Float32MultiArray):
        if len(m.data) >= 2:
            self.slip_l_f = float(m.data[0])
            self.slip_r_f = float(m.data[1])

    def on_cmd(self, m: Float32MultiArray):
        if len(m.data) >= 3:
            self.curv = abs(math.tan(math.radians(float(m.data[2]))) / 0.24)

    def on_timer(self):
        front_eff = max(self.front, 0.01)
        v_forward = max(self.v, 0.0)
        ttc = (front_eff - self.d_safe) / max(v_forward, 0.05) if v_forward > 0.05 else 999.0

        r_ttc = 0.0 if v_forward < 0.2 else clamp((self.ttc_slow - ttc) / self.ttc_slow, 0.0, 1.0)
        r_slip = clamp(max(abs(self.slip_l_f), abs(self.slip_r_f)) / 0.4, 0.0, 1.0)
        r_curv = clamp(abs(self.curv) / self.curv_max, 0.0, 1.0)

        risk = self.w_ttc * r_ttc + self.w_slip * r_slip + self.w_curv * r_curv
        risk += self.risk_beta * r_slip * r_curv
        risk += self.risk_beta_ttc_curv * r_ttc * r_curv
        if ttc <= 0.0:
            risk = 1.0

        estop_raw = (v_forward > 0.05) and ((ttc <= 0.0) or (ttc < self.ttc_estop))
        scale = 1.0
        if estop_raw:
            scale = 0.0
        elif (v_forward > 0.05) and (ttc < self.ttc_slow):
            scale = (ttc / self.ttc_slow) ** 2

        scale = min(scale, 1.0 - 0.5 * clamp(self.score * self.conf, 0.0, 1.0))
        if (time.monotonic() - self.last_vision_hb) > self.hb_timeout:
            scale = min(scale, 0.35)

        if self.risk_slow:
            if risk < 0.65:
                self.risk_slow = False
        else:
            if risk > 0.80:
                self.risk_slow = True
        if self.risk_slow:
            scale = min(scale, 0.4)
        if risk > 0.95:
            estop_raw, scale = True, 0.0

        if self.estop_latched:
            if risk < 0.65 and ttc > self.ttc_slow:
                self.estop_latched = False
        elif estop_raw:
            self.estop_latched = True

        self.pub_scale.publish(Float32(data=clamp(scale, 0.0, 1.0)))
        self.pub_estop.publish(Bool(data=self.estop_latched))
        self.pub_risk.publish(Float32(data=clamp(risk, 0.0, 1.0)))




class SlipObserverNode(Node):
    def __init__(self):
        super().__init__('slip_observer_node')
        self.declare_parameter('track_width_m', 0.302)
        self.declare_parameter('slip_deadband_speed_mps', 0.30)
        self.tw = float(self.get_parameter('track_width_m').value)
        self.deadband = float(self.get_parameter('slip_deadband_speed_mps').value)
        if self.deadband < 0.1:
            self.get_logger().warn('slip_deadband_speed_mps very low; low-speed noise may cause false slip')

        self.v = 0.0
        self.yaw_rate = 0.0
        self.slip_l_f = 0.0
        self.slip_r_f = 0.0

        self.create_subscription(Odometry, '/odom', self.on_odom, 30)
        self.create_subscription(Float32MultiArray, '/esp32/telemetry', self.on_tlm, 30)
        self.pub = self.create_publisher(Float32MultiArray, '/diag/slip', 20)

    def on_odom(self, m: Odometry):
        self.v = float(m.twist.twist.linear.x)
        self.yaw_rate = float(m.twist.twist.angular.z)

    def on_tlm(self, m: Float32MultiArray):
        if len(m.data) < 8:
            return
        vl = float(m.data[2]); vr = float(m.data[3])
        vexp_l = self.v - 0.5 * self.yaw_rate * self.tw
        vexp_r = self.v + 0.5 * self.yaw_rate * self.tw
        if abs(self.v) < self.deadband:
            slip_l = 0.0
            slip_r = 0.0
        else:
            slip_l = (vl - vexp_l) / max(abs(vexp_l), 0.2)
            slip_r = (vr - vexp_r) / max(abs(vexp_r), 0.2)
        self.slip_l_f = 0.8 * self.slip_l_f + 0.2 * slip_l
        self.slip_r_f = 0.8 * self.slip_r_f + 0.2 * slip_r
        self.pub.publish(Float32MultiArray(data=[self.slip_l_f, self.slip_r_f]))


class EkfOdomNode(Node):
    """2D EKF (Jacobian-based) for [x, y, yaw, v, gyro_bias]."""
    def __init__(self):
        super().__init__('ekf_odom_node')
        self.declare_parameter('gps_lever_arm_x_m', 0.12)
        self.declare_parameter('gps_lever_arm_y_m', 0.0)
        self.declare_parameter('q_xy', 0.08)
        self.declare_parameter('q_yaw', 0.05)
        self.declare_parameter('q_v', 0.2)
        self.declare_parameter('q_bias', 0.005)
        self.declare_parameter('r_v', 0.08)
        self.declare_parameter('r_yaw_rate', 0.12)
        self.declare_parameter('q_speed_gain', 0.35)
        self.declare_parameter('q_yaw_rate_gain', 0.45)
        self.declare_parameter('r_gps_min', 0.08)
        self.declare_parameter('max_sensor_age_s', 0.35)
        self.declare_parameter('gps_dropout_q_scale', 2.5)
        self.lever_x = float(self.get_parameter('gps_lever_arm_x_m').value)
        self.lever_y = float(self.get_parameter('gps_lever_arm_y_m').value)
        self.q_xy = float(self.get_parameter('q_xy').value)
        self.q_yaw = float(self.get_parameter('q_yaw').value)
        self.q_v = float(self.get_parameter('q_v').value)
        self.q_bias = float(self.get_parameter('q_bias').value)
        self.r_v = float(self.get_parameter('r_v').value)
        self.r_yaw_rate = float(self.get_parameter('r_yaw_rate').value)
        self.q_speed_gain = float(self.get_parameter('q_speed_gain').value)
        self.q_yaw_rate_gain = float(self.get_parameter('q_yaw_rate_gain').value)
        self.r_gps_min = float(self.get_parameter('r_gps_min').value)
        self.max_sensor_age = float(self.get_parameter('max_sensor_age_s').value)
        self.gps_dropout_q_scale = float(self.get_parameter('gps_dropout_q_scale').value)

        self.get_logger().info(f'GPS lever-arm set to x={self.lever_x:.3f} m, y={self.lever_y:.3f} m relative to base_link')

        self.x = np.zeros((5, 1), dtype=np.float64)
        self.P = np.diag([2.0, 2.0, 0.5, 0.5, 0.1]).astype(np.float64)
        self.last_ts = time.monotonic()
        self.gps_origin: tuple[float, float] | None = None
        self.last_wz = 0.0
        self.wheel_yaw_rate = 0.0
        self.scan_dx = 0.0
        self.scan_dy = 0.0
        self.scan_dyaw = 0.0
        self.ready = False
        self.last_imu_mono = 0.0
        self.last_wheel_mono = 0.0
        self.last_gps_mono = 0.0

        self.odom = Odometry()
        self.pub = self.create_publisher(Odometry, '/odom', 30)
        self.pub_perf = self.create_publisher(Float32, '/perf/ekf_loop_dt_ms', 10)
        self.pub_mode = self.create_publisher(String, '/ekf/fusion_mode', 10)
        self.create_subscription(Odometry, '/wheel/odom', self.on_wheel_odom, 50)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 100)
        self.create_subscription(NavSatFix, '/gps/fix', self.on_gps, 20)
        self.create_subscription(Float32MultiArray, '/scan_match/delta', self.on_scan_match, 20)
        self.create_timer(0.02, self.on_timer)

    def predict(self, dt: float):
        px, py, yaw, v, b = [float(self.x[i, 0]) for i in range(5)]
        wz = self.last_wz
        yaw_rate = wz - b

        nx = px + v * math.cos(yaw) * dt
        ny = py + v * math.sin(yaw) * dt
        nyaw = wrap_pi(yaw + yaw_rate * dt)
        self.x[0, 0] = nx
        self.x[1, 0] = ny
        self.x[2, 0] = nyaw

        F = np.eye(5, dtype=np.float64)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt
        F[2, 4] = -dt

        speed_gain = 1.0 + self.q_speed_gain * abs(v)
        yaw_gain = 1.0 + self.q_yaw_rate_gain * abs(yaw_rate)
        q_xy_eff = self.q_xy * speed_gain
        q_yaw_eff = self.q_yaw * yaw_gain
        q_v_eff = self.q_v * speed_gain
        q_b_eff = self.q_bias * (1.0 + 0.25 * abs(yaw_rate))
        gps_age = time.monotonic() - self.last_gps_mono if self.last_gps_mono > 0.0 else 1e9
        if gps_age > self.max_sensor_age:
            scale = min(self.gps_dropout_q_scale, 1.0 + 0.8 * (gps_age - self.max_sensor_age))
            q_xy_eff *= scale
            q_yaw_eff *= min(self.gps_dropout_q_scale, 1.0 + 0.5 * (gps_age - self.max_sensor_age))

        Q = np.diag([
            q_xy_eff * dt * dt,
            q_xy_eff * dt * dt,
            q_yaw_eff * dt * dt,
            q_v_eff * dt,
            q_b_eff * dt,
        ])
        self.P = F @ self.P @ F.T + Q

    def update_linear(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        y = z - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2, 0] = wrap_pi(float(self.x[2, 0]))
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P

    def on_wheel_odom(self, m: Odometry):
        now_msg = stamp_to_sec(m.header.stamp)
        now_clock = stamp_to_sec(self.get_clock().now().to_msg())
        if now_msg > 0.0 and (now_clock - now_msg) > self.max_sensor_age:
            return
        self.last_wheel_mono = time.monotonic()
        v_meas = float(m.twist.twist.linear.x)
        self.wheel_yaw_rate = float(m.twist.twist.angular.z)
        z = np.array([[v_meas]], dtype=np.float64)
        H = np.array([[0, 0, 0, 1, 0]], dtype=np.float64)
        R = np.array([[self.r_v]], dtype=np.float64)
        self.update_linear(z, H, R)
        if abs(self.wheel_yaw_rate) > 0.02:
            bias_obs = self.last_wz - self.wheel_yaw_rate
            z_b = np.array([[bias_obs]], dtype=np.float64)
            H_b = np.array([[0, 0, 0, 0, 1]], dtype=np.float64)
            R_b = np.array([[self.r_yaw_rate]], dtype=np.float64)
            self.update_linear(z_b, H_b, R_b)
        if not self.ready:
            self.x[0, 0] = float(m.pose.pose.position.x)
            self.x[1, 0] = float(m.pose.pose.position.y)
            self.x[2, 0] = quat_to_yaw(float(m.pose.pose.orientation.x), float(m.pose.pose.orientation.y), float(m.pose.pose.orientation.z), float(m.pose.pose.orientation.w))
            self.ready = True

    def on_imu(self, m: Imu):
        now_msg = stamp_to_sec(m.header.stamp)
        now_clock = stamp_to_sec(self.get_clock().now().to_msg())
        if now_msg > 0.0 and (now_clock - now_msg) > self.max_sensor_age:
            return
        self.last_imu_mono = time.monotonic()
        self.last_wz = float(m.angular_velocity.z)

    def on_scan_match(self, m: Float32MultiArray):
        if len(m.data) >= 4:
            self.scan_dx = float(m.data[0])
            self.scan_dy = float(m.data[1])
            self.scan_dyaw = float(m.data[2])

    def on_gps(self, m: NavSatFix):
        now_msg = stamp_to_sec(m.header.stamp)
        now_clock = stamp_to_sec(self.get_clock().now().to_msg())
        if now_msg > 0.0 and (now_clock - now_msg) > self.max_sensor_age:
            return
        if not (math.isfinite(m.latitude) and math.isfinite(m.longitude)):
            return
        if self.gps_origin is None:
            self.gps_origin = (float(m.latitude), float(m.longitude))
            self.last_gps_mono = time.monotonic()
            return
        gx, gy = geodetic_to_local_xy(self.gps_origin[0], self.gps_origin[1], float(m.latitude), float(m.longitude))
        yaw = float(self.x[2, 0])
        bx = gx - (self.lever_x * math.cos(yaw) - self.lever_y * math.sin(yaw))
        by = gy - (self.lever_x * math.sin(yaw) + self.lever_y * math.cos(yaw))
        z = np.array([[bx + self.scan_dx], [by + self.scan_dy]], dtype=np.float64)
        H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float64)
        cov_x = float(m.position_covariance[0]) if len(m.position_covariance) >= 1 else 0.8
        cov_y = float(m.position_covariance[4]) if len(m.position_covariance) >= 5 else cov_x
        cov_x = max(cov_x, self.r_gps_min)
        cov_y = max(cov_y, self.r_gps_min)
        status = getattr(m.status, 'status', 0)
        cov_type = int(getattr(m, 'position_covariance_type', 0))
        if status < 0:
            cov_x *= 4.0
            cov_y *= 4.0
        if cov_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
            cov_x *= 3.0
            cov_y *= 3.0
        R = np.diag([cov_x, cov_y]).astype(np.float64)
        self.update_linear(z, H, R)
        self.last_gps_mono = time.monotonic()

        # scan yaw correction as pseudo measurement
        z_yaw = np.array([[wrap_pi(float(self.x[2, 0]) + self.scan_dyaw)]], dtype=np.float64)
        H_yaw = np.array([[0, 0, 1, 0, 0]], dtype=np.float64)
        R_yaw = np.array([[0.08]], dtype=np.float64)
        self.update_linear(z_yaw, H_yaw, R_yaw)
    def on_timer(self):
        now = time.monotonic()
        dt = clamp(now - self.last_ts, 1e-3, 0.1)
        self.last_ts = now
        self.predict(dt)
        self.pub_perf.publish(Float32(data=dt * 1000.0))
        gps_age = time.monotonic() - self.last_gps_mono if self.last_gps_mono > 0.0 else 1e9
        imu_age = time.monotonic() - self.last_imu_mono if self.last_imu_mono > 0.0 else 1e9
        if gps_age > self.max_sensor_age:
            mode = 'IMU_ODOM_DOMINANT'
        elif imu_age > self.max_sensor_age:
            mode = 'GPS_DOMINANT_FALLBACK'
        else:
            mode = 'FUSION_NOMINAL'
        self.pub_mode.publish(String(data=mode))

        yaw = float(self.x[2, 0])
        cy = math.cos(0.5 * yaw)
        sy = math.sin(0.5 * yaw)
        self.odom.header.stamp = self.get_clock().now().to_msg()
        self.odom.header.frame_id = 'odom'
        self.odom.child_frame_id = 'base_link'
        self.odom.pose.pose.position.x = float(self.x[0, 0])
        self.odom.pose.pose.position.y = float(self.x[1, 0])
        self.odom.pose.pose.orientation.z = sy
        self.odom.pose.pose.orientation.w = cy
        self.odom.twist.twist.linear.x = float(self.x[3, 0])
        self.odom.twist.twist.angular.z = float(self.last_wz - self.x[4, 0])

        p = [0.0] * 36
        t = [0.0] * 36
        p[0] = float(self.P[0, 0]); p[7] = float(self.P[1, 1]); p[35] = float(self.P[2, 2])
        t[0] = float(self.P[3, 3]); t[35] = float(self.r_yaw_rate + self.P[4, 4])
        self.odom.pose.covariance = p
        self.odom.twist.covariance = t
        self.pub.publish(self.odom)


class TfChainNode(Node):
    def __init__(self):
        super().__init__('tf_chain_node')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('laser_x', 0.12)
        self.declare_parameter('laser_y', 0.0)
        self.declare_parameter('publish_map_to_odom', False)

        self.laser_frame = str(self.get_parameter('laser_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.publish_map_to_odom = bool(self.get_parameter('publish_map_to_odom').value)

        self.br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)
        self.publish_static()
        self.create_subscription(Odometry, '/odom', self.on_odom, 30)

    def publish_static(self):
        now = self.get_clock().now().to_msg()
        static_tfs = []
        if self.publish_map_to_odom:
            t_map = TransformStamped()
            t_map.header.stamp = now
            t_map.header.frame_id = self.map_frame
            t_map.child_frame_id = self.odom_frame
            t_map.transform.rotation.w = 1.0
            static_tfs.append(t_map)

        t_laser = TransformStamped()
        t_laser.header.stamp = now
        t_laser.header.frame_id = self.base_frame
        t_laser.child_frame_id = self.laser_frame
        t_laser.transform.translation.x = float(self.get_parameter('laser_x').value)
        t_laser.transform.translation.y = float(self.get_parameter('laser_y').value)
        t_laser.transform.rotation.w = 1.0
        static_tfs.append(t_laser)
        self.static_br.sendTransform(static_tfs)

    def on_odom(self, m: Odometry):
        t = TransformStamped()
        t.header.stamp = m.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = float(m.pose.pose.position.x)
        t.transform.translation.y = float(m.pose.pose.position.y)
        t.transform.translation.z = float(m.pose.pose.position.z)
        t.transform.rotation = m.pose.pose.orientation
        self.br.sendTransform(t)


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    yaw_rate: float = 0.0
    has_odom: bool = False


class AutonomyNode(Node):
    def __init__(self):
        super().__init__('mpc_local_controller')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('base_speed_mps', 1.0)
        self.declare_parameter('goal_tolerance_m', 0.2)
        self.declare_parameter('max_steer_deg', 25.0)
        self.declare_parameter('wheel_base_m', 0.24)
        self.declare_parameter('track_width_m', 0.302)
        self.declare_parameter('a_lat_max', 2.0)
        self.declare_parameter('mu_friction', 0.6)
        self.declare_parameter('slip_mu_gain', 0.30)
        self.declare_parameter('a_long_accel_max', 1.0)
        self.declare_parameter('a_long_decel_max', 1.6)
        self.declare_parameter('speed_ramp_alpha', 0.2)
        self.declare_parameter('lookahead_k1', 0.35)
        self.declare_parameter('lookahead_k2', 0.9)
        self.declare_parameter('steer_rate_deg_s', 120.0)
        self.declare_parameter('dwa_v_samples', 5)
        self.declare_parameter('dwa_steer_samples', 9)
        self.declare_parameter('data_stale_timeout_s', 0.35)
        self.declare_parameter('mpc_time_budget_ms', 8.0)

        self.hz = float(self.get_parameter('control_hz').value)
        self.base_speed = float(self.get_parameter('base_speed_mps').value)
        self.goal_tol = float(self.get_parameter('goal_tolerance_m').value)
        self.max_steer = float(self.get_parameter('max_steer_deg').value)
        self.wb = float(self.get_parameter('wheel_base_m').value)
        if self.wb < 0.15 or self.wb > 0.6:
            self.get_logger().warn('wheel_base_m outside expected range for this platform; Ackermann turning may be inaccurate')
        self.tw = float(self.get_parameter('track_width_m').value)
        self.a_lat_max = float(self.get_parameter('a_lat_max').value)
        self.mu_friction = float(self.get_parameter('mu_friction').value)
        self.slip_mu_gain = float(self.get_parameter('slip_mu_gain').value)
        self.a_long_accel_max = float(self.get_parameter('a_long_accel_max').value)
        self.a_long_decel_max = float(self.get_parameter('a_long_decel_max').value)
        self.speed_ramp_alpha = float(self.get_parameter('speed_ramp_alpha').value)
        self.k1 = float(self.get_parameter('lookahead_k1').value)
        self.k2 = float(self.get_parameter('lookahead_k2').value)
        self.steer_rate = float(self.get_parameter('steer_rate_deg_s').value)
        self.dwa_v_samples = int(self.get_parameter('dwa_v_samples').value)
        self.dwa_steer_samples = int(self.get_parameter('dwa_steer_samples').value)
        self.data_stale_timeout = float(self.get_parameter('data_stale_timeout_s').value)
        self.mpc_budget_ms = float(self.get_parameter('mpc_time_budget_ms').value)

        self.state = RobotState()
        self.goal: PoseStamped | None = None
        self.path_points: list[tuple[float, float]] = []
        self.gps_origin: tuple[float, float] | None = None
        self._gps_samples: list[tuple[float, float]] = []

        self.estop, self.scale, self.risk, self.link_lat_ms = False, 1.0, 0.0, 0.0
        self.cost_scale = 1.0
        self.nav2_ready = True
        self.brake_override = False
        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.slip_mag = 0.0
        self.safe_slow = False
        self.last_control_ts = time.monotonic()
        self.last_odom_mono = 0.0
        self.last_plan_mono = 0.0
        self.last_imu_ts = 0.0
        self.imu_yaw_rate = 0.0
        self.local_map: OccupancyGrid | None = None
        self.local_obs_hint = np.zeros(6, dtype=np.float64)

        self.create_subscription(PoseStamped, '/goal_pose', lambda m: setattr(self, 'goal', m), 10)
        self.create_subscription(NavSatFix, '/goal_gps', self.on_goal_gps, 10)
        self.create_subscription(NavSatFix, '/gps/fix', self.on_gps, 20)
        self.create_subscription(NavPath, '/plan', self.on_path, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 30)
        self.create_subscription(Bool, '/safety/brake_override', lambda m: setattr(self, 'brake_override', bool(m.data)), 20)
        self.create_subscription(Float32MultiArray, '/esp32/telemetry', self.on_tlm, 20)
        self.create_subscription(Bool, '/safety/estop', lambda m: setattr(self, 'estop', bool(m.data)), 20)
        self.create_subscription(Bool, '/nav2/ready', lambda m: setattr(self, 'nav2_ready', bool(m.data)), 20)
        self.create_subscription(Float32, '/ai_speed_scale', lambda m: setattr(self, 'scale', clamp(float(m.data), 0.0, 1.0)), 20)
        self.create_subscription(Float32, '/ai/risk_index', lambda m: setattr(self, 'risk', clamp(float(m.data), 0.0, 1.0)), 20)
        self.create_subscription(Float32, '/esp32/link_latency_ms', lambda m: setattr(self, 'link_lat_ms', float(m.data)), 20)
        self.create_subscription(Float32, '/local_cost_scale', lambda m: setattr(self, 'cost_scale', clamp(float(m.data), 0.1, 1.0)), 20)
        self.create_subscription(OccupancyGrid, '/local_costmap', lambda m: setattr(self, 'local_map', m), 10)
        self.create_subscription(Float32MultiArray, '/local/obstacle_hint', self.on_obstacle_hint, 20)

        self.pub_cmd = self.create_publisher(Float32MultiArray, '/wheel_cmd', 20)
        self.pub_perf = self.create_publisher(Float32, '/perf/mpc_loop_dt_ms', 10)
        self.pub_solve = self.create_publisher(Float32, '/perf/mpc_solve_ms', 10)
        self.pub_mode = self.create_publisher(String, '/ai/mode', 10)
        self.create_timer(1.0 / self.hz, self.on_timer)

    def on_gps(self, m: NavSatFix):
        if self.gps_origin is None and math.isfinite(m.latitude) and math.isfinite(m.longitude):
            self._gps_samples.append((float(m.latitude), float(m.longitude)))
            if len(self._gps_samples) >= 10:
                lat = sum(x for x, _ in self._gps_samples[-10:]) / 10.0
                lon = sum(y for _, y in self._gps_samples[-10:]) / 10.0
                self.gps_origin = (lat, lon)

    def on_goal_gps(self, m: NavSatFix):
        if self.gps_origin is None:
            return
        x, y = geodetic_to_local_xy(self.gps_origin[0], self.gps_origin[1], float(m.latitude), float(m.longitude))
        g = PoseStamped()
        g.header.frame_id = 'map'
        g.pose.position.x = x
        g.pose.position.y = y
        g.pose.orientation.w = 1.0
        self.goal = g

    def on_path(self, m: NavPath):
        self.path_points = [(float(p.pose.position.x), float(p.pose.position.y)) for p in m.poses]
        self.last_plan_mono = time.monotonic()

    def on_odom(self, m: Odometry):
        self.last_odom_mono = time.monotonic()
        self.state.x = float(m.pose.pose.position.x)
        self.state.y = float(m.pose.pose.position.y)
        self.state.yaw = quat_to_yaw(float(m.pose.pose.orientation.x), float(m.pose.pose.orientation.y), float(m.pose.pose.orientation.z), float(m.pose.pose.orientation.w))
        self.state.v = float(m.twist.twist.linear.x)
        self.state.yaw_rate = float(m.twist.twist.angular.z)
        self.state.has_odom = True

    def on_imu(self, m: Imu):
        self.imu_yaw_rate = float(m.angular_velocity.z)
        self.last_imu_ts = time.monotonic()

    def on_tlm(self, m: Float32MultiArray):
        if len(m.data) >= 8:
            self.slip_mag = max(abs(float(m.data[6])), abs(float(m.data[7])))

    def publish_cmd(self, vl: float, vr: float, sd: float, estop=False):
        self.pub_cmd.publish(Float32MultiArray(data=[vl, vr, sd, 1.0 if estop else 0.0]))

    def pick_target_point(self, lookahead_m: float) -> tuple[float, float]:
        if not self.path_points:
            if self.goal is None:
                return self.state.x, self.state.y
            return float(self.goal.pose.position.x), float(self.goal.pose.position.y)

        x, y = self.state.x, self.state.y
        nearest_idx = 0
        best = float('inf')
        for i, (px, py) in enumerate(self.path_points):
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best:
                best = d2
                nearest_idx = i
        for i in range(nearest_idx, len(self.path_points)):
            px, py = self.path_points[i]
            if math.hypot(px - x, py - y) >= lookahead_m:
                return px, py
        return self.path_points[-1]

    def solve_mpc_steer(self, tx: float, ty: float, speed: float, horizon_s: float = 0.8) -> float:
        dt = 0.1
        steps = max(3, int(horizon_s / dt))
        best_cost = float('inf')
        best_steer = 0.0
        for steer in np.linspace(-self.max_steer, self.max_steer, 13):
            x, y, yaw = self.state.x, self.state.y, self.state.yaw
            curv = math.tan(math.radians(float(steer))) / self.wb if abs(steer) > 1e-3 else 0.0
            for _ in range(steps):
                x += speed * math.cos(yaw) * dt
                y += speed * math.sin(yaw) * dt
                yaw += speed * curv * dt
            cte = math.hypot(tx - x, ty - y)
            steer_pen = 0.02 * (steer ** 2)
            cost = cte + steer_pen
            if cost < best_cost:
                best_cost = cost
                best_steer = float(steer)
        return best_steer


    def on_obstacle_hint(self, m: Float32MultiArray):
        if len(m.data) >= 6:
            self.local_obs_hint = np.array(m.data[:6], dtype=np.float64)

    def cost_at_local(self, x_local: float, y_local: float) -> float:
        if self.local_map is None or self.local_map.info.width == 0:
            return 0.0
        info = self.local_map.info
        ix = int((x_local - info.origin.position.x) / max(info.resolution, 1e-3))
        iy = int((y_local - info.origin.position.y) / max(info.resolution, 1e-3))
        if ix < 0 or iy < 0 or ix >= int(info.width) or iy >= int(info.height):
            return 100.0
        k = iy * int(info.width) + ix
        if k < 0 or k >= len(self.local_map.data):
            return 100.0
        return float(max(self.local_map.data[k], 0))

    def dwa_adjust(self, base_v: float, base_steer: float, tx: float, ty: float) -> tuple[float, float]:
        v_min = max(0.0, self.v_cmd - self.a_long_decel_max * 0.2)
        v_max = min(base_v, self.v_cmd + self.a_long_accel_max * 0.2 + 0.4)
        if v_max < v_min:
            v_max = v_min
        v_candidates = np.linspace(v_min, v_max, max(2, self.dwa_v_samples))
        steer_window = self.steer_rate * 0.2
        s_lo = clamp(self.steer_cmd - steer_window, -self.max_steer, self.max_steer)
        s_hi = clamp(self.steer_cmd + steer_window, -self.max_steer, self.max_steer)
        s_candidates = np.linspace(s_lo, s_hi, max(3, self.dwa_steer_samples))
        best = (-1e9, base_v, base_steer)
        for v in v_candidates:
            for sd in s_candidates:
                curv = math.tan(math.radians(float(sd))) / self.wb if abs(sd) > 1e-3 else 0.0
                x = y = yaw = 0.0
                # short horizon rollout in base_link frame
                for _ in range(6):
                    x += v * math.cos(yaw) * 0.1
                    y += v * math.sin(yaw) * 0.1
                    yaw += v * curv * 0.1
                cost = self.cost_at_local(x, y)
                # semantic S strategy: bias to side-pass when obstacle exists
                side_bonus = 0.0
                if self.local_obs_hint[2] > 0.5 and x > 0.2:
                    pref_side = 1.0 if self.local_obs_hint[3] >= 0.0 else -1.0
                    side_align = 1.0 - min(1.0, abs((pref_side * 0.6) - y) / 1.2)
                    side_bonus = 12.0 * side_align
                gx = tx - self.state.x
                gy = ty - self.state.y
                heading = math.atan2(gy, gx) - self.state.yaw
                yaw_err = abs(wrap_pi(heading - yaw))
                score = (2.0 * v) - (0.06 * abs(sd - base_steer)) - (0.025 * cost) - (1.5 * yaw_err) + side_bonus
                if score > best[0]:
                    best = (score, float(v), float(sd))
        return best[1], best[2]

    def on_timer(self):
        if self.estop or self.brake_override:
            self.pub_mode.publish(String(data='ESTOP' if self.estop else 'BRAKE_OVERRIDE'))
            self.publish_cmd(0.0, 0.0, 0.0, True)
            return
        now = time.monotonic()
        if self.last_odom_mono > 0.0 and (now - self.last_odom_mono) > self.data_stale_timeout:
            self.pub_mode.publish(String(data='WAIT_ODOM_STALE'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return
        if self.last_plan_mono > 0.0 and (now - self.last_plan_mono) > self.data_stale_timeout:
            self.pub_mode.publish(String(data='WAIT_PLAN_STALE'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return
        if not self.state.has_odom:
            self.pub_mode.publish(String(data='INIT'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return
        if not self.nav2_ready:
            self.pub_mode.publish(String(data='WAIT_NAV2'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return
        if self.goal is None:
            self.pub_mode.publish(String(data='READY'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        goal_x = float(self.goal.pose.position.x)
        goal_y = float(self.goal.pose.position.y)
        dist = math.hypot(goal_x - self.state.x, goal_y - self.state.y)
        if dist < self.goal_tol:
            self.pub_mode.publish(String(data='READY'))
            self.publish_cmd(0.0, 0.0, 0.0)
            return

        dt = clamp(now - self.last_control_ts, 1e-3, 0.2)
        self.last_control_ts = now
        self.pub_perf.publish(Float32(data=dt * 1000.0))

        v_target = self.base_speed * self.scale * self.cost_scale
        if self.link_lat_ms > 120.0:
            v_target *= 0.7

        mu_eff = clamp(self.mu_friction * (1.0 - self.slip_mu_gain * self.slip_mag), 0.15, self.mu_friction)
        Ld = self.k1 + self.k2 * max(abs(self.state.v), 0.0)
        tx, ty = self.pick_target_point(Ld)
        if self.local_obs_hint[2] > 0.5:
            lat_bias = clamp(self.local_obs_hint[3], -1.0, 1.0)
            dxp = tx - self.state.x
            dyp = ty - self.state.y
            n = max(math.hypot(dxp, dyp), 1e-3)
            nx = -dyp / n
            ny = dxp / n
            tx += nx * lat_bias
            ty += ny * lat_bias

        alpha0 = wrap_pi(math.atan2(ty - self.state.y, tx - self.state.x) - self.state.yaw)
        delta0 = math.atan2(2.0 * self.wb * math.sin(alpha0), max(Ld, 1e-3))
        steer0 = clamp(math.degrees(delta0), -self.max_steer, self.max_steer)

        curvature0 = math.tan(math.radians(steer0)) / self.wb if abs(steer0) > 1e-3 else 0.0
        a_lat_limit = min(self.a_lat_max, mu_eff * 9.81)
        if abs(curvature0) > 1e-4:
            v_target = min(v_target, math.sqrt(a_lat_limit / abs(curvature0)))

        delay_s = max(self.link_lat_ms, 0.0) * 1e-3
        yaw_rate_est = self.imu_yaw_rate if (now - self.last_imu_ts) < 0.25 else v_target * curvature0
        pred_yaw = self.state.yaw + yaw_rate_est * delay_s
        alpha1 = wrap_pi(math.atan2(ty - self.state.y, tx - self.state.x) - pred_yaw)
        delta1 = math.atan2(2.0 * self.wb * math.sin(alpha1), max(Ld, 1e-3))
        pp_steer = clamp(math.degrees(delta1), -self.max_steer, self.max_steer)
        t_solve0 = time.monotonic()
        mpc_steer = self.solve_mpc_steer(tx, ty, max(self.state.v, 0.1))
        solve_ms = (time.monotonic() - t_solve0) * 1000.0
        self.pub_solve.publish(Float32(data=solve_ms))
        if solve_ms > self.mpc_budget_ms:
            v_target *= max(0.4, self.mpc_budget_ms / max(solve_ms, 1e-3))
        steer_des = clamp(0.6 * mpc_steer + 0.4 * pp_steer, -self.max_steer, self.max_steer)
        v_target, steer_des = self.dwa_adjust(v_target, steer_des, tx, ty)

        dsteer_max = self.steer_rate * dt
        self.steer_cmd = self.steer_cmd + clamp(steer_des - self.steer_cmd, -dsteer_max, dsteer_max)
        steer = clamp(self.steer_cmd, -self.max_steer, self.max_steer)
        curvature = math.tan(math.radians(steer)) / self.wb if abs(steer) > 1e-3 else 0.0

        a = clamp(self.speed_ramp_alpha, 0.01, 1.0)
        v_ramp = self.v_cmd + a * (v_target - self.v_cmd)

        a_lat = abs((self.v_cmd ** 2) * curvature)
        a_total_max = max(mu_eff * 9.81, 0.5)
        a_long_circle = max(0.2, a_total_max * math.sqrt(max(0.0, 1.0 - (a_lat / a_total_max) ** 2)))

        if v_ramp > self.v_cmd:
            dv = min(self.a_long_accel_max, a_long_circle) * dt
            self.v_cmd = self.v_cmd + clamp(v_ramp - self.v_cmd, 0.0, dv)
        else:
            dv = min(self.a_long_decel_max, a_long_circle + 0.5) * dt
            self.v_cmd = self.v_cmd - clamp(self.v_cmd - v_ramp, 0.0, dv)

        v = max(0.0, self.v_cmd)
        yr = 0.0 if abs(curvature) < 1e-4 else v * curvature
        vl = v - yr * self.tw * 0.5
        vr = v + yr * self.tw * 0.5

        if self.safe_slow:
            if self.risk < 0.35:
                self.safe_slow = False
        else:
            if self.risk > 0.55:
                self.safe_slow = True
        mode = 'SAFE_SLOW' if self.safe_slow else ('AUTONOMOUS_S' if self.local_obs_hint[2] > 0.5 else 'AUTONOMOUS')
        self.pub_mode.publish(String(data=mode))
        self.publish_cmd(vl, vr, steer)


class BridgeNode(Node):
    FRAME_TYPE_CMD = 0x02
    FRAME_TYPE_TLM = 0x12

    def __init__(self):
        super().__init__('bridge_node')
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('publish_hz', 100.0)
        self.declare_parameter('imu_lpf_cutoff_hz', 3.0)
        self.declare_parameter('imu_lpf_cutoff_gain', 0.6)
        self.declare_parameter('max_wheel_speed_mps', 2.0)
        self.declare_parameter('max_steer_deg', 25.0)
        self.declare_parameter('rx_mode', 'ascii')
        self.declare_parameter('esp32_watchdog_ms', 200)
        self.declare_parameter('esp32_ramp_down_ms', 300)

        self.ser = serial.Serial(str(self.get_parameter('serial_port').value),
                                 baudrate=int(self.get_parameter('baudrate').value), timeout=0)
        self.hz = float(self.get_parameter('publish_hz').value)
        self.cut_base = float(self.get_parameter('imu_lpf_cutoff_hz').value)
        self.cut_gain = float(self.get_parameter('imu_lpf_cutoff_gain').value)
        self.max_v = float(self.get_parameter('max_wheel_speed_mps').value)
        self.max_s = float(self.get_parameter('max_steer_deg').value)
        self.rx_mode = str(self.get_parameter('rx_mode').value).lower()
        self.esp32_watchdog_ms = int(self.get_parameter('esp32_watchdog_ms').value)
        self.esp32_ramp_ms = int(self.get_parameter('esp32_ramp_down_ms').value)

        self.seq, self.last_imu = 0, time.monotonic()
        self.cmd = {'vl': 0.0, 'vr': 0.0, 'sd': 0.0, 'yz': 0.0, 'ax': 0.0, 'es': False}
        self.brake_override = False
        self.sent: dict[int, float] = {}
        self.rx_buf = bytearray()

        self.create_subscription(Float32MultiArray, '/wheel_cmd', self.on_cmd, 20)
        self.create_subscription(Bool, '/safety/estop', lambda m: self.cmd.__setitem__('es', bool(m.data)), 20)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 30)
        self.create_subscription(Bool, '/safety/brake_override', lambda m: setattr(self, 'brake_override', bool(m.data)), 20)

        self.pub_fault = self.create_publisher(Int32, '/esp32/fault_code', 10)
        self.pub_tlm = self.create_publisher(Float32MultiArray, '/esp32/telemetry', 20)
        self.pub_lat = self.create_publisher(Float32, '/esp32/link_latency_ms', 20)
        self.pub_hb = self.create_publisher(Float32, '/heartbeat/bridge', 10)
        self.pub_contract = self.create_publisher(String, '/esp32/control_contract', 2)

        self.create_timer(1.0 / self.hz, self.on_tx)
        self.create_timer(0.01, self.on_rx)
        self.create_timer(0.2, lambda: self.pub_hb.publish(Float32(data=time.monotonic())))
        self.create_timer(1.0, self.publish_contract)


    def publish_contract(self):
        msg = f'watchdog_ms={self.esp32_watchdog_ms},ramp_down_ms={self.esp32_ramp_ms},ackermann=wheel_cmd(vl,vr,steer)'
        self.pub_contract.publish(String(data=msg))

    def adaptive_lpf(self, old: float, new: float, dt: float) -> float:
        speed = 0.5 * (abs(self.cmd['vl']) + abs(self.cmd['vr']))
        cutoff = max(0.2, self.cut_base + self.cut_gain * speed)
        rc = 1.0 / (2 * math.pi * cutoff)
        a = clamp(dt / (rc + dt), 0.0, 1.0)
        return old + a * (new - old)

    def on_cmd(self, m: Float32MultiArray):
        if len(m.data) < 3:
            return
        self.cmd['vl'] = clamp(float(m.data[0]), -self.max_v, self.max_v)
        self.cmd['vr'] = clamp(float(m.data[1]), -self.max_v, self.max_v)
        self.cmd['sd'] = clamp(float(m.data[2]), -self.max_s, self.max_s)
        if len(m.data) >= 4:
            self.cmd['es'] = bool(m.data[3] > 0.5)

    def on_imu(self, m: Imu):
        now = time.monotonic()
        dt = clamp(now - self.last_imu, 1e-3, 0.1)
        self.last_imu = now
        self.cmd['yz'] = self.adaptive_lpf(self.cmd['yz'], float(m.angular_velocity.z), dt)
        self.cmd['ax'] = self.adaptive_lpf(self.cmd['ax'], float(m.linear_acceleration.x), dt)

    def on_tx(self):
        cmd_es = bool(self.cmd['es']) or self.brake_override
        flags = (1 if cmd_es else 0) | (2 if self.brake_override else 0)
        vl = 0.0 if self.brake_override else self.cmd['vl']
        vr = 0.0 if self.brake_override else self.cmd['vr']
        payload = struct.pack('<HBfffff', self.seq, flags, vl, vr, self.cmd['sd'], self.cmd['yz'], self.cmd['ax'])
        length = len(payload)
        header = bytes([0xAA, 0x55, self.FRAME_TYPE_CMD, length])
        frame = header + payload + struct.pack('<H', crc16_ccitt(bytes([self.FRAME_TYPE_CMD, length]) + payload))
        self.ser.write(frame)

        now = time.monotonic()
        self.sent[self.seq] = now
        for k, t in list(self.sent.items()):
            if now - t > 0.5:
                del self.sent[k]
        self.seq = (self.seq + 1) & 0xFFFF

    def on_rx(self):
        while self.ser.in_waiting > 0:
            self.rx_buf.extend(self.ser.read(self.ser.in_waiting))
        if self.rx_mode == 'binary':
            self.parse_binary_rx()
        else:
            self.parse_ascii_rx()

    def parse_ascii_rx(self):
        while True:
            idx = self.rx_buf.find(b'\n')
            if idx < 0:
                break
            raw = self.rx_buf[:idx]
            del self.rx_buf[:idx + 1]
            line = raw.decode(errors='ignore').strip()
            if not line.startswith('TLM,'):
                continue
            p = line.split(',')
            if len(p) not in (9, 10):
                continue
            if len(p) == 10:
                try:
                    recv_crc = int(p[9], 16)
                except ValueError:
                    continue
                if recv_crc != crc16_ccitt(','.join(p[:9]).encode()):
                    continue
            try:
                seq = int(float(p[1])); err = int(float(p[2]))
                vals = [float(x) for x in p[3:9]]
            except ValueError:
                continue
            self.pub_fault.publish(Int32(data=err))
            self.pub_tlm.publish(Float32MultiArray(data=[float(seq), float(err)] + vals))
            t0 = self.sent.pop(seq, None)
            if t0 is not None:
                self.pub_lat.publish(Float32(data=(time.monotonic() - t0) * 1000.0))

        if len(self.rx_buf) > 8192:
            # keep tail only, avoid aggressive line-based wipe
            del self.rx_buf[:-1024]

    def parse_binary_rx(self):
        while True:
            if len(self.rx_buf) > 8192:
                sof_tail = self.rx_buf.rfind(b'\xAA\x55')
                if sof_tail > 0:
                    del self.rx_buf[:sof_tail]
                elif sof_tail < 0:
                    del self.rx_buf[:-1024]
            if len(self.rx_buf) < 6:
                break
            sof = self.rx_buf.find(b'\xAA\x55')
            if sof < 0:
                self.rx_buf.clear()
                break
            if sof > 0:
                del self.rx_buf[:sof]
            if len(self.rx_buf) < 6:
                break

            ftype = self.rx_buf[2]
            length = self.rx_buf[3]
            frame_len = 4 + length + 2
            if len(self.rx_buf) < frame_len:
                break

            frame = bytes(self.rx_buf[:frame_len])
            del self.rx_buf[:frame_len]
            payload = frame[4:4 + length]
            recv_crc = struct.unpack('<H', frame[-2:])[0]
            if recv_crc != crc16_ccitt(bytes([ftype, length]) + payload):
                continue
            if ftype != self.FRAME_TYPE_TLM or length != 27:
                continue

            try:
                seq, err, vl, vr, refl, refr, slipl, slipr = struct.unpack('<HBffffff', payload)
            except struct.error:
                continue
            self.pub_fault.publish(Int32(data=err))
            self.pub_tlm.publish(Float32MultiArray(data=[float(seq), float(err), vl, vr, refl, refr, slipl, slipr]))
            t0 = self.sent.pop(seq, None)
            if t0 is not None:
                self.pub_lat.publish(Float32(data=(time.monotonic() - t0) * 1000.0))


def main():
    if len(sys.argv) < 2:
        print('usage: unified_robot_stack_v24.py [vision|risk|mpc|bridge|ekf|slip|tf|scanmatch|costmap|mapbridge|nav2guard|watchdog|safety|perf]')
        sys.exit(1)
    mode = sys.argv[1].lower()
    rclpy.init()
    node = {'vision': VisionNode, 'risk': RiskNode, 'autonomy': AutonomyNode, 'mpc': AutonomyNode, 'bridge': BridgeNode, 'ekf': EkfOdomNode, 'slip': SlipObserverNode, 'tf': TfChainNode, 'scanmatch': ScanMatcherNode, 'costmap': LocalCostmapNode, 'mapbridge': SlamNav2BridgeNode, 'nav2guard': Nav2LifecycleGuardNode, 'watchdog': FailSafeWatchdogNode, 'safety': SafetySupervisorNode, 'perf': PerformanceMonitorNode}.get(mode)
    if node is None:
        print('invalid mode')
        sys.exit(2)
    n = node()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
