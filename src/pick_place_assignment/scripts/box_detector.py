#!/usr/bin/env python3
"""Detects cartons on the source pallet from the overhead depth image.

Pipeline: depth -> world points -> height map -> flat regions -> carton faces.
Touching cartons at the same height form one flat region; tile_region() splits it using
the known carton sizes. Results are only published once they have been seen at the same
pose for `stable_frames` scans, and nothing is updated while the arm is over the pallet.
"""
import json
import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def dominant_angle(mask):
    """Direction (rad, in (-pi/4, pi/4]) of the longest straight edges of a mask, modulo 90 deg."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 2.0, True)[:, 0, :].astype(float)
    if len(poly) < 3:
        return 0.0
    e = np.roll(poly, -1, axis=0) - poly
    s = (np.hypot(e[:, 0], e[:, 1]) * np.exp(4j * np.arctan2(e[:, 1], e[:, 0]))).sum()
    return float(np.angle(s) / 4.0)


def tile_region(mask, faces, cell, z_above=None, max_leftover=0.08, min_cover=0.9,
                max_tiles=8, max_nodes=6000):
    """Cover a merged flat region with carton-sized rectangles.

    mask    : bool grid (x = column, y = row), cells of `cell` metres
    faces   : {(short, long): carton_height}
    z_above : height of this region's top above the pallet (used to rank alternatives)
    Returns [] if no good tiling exists, else a list of dicts with the tile centre
    (grid coordinates), yaw, length, width, carton height and corners.

    Several tilings can cover the same flat region (e.g. 0.3+0.4 vs 0.4+0.3 in a row);
    depth alone cannot tell them apart. Candidates are ranked: stack height consistent
    with the carton height, then one carton orientation per layer, then more tiles.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    th = dominant_angle(mask)
    ct, st = math.cos(th), math.sin(th)
    cx, cy = xs.mean(), ys.mean()
    size = int(math.ceil(math.hypot(xs.max() - xs.min(), ys.max() - ys.min()))) + 8
    # aligned frame: u runs along (cos th, sin th), v along (-sin th, cos th)
    M = np.array([[ct, st, size / 2 - (ct * cx + st * cy)],
                  [-st, ct, size / 2 - (-st * cx + ct * cy)]])
    al = cv2.warpAffine(mask.astype(np.float32), M, (size, size), flags=cv2.INTER_LINEAR) > 0.5

    shapes = []  # (w, h, face_key) in aligned cells; w along u, h along v
    for (s_, l_) in faces:
        a_, b_ = int(round(s_ / cell)), int(round(l_ / cell))
        shapes += [(b_, a_, (s_, l_)), (a_, b_, (s_, l_))]
    shapes.sort(key=lambda t: -t[0] * t[1])

    kern = np.ones((5, 5), np.uint8)
    budget = int(max_leftover * al.sum())
    H, W = al.shape
    nodes = [0]
    sols = []

    def solve(rem, dropped, placed):
        nodes[0] += 1
        if nodes[0] > max_nodes or len(sols) >= 40:
            return
        opened = cv2.morphologyEx(rem.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
        dropped += int(rem.sum() - opened.sum())      # thin slivers = edge noise
        if dropped > budget:
            return
        rem = opened
        r_idx, c_idx = np.nonzero(rem)
        if len(r_idx) == 0:
            if placed:
                sols.append(placed)
            return
        if len(placed) >= max_tiles:
            return
        rows = np.nonzero(rem.sum(axis=1) >= 5)[0]      # top row that is a real edge, not noise
        if len(rows) == 0:
            if placed:
                sols.append(placed)
            return
        r0 = int(rows[0])
        cols = np.nonzero(rem[r0:r0 + 8].sum(axis=0) >= 3)[0]   # left edge, ignoring ragged corner cells
        if len(cols) == 0:
            return
        c0 = int(cols[0])
        for (w, h, key) in shapes:
            if r0 + h > H or c0 + w > W:
                continue
            if rem[r0:r0 + h, c0:c0 + w].mean() < min_cover:
                continue
            nxt = rem.copy()
            nxt[r0:r0 + h, c0:c0 + w] = False
            solve(nxt, dropped, placed + [(c0, r0, w, h, key)])

    solve(al.copy(), 0, [])
    if not sols:
        return []

    def score(sol):
        keys = {k for (_, _, _, _, k) in sol}
        inconsistent = 0
        if z_above is not None:
            inconsistent = sum(1 for (_, _, _, _, k) in sol
                               if abs(z_above - round(z_above / faces[k]) * faces[k]) > 0.03
                               or z_above < faces[k] - 0.03)
        return (inconsistent, len(keys), -len(sol))
    sol = min(sols, key=score)

    inv = np.array([[ct, -st], [st, ct]])
    out = []
    for (c0, r0, w, h, key) in sol:
        u, v = c0 + w / 2.0 - 0.5, r0 + h / 2.0 - 0.5   # tile centre in aligned cell indices
        p = inv @ np.array([u - size / 2.0, v - size / 2.0]) + np.array([cx, cy])
        yaw = th if w >= h else th + math.pi / 2
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        corners = np.array([p + inv @ np.array(o) for o in
                            [(-w / 2.0, -h / 2.0), (w / 2.0, -h / 2.0),
                             (w / 2.0, h / 2.0), (-w / 2.0, h / 2.0)]])
        out.append({'cx': float(p[0]), 'cy': float(p[1]), 'yaw': float(yaw),
                    'length': max(w, h) * cell, 'width': min(w, h) * cell,
                    'height': faces[key], 'corners': corners.astype(np.float32)})
    return out


def split_region(mask, cell, neck=0.03):
    """Split a flat region where cartons touch at a corner or thin neck.
    Erode to find one core per carton, then give every pixel to its nearest core."""
    m = mask.astype(np.uint8)
    r = max(2, int(round(neck / cell)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(m, k)
    n, _ = cv2.connectedComponents(core, connectivity=4)
    if n <= 2:
        return [mask]
    src = (core == 0).astype(np.uint8)
    _, lbl = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_CCOMP)
    parts = [(lbl == i) & mask for i in np.unique(lbl[core > 0])]
    return [q for q in parts if q.sum() > 20]


class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        d = self.declare_parameter
        d('world_frame', 'world')
        d('camera_frame', '')            # empty = use the frame_id of the depth image
        d('depth_topic', '/camera/depth_image')
        d('info_topic', '/camera/camera_info')
        d('roi_center', [0.80, -0.55])    # source pallet centre (x, y) in world
        d('roi_half_size', 0.55)         # X half-width [m]
        d('roi_half_size_y', 0.70)       # Y half-width [m]
        d('pallet_top', 0.15)            # height of the pallet surface [m]
        d('min_height', 0.05)            # ignore anything lower than this above the pallet
        d('carton_dims', [0.2, 0.3, 0.4])
        d('cell', 0.006)                 # height-map resolution [m]
        d('edge_tol', 0.01)              # height jump that separates two regions [m]
        d('size_tol', 0.025)             # allowed error on carton face size [m]
        d('min_fill', 0.88)              # how rectangular a top face must be
        d('min_area', 0.03)              # smallest region considered [m^2]
        d('arm_frames', ['J6_link', 'gripper_link', 'tcp'])
        d('arm_margin', 0.20)            # arm frame this close to the pallet area blocks detection [m]
        d('stable_frames', 3)            # scans a box must persist before it is published
        d('match_dist', 0.015)           # same box if centre moved less than this [m]
        d('match_yaw', 0.05)             # ... and rotated less than this [rad]
        d('target_frame', 'next_target_carton')   # TF child frame of the next pick
        d('robot_base_xy', [0.0, 0.0])   # tie-break: among equally high boxes pick the nearest to this
        g = lambda n: self.get_parameter(n).value
        self.world = g('world_frame')
        self.cam_frame = g('camera_frame')
        self.rc = g('roi_center')
        self.half = g('roi_half_size')
        self.half_y = g('roi_half_size_y')
        self.pallet_top = g('pallet_top')
        self.min_h = g('min_height')
        self.cell = g('cell')
        self.edge_tol = g('edge_tol')
        self.size_tol = g('size_tol')
        self.min_fill = g('min_fill')
        self.min_area = g('min_area')
        self.arm_frames = g('arm_frames')
        self.arm_margin = g('arm_margin')
        self.stable_frames = g('stable_frames')
        self.match_dist = g('match_dist')
        self.match_yaw = g('match_yaw')
        self.target_frame = g('target_frame')
        self.base_xy = g('robot_base_xy')
        self.last_json = None
        self.last_payload = None
        self.scan_id = 0
        self.tracks = []
        # Orientation memory: layer top height -> set of carton faces seen as a lone, unambiguous
        # rectangle in that layer. Used to resolve "one lying carton" vs "two standing cartons".
        self.layer_faces = {}
        a, b, c = sorted(g('carton_dims'))
        # (short side, long side) of the visible top face -> carton height in that pose
        self.faces = {(a, b): c, (a, c): b, (b, c): a}

        self.K = None
        self.uv = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(CameraInfo, g('info_topic'), self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self.on_depth,
                                 qos_profile_sensor_data)
        self.pub_json = self.create_publisher(String, '/perception/boxes_json', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/markers', 10)
        self.pub_debug = self.create_publisher(Image, '/perception/debug_image', 10)
        self.get_logger().info('box_detector ready, waiting for camera data')

    def on_info(self, msg):
        self.K = list(msg.k)

    def arm_over_pallet(self):
        for fr in self.arm_frames:
            try:
                p = self.tf_buffer.lookup_transform(
                    self.world, fr, rclpy.time.Time()).transform.translation
            except Exception:  # noqa: BLE001
                continue
            if (abs(p.x - self.rc[0]) < self.half + self.arm_margin and
                    abs(p.y - self.rc[1]) < self.half + self.arm_margin and
                    p.z > self.pallet_top):
                return fr
        return None

    def on_depth(self, msg):
        if self.K is None:
            return
        cam_frame = self.cam_frame or msg.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world, cam_frame, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'waiting for TF: {e}', throttle_duration_sec=2.0)
            return
        # the arm over the pallet is the closest surface and would be detected as cartons:
        # skip this frame and keep the last detections from when the view was clear
        fr = self.arm_over_pallet()
        if fr is not None:
            self.get_logger().info(f'arm ({fr}) over the pallet, holding last detections',
                                   throttle_duration_sec=2.0)
            if self.last_payload is not None:   # same boxes again, flagged so the picker can ignore it
                self.pub_json.publish(String(data=json.dumps(dict(self.last_payload, stale=True))))
            return
        q = tf.transform.rotation
        tr = tf.transform.translation
        R = quat_to_matrix(q.x, q.y, q.z, q.w)
        t = np.array([tr.x, tr.y, tr.z])

        # 1. depth image -> 3D points in the world frame
        depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        fx, fy, cx, cy = self.K[0], self.K[4], self.K[2], self.K[5]
        if self.uv is None or self.uv[0].shape != depth.shape:
            self.uv = np.meshgrid(np.arange(msg.width), np.arange(msg.height))
        u, v = self.uv
        ok = np.isfinite(depth) & (depth > 0.1)
        z = depth[ok]
        pc = np.stack([(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z], axis=1)  # optical
        if 'optical' in cam_frame:
            pl = pc
        else:
            # Gazebo sensor link frame: x forward, y left, z up
            pl = np.stack([pc[:, 2], -pc[:, 0], -pc[:, 1]], axis=1)
        pw = pl @ R.T + t

        # 2. crop to the pallet area, drop pallet surface and floor
        x0 = self.rc[0] - self.half
        y0 = self.rc[1] - self.half_y

        m = ((np.abs(pw[:, 0] - self.rc[0]) < self.half) &
             (np.abs(pw[:, 1] - self.rc[1]) < self.half_y) &
             (pw[:, 2] > self.pallet_top + self.min_h))
        pw = pw[m]

        nx = int(round(2 * self.half / self.cell))
        ny = int(round(2 * self.half_y / self.cell))

        dets = []
        hm = np.full((ny, nx), -np.inf)

        if len(pw) > 0:
            ix = np.clip(((pw[:, 0] - x0) / self.cell).astype(int), 0, nx - 1)
            iy = np.clip(((pw[:, 1] - y0) / self.cell).astype(int), 0, ny - 1)

            flat = np.full(ny * nx, -np.inf)
            np.maximum.at(flat, iy * nx + ix, pw[:, 2])
            hm = flat.reshape(ny, nx)
        valid = np.isfinite(hm)

        # 3. split into flat regions wherever the height jumps
        hmf = np.where(valid, hm, -10.0)
        edge = np.zeros((ny, nx), bool)
        jx = np.abs(np.diff(hmf, axis=1)) > self.edge_tol
        edge[:, 1:] |= jx
        edge[:, :-1] |= jx
        jy = np.abs(np.diff(hmf, axis=0)) > self.edge_tol
        edge[1:, :] |= jy
        edge[:-1, :] |= jy
        cnt, labels = cv2.connectedComponents((valid & ~edge).astype(np.uint8), connectivity=4)
        kernel = np.ones((3, 3), np.uint8)

        for k in range(1, cnt):
            comp = labels == k
            if comp.sum() < 20:
                continue
            zmed = float(np.median(hm[comp]))
            # win back the 1-cell border removed by the edge test
            grown = (cv2.dilate(comp.astype(np.uint8), kernel).astype(bool) & valid &
                     (np.abs(hmf - zmed) < self.edge_tol))
            if grown.sum() * self.cell ** 2 < self.min_area:
                continue
            parts = split_region(grown, self.cell)
            self.get_logger().info(f'split: {len(parts)} part(s) from region z={zmed:.2f}', throttle_duration_sec=2.0)
            for part in parts:
                if part.sum() * self.cell ** 2 < self.min_area:
                    continue
                dets += self.classify_region(part, zmed, x0, y0)

        dets.sort(key=lambda d: -d['z_top'])
        self.remember_layers(dets)
        self.update_tracks(dets)
        self.publish(msg.header.stamp, dets, hm, valid)

    def classify_region(self, grown, zmed, x0, y0):
        dets = []
        ys, xs = np.nonzero(grown)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        corners = cv2.boxPoints(rect)
        e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
        long_e, l1, l2 = (e1, np.linalg.norm(e1), np.linalg.norm(e2)) if \
            np.linalg.norm(e1) >= np.linalg.norm(e2) else \
            (e2, np.linalg.norm(e2), np.linalg.norm(e1))
        yaw = math.atan2(long_e[1], long_e[0])
        if yaw > math.pi / 2:
            yaw -= math.pi
        elif yaw <= -math.pi / 2:
            yaw += math.pi
        length = (l1 + 1) * self.cell
        width = (l2 + 1) * self.cell
        fill = float(grown.sum()) / ((l1 + 1) * (l2 + 1))

        height, face = None, None
        for (s_, l_), hgt in self.faces.items():
            if abs(length - l_) < self.size_tol and abs(width - s_) < self.size_tol:
                height, face = hgt, (s_, l_)
        if height is not None and fill >= self.min_fill:
            # A flat rectangle matching one carton face can also be several smaller cartons
            # touching (two standing 0.3x0.2 faces side by side = one lying 0.3x0.4 face).
            # Depth alone cannot tell, so use the orientation already seen in this layer.
            seen = self.layer_seen(zmed)
            if seen and face not in seen:
                sub = {k: self.faces[k] for k in seen}
                tiles = tile_region(grown, sub, self.cell, z_above=zmed - self.pallet_top)
                if len(tiles) > 1:
                    return self.tiles_to_dets(tiles, zmed, x0, y0)
            dets.append({
                'x': x0 + (rect[0][0] + 0.5) * self.cell,
                'y': y0 + (rect[0][1] + 0.5) * self.cell,
                'z_top': zmed, 'yaw': yaw,
                'length': length, 'width': width,
                'height': height, 'fill': fill, 'pickable': True,
                'tiled': False, '_corners': corners, '_face': face})
            return dets
        tiles = tile_region(grown, self.faces, self.cell, z_above=zmed - self.pallet_top)
        if tiles:
            dets += self.tiles_to_dets(tiles, zmed, x0, y0)
        else:
            dets.append({
                'x': x0 + (rect[0][0] + 0.5) * self.cell,
                'y': y0 + (rect[0][1] + 0.5) * self.cell,
                'z_top': zmed, 'yaw': yaw,
                'length': length, 'width': width,
                'height': 0.0, 'fill': fill, 'pickable': False,
                'tiled': False, '_corners': corners})
        return dets

    def tiles_to_dets(self, tiles, zmed, x0, y0):
        return [{
            'x': x0 + (t_['cx'] + 0.5) * self.cell,
            'y': y0 + (t_['cy'] + 0.5) * self.cell,
            'z_top': zmed, 'yaw': t_['yaw'],
            'length': t_['length'], 'width': t_['width'],
            'height': t_['height'], 'fill': 1.0, 'pickable': True,
            'tiled': True, '_corners': t_['corners']} for t_ in tiles]

    def layer_seen(self, z_top):
        for lz, keys in self.layer_faces.items():
            if abs(lz - z_top) < 0.02:
                return keys
        return set()

    def remember_layers(self, dets):
        """Record which carton face was seen as a lone rectangle at each layer height."""
        if not dets:                       # empty pallet: forget everything
            self.layer_faces.clear()
            return
        for d in dets:
            if d['pickable'] and not d['tiled'] and d.get('_face') is not None:
                for lz in self.layer_faces:
                    if abs(lz - d['z_top']) < 0.02:
                        self.layer_faces[lz].add(d['_face'])
                        break
                else:
                    self.layer_faces[d['z_top']] = {d['_face']}

    def update_tracks(self, dets):
        """Count how many consecutive scans each detection has persisted at the same pose."""
        old = self.tracks
        taken = set()
        new = []
        for d in dets:
            best, best_dist = None, None
            for i, tr in enumerate(old):
                if i in taken or tr['pickable'] != d['pickable']:
                    continue
                dist = math.hypot(d['x'] - tr['x'], d['y'] - tr['y'])
                dyaw = abs((d['yaw'] - tr['yaw'] + math.pi / 2) % math.pi - math.pi / 2)
                if (dist < self.match_dist and dyaw < self.match_yaw and
                        abs(d['z_top'] - tr['z_top']) < 0.02 and
                        (best is None or dist < best_dist)):
                    best, best_dist = i, dist
            if best is not None:
                taken.add(best)
                d['_count'] = old[best]['count'] + 1
            else:
                d['_count'] = 1
            d['_stable'] = d['_count'] >= self.stable_frames
            new.append({'x': d['x'], 'y': d['y'], 'yaw': d['yaw'], 'z_top': d['z_top'],
                        'pickable': d['pickable'], 'count': min(d['_count'], 1000)})
        self.tracks = new

    def broadcast_target(self, stamp, dets):
        """TF frame of the next carton to pick: top-face centre, tool z pointing down, x along the long edge."""
        cands = [d for d in dets if d['pickable'] and d['_stable']]
        if not cands:
            return
        top = max(d['z_top'] for d in cands)
        layer = [d for d in cands if top - d['z_top'] < 0.02]    # highest layer first
        tgt = min(layer, key=lambda d: math.hypot(d['x'] - self.base_xy[0], d['y'] - self.base_xy[1]))
        ts = TransformStamped()
        ts.header.stamp = stamp
        ts.header.frame_id = self.world
        ts.child_frame_id = self.target_frame
        ts.transform.translation.x = float(tgt['x'])
        ts.transform.translation.y = float(tgt['y'])
        ts.transform.translation.z = float(tgt['z_top'])
        # Rz(yaw) * Rx(180 deg): z axis down, x axis along the long edge
        ts.transform.rotation.x = math.cos(tgt['yaw'] / 2.0)
        ts.transform.rotation.y = math.sin(tgt['yaw'] / 2.0)
        ts.transform.rotation.z = 0.0
        ts.transform.rotation.w = 0.0
        self.tf_broadcaster.sendTransform(ts)

    def publish(self, stamp, dets, hm, valid):
        out = []
        for i, d in enumerate(dets):
            d['id'] = i
            if d['_stable']:
                out.append({k: (round(float(val), 4) if isinstance(val, (float, np.floating)) else val)
                            for k, val in d.items() if not k.startswith('_')})
        self.scan_id += 1
        self.last_payload = {'scan_id': self.scan_id, 'stale': False, 'boxes': out}
        self.last_json = String(data=json.dumps(self.last_payload))
        self.pub_json.publish(self.last_json)
        self.broadcast_target(stamp, dets)
        npick = sum(1 for d in dets if d['pickable'] and d['_stable'])
        self.get_logger().info(
            f'{len(dets)} regions, {npick} stable pickable | ' + '; '.join(
                f"#{d['id']} z={d['z_top']:.2f} {d['length']:.3f}x{d['width']:.3f} "
                f"{'tiled' if d['tiled'] else 'single'} "
                f"{'PICK' if d['pickable'] else 'skip'} n={d['_count']}" for d in dets),
            throttle_duration_sec=2.0)

        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for d in dets:
            mk = Marker()
            mk.header.frame_id = self.world
            mk.header.stamp = stamp
            mk.ns = 'cartons'
            mk.id = d['id']
            mk.type = Marker.CUBE
            h = d['height'] if d['height'] > 0 else 0.05
            mk.pose.position.x = d['x']
            mk.pose.position.y = d['y']
            mk.pose.position.z = d['z_top'] - h / 2.0
            mk.pose.orientation.z = math.sin(d['yaw'] / 2.0)
            mk.pose.orientation.w = math.cos(d['yaw'] / 2.0)
            mk.scale.x, mk.scale.y, mk.scale.z = d['length'], d['width'], h
            mk.color.a = 0.6
            if not d['pickable']:
                mk.color.r = 1.0                      # red: not a whole carton face
            elif d['_stable']:
                mk.color.g = 1.0                      # green: confirmed
            else:
                mk.color.r, mk.color.g = 1.0, 1.0     # yellow: not yet stable
            ma.markers.append(mk)
        self.pub_markers.publish(ma)

        g = np.where(valid, np.clip((hm - self.pallet_top) / 1.0, 0, 1) * 255, 0).astype(np.uint8)
        vis = cv2.applyColorMap(g, cv2.COLORMAP_JET)
        vis[~valid] = (30, 30, 30)
        s = 4
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        for d in dets:
            c = ((d['_corners'] + 0.5) * s).astype(np.int32)
            col = (0, 0, 255) if not d['pickable'] else ((0, 255, 0) if d['_stable'] else (0, 255, 255))
            cv2.polylines(vis, [c], True, col, 2)
            cv2.putText(vis, str(d['id']), tuple(int(v) for v in c.mean(axis=0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        im = Image()
        im.header.stamp = stamp
        im.header.frame_id = self.world
        im.height, im.width = vis.shape[:2]
        im.encoding = 'bgr8'
        im.step = im.width * 3
        im.data = vis.tobytes()
        self.pub_debug.publish(im)


def main():
    rclpy.init()
    node = BoxDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()