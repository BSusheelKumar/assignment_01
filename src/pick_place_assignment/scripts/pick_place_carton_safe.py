#!/usr/bin/env python3
"""Safe carton pick-and-place for MoveIt2.

This script intentionally avoids the brittle long Cartesian move that can fail
midway with an incomplete path, especially when carrying the carton across a
pallet edge or through a tight clearance region. Instead it follows the MoveIt2
recommendation to use:

  * joint-space / pose planning for collision-aware obstacle clearance,
  * short Cartesian segments for local straight-line motions,
  * staged waypoints (pre-grasp -> grasp -> lift -> clear -> transfer -> place).

The algorithm is based on the general MoveIt2 guidance for Cartesian path use:
short, collision-checked local segments are reliable; long multi-axis moves are not.
"""

import json
import math
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    JointConstraint,
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    OrientationConstraint,
    PlanningScene,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from moveit_msgs.srv import GetStateValidity
import sys


def input(prompt=''):
    print(prompt.replace('Enter: ', '>> '))
    time.sleep(0.5)

import numpy as np

MAX_REACH = 1.20   # horizontal base->TCP limit (tune; CRX-10iA/L max is ~1.25)
MAX_STACK = 0.85   # max stack height above the pallet [m]

MAX_JOINT_STEP = 0.25   # rad per 5 mm step; real branch flips are ~3 rad, and MoveIt's own jump threshold (0.35) still applies
GROUP = 'manipulator'
TIP = 'tcp'
APPROACH_HEIGHT = 0.22
GRASP_OFFSET = 0.0
LIFT_HEIGHT = 0.45
SOURCE_CENTER_X = 0.80
SOURCE_CENTER_Y = -0.55
DEST_CENTER_X = -0.35
DEST_CENTER_Y = -0.95
PALLET_LENGTH = 0.60
PALLET_WIDTH = 0.60
PALLET_HEIGHT = 0.15
PLACE_CLEARANCE = 0.015
DEST_SLOT_GAP = 0.002
MAX_DEST_CARTONS = 6
TOUCH_LINKS = ['gripper_link', 'ee_link', 'tcp', 'J6_link', 'flange', 'fanuc_flange',
               'J5_link', 'J4_link']
CARTESIAN_SCALE = 0.12
MAX_J1_SWING = 3.0      # rad of total base (J1) travel allowed in one move
BASE_XY = (0.0, 0.0)    # world x,y of the J1 axis: set from tf2_echo
ARC_STEP_DEG = 6.0
ALLOW_TOOL_FLIP = True   # only valid if the gripper is symmetric under a 180 deg turn


class SafeCartonPicker(Node):
    def __init__(self):
        super().__init__('safe_carton_picker')

        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.cartesian_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        self.move_group_client.wait_for_server()
        self.execute_client.wait_for_server()
        self.cartesian_client.wait_for_service()
        self.scene_client.wait_for_service()

        self.vacuum_pub = self.create_publisher(Bool, '/vacuum/enable', 10)
        self.vacuum_attached = None
        self.create_subscription(Bool, '/vacuum/attached', self._on_vacuum_state, 10)
        self.boxes = None
        self.boxes_fresh = False
        self.create_subscription(String, '/perception/boxes_json', self._on_boxes, 10)
        self.validity_client = self.create_client(GetStateValidity, '/check_state_validity')
        self.validity_client.wait_for_service()
        self.group_joints = None

    def _on_boxes(self, msg):
        decoded = json.loads(msg.data)
        self.boxes_fresh = not decoded.get('stale', False)
        if self.boxes_fresh:
            self.boxes = decoded.get('boxes', [])

    def _on_vacuum_state(self, msg):
        self.vacuum_attached = msg.data

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def check_path(self, traj, label, max_winding=2.0, max_j123=1.5,
                max_j46=3.5, min_j5_deg=15.0, stride=4):
        jt = traj.joint_trajectory
        if len(jt.points) < 2:
            return True
        names = list(jt.joint_names)
        self.group_joints = names
        q = [list(p.positions) for p in jt.points]
        n = len(names)
        ix = {nm: i for i, nm in enumerate(names)}
        travel = [sum(abs(b[j] - a[j]) for a, b in zip(q, q[1:])) for j in range(n)]
        length = sum(math.sqrt(sum((b[j] - a[j]) ** 2 for j in range(n))) for a, b in zip(q, q[1:]))
        net = math.sqrt(sum((q[-1][j] - q[0][j]) ** 2 for j in range(n)))
        winding = length / max(net, 1e-3)
        max_joint_step = max(
            abs(b[j] - a[j]) for a, b in zip(q, q[1:]) for j in range(n)
        )
        j5_min = min(abs(p[ix['J5']]) for p in q)
        self.get_logger().info(
            f'{label}: q_start={[round(math.degrees(x)) for x in q[0]]} '
            f'q_end={[round(math.degrees(x)) for x in q[-1]]} deg, '
            f'travel={[round(t, 2) for t in travel]}, '
            f'max_step={max_joint_step:.3f} rad, winding={winding:.2f}, '
            f'min|J5|={math.degrees(j5_min):.0f} deg')

        reasons = []
        if net >= 0.5 and winding > max_winding:
            reasons.append(f'winding {winding:.2f} > {max_winding}')
        if max_joint_step > MAX_JOINT_STEP:
            reasons.append(f'joint step {max_joint_step:.3f} rad > {MAX_JOINT_STEP}')
        if max(travel[ix[j]] for j in ('J2', 'J3')) > max_j123:
            reasons.append('J2/J3 travel too large (arm folding / elbow flip)')
        if travel[ix['J1']] > MAX_J1_SWING:
            reasons.append('J1 travel too large')
        if max(travel[ix[j]] for j in ('J4', 'J5', 'J6')) > max_j46:
            reasons.append('wrist travel too large')
        if j5_min < math.radians(min_j5_deg):
            reasons.append(f'J5 within {min_j5_deg:.0f} deg of the wrist singularity')
        if reasons:
            self.get_logger().error(f'{label}: REJECTED: ' + '; '.join(reasons))
            return False

        for i in list(range(0, len(q), stride)) + [len(q) - 1]:
            req = GetStateValidity.Request()
            req.group_name = GROUP
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = names
            req.robot_state.joint_state.position = q[i]
            fut = self.validity_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            res = fut.result()
            if res is None or not res.valid:
                pairs = sorted({(c.contact_body_1, c.contact_body_2) for c in res.contacts}) if res else []
                self.get_logger().error(f'{label}: collision at point {i}: {pairs}')
                return False
        return True

    def get_next_carton(self, timeout=8.0):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not self.boxes_fresh:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.boxes_fresh or self.boxes is None:
            return None, None, None

        candidates = [
            (index, box) for index, box in enumerate(self.boxes)
            if box.get('pickable', False) and not self.is_blocked(box)
        ]
        if not candidates:
            return None, None, None

        highest = max(box['z_top'] for _, box in candidates)
        top_layer = [item for item in candidates if highest - item[1]['z_top'] < 0.02]
        index, box = min(
            top_layer,
            key=lambda item: math.hypot(item[1]['x'], item[1]['y']),
        )

        target_pose = Pose()
        target_pose.position.x = box['x']
        target_pose.position.y = box['y']
        target_pose.position.z = box['z_top']
        target_pose.orientation.x = math.cos(box['yaw'] / 2.0)
        target_pose.orientation.y = math.sin(box['yaw'] / 2.0)
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0
        return index, box, target_pose

    def branch_constraints(self, x, y):
        spec = {
            'J1': (math.atan2(y, x), math.radians(40)),
            'J2': (math.radians(50), math.radians(60)),
            'J3': (math.radians(40), math.radians(70)),
            'J5': (math.radians(-75), math.radians(55)),
            'J4': (math.radians(-105), math.radians(40)),
            'J6': (math.radians(110), math.radians(80)),
        }
        out = []
        for name, (center, tol) in spec.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = center
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            out.append(jc)
        return out

    def scene_carton_ids(self):
        from moveit_msgs.srv import GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents
        if not hasattr(self, 'get_scene_client'):
            self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
            self.get_scene_client.wait_for_service()
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        fut = self.get_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            return []
        return [o.id for o in res.scene.world.collision_objects
                if o.id.startswith('carton_')]

    def sync_cartons(self):
        """Mirror cartons on the SOURCE pallet into the scene. Never fatal."""
        boxes = self.boxes or []
        # clear everything synced before (ids may have shifted)
        clear = PlanningScene(is_diff=True)
        for cid in self.scene_carton_ids():
            obj = CollisionObject()
            obj.header.frame_id = 'world'
            obj.id = cid
            obj.operation = CollisionObject.REMOVE
            clear.world.collision_objects.append(obj)
        if clear.world.collision_objects:
            self.apply_scene(clear, strict=False)

        scene = PlanningScene(is_diff=True)
        for i, b in enumerate(boxes):
            # skip unsegmented blobs (flat or bigger than any carton)
            if b['height'] < 0.05 or max(b['length'], b['width']) > 0.5:
                continue
            # skip anything not on the source pallet footprint
            if (abs(b['x'] - SOURCE_CENTER_X) > PALLET_LENGTH / 2 + 0.15 or
                    abs(b['y'] - SOURCE_CENTER_Y) > PALLET_WIDTH / 2 + 0.15):
                continue
            obj = CollisionObject()
            obj.header.frame_id = 'world'
            obj.id = f'carton_{i}'
            obj.primitives.append(SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[_snap(b['length']), _snap(b['width']), b['height']]))
            pose = Pose()
            pose.position.x = b['x']
            pose.position.y = b['y']
            pose.position.z = b['z_top'] - b['height'] / 2.0
            pose.orientation.z = math.sin(b['yaw'] / 2.0)
            pose.orientation.w = math.cos(b['yaw'] / 2.0)
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        self._synced_ids = [o.id for o in scene.world.collision_objects]
        self.apply_scene(scene, strict=False)

    def solve_ik(self, pose):
        from moveit_msgs.srv import GetPositionIK
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'ik_client'):
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.ik_client.wait_for_service()
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        j1 = math.atan2(pose.position.y, pose.position.x)
        for j6 in (110, 150, 70, 30, 190):
            seed = [j1, math.radians(38), math.radians(21),
                    math.radians(-106), math.radians(-45), math.radians(j6)]
            req = GetPositionIK.Request()
            r = req.ik_request
            r.group_name = GROUP
            r.avoid_collisions = True
            r.ik_link_name = TIP
            r.pose_stamped.header.frame_id = 'world'
            r.pose_stamped.pose = pose
            r.timeout = Duration(sec=0, nanosec=300000000)
            r.robot_state.joint_state.name = names
            r.robot_state.joint_state.position = seed
            fut = self.ik_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            res = fut.result()
            if res is None or res.error_code.val != 1:
                continue
            js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
            q = [js[n] for n in names]
            d = [math.degrees(x) for x in q]
            ok = (-175 <= d[3] <= -50 and -130 <= d[4] <= -20
                  and -60 <= d[5] <= 215 and -100 <= d[0] <= 100)
            self.get_logger().info(f'IK seed J6={j6}: {[round(x) for x in d]} accepted={ok}')
            if ok:
                return names, q
        return None

    def move_joints(self, names, q):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = GROUP
        req.num_planning_attempts = 12
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.25
        req.max_acceleration_scaling_factor = 0.25
        req.start_state.is_diff = True
        c = Constraints()
        for n, v in zip(names, q):
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = v
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().warn('joint goal rejected by MoveIt')
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        code = rf.result().result.error_code.val
        if code != 1:
            self.get_logger().warn(f'joint goal failed, MoveIt error code = {code}')
        return code == 1

    def transfer_joint(self, pose):
        """Joint-space transfer: IK seeded with a good destination posture, then planned."""
        from moveit_msgs.srv import GetPositionIK
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'ik_client'):
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.ik_client.wait_for_service()
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        base = math.degrees(math.atan2(pose.position.y, pose.position.x))
        for dj in (-8.0, 0.0, 8.0):
            for j6 in (81.0, 120.0, 40.0):
                j1 = base + dj
                seed = [math.radians(v) for v in (j1, 37.0, 43.0, -95.0, j1, j6)]
                req = GetPositionIK.Request()
                r = req.ik_request
                r.group_name = GROUP
                r.avoid_collisions = True
                r.ik_link_name = TIP
                r.pose_stamped.header.frame_id = 'world'
                r.pose_stamped.pose = pose
                r.timeout = Duration(sec=0, nanosec=500000000)
                r.robot_state.joint_state.name = names
                r.robot_state.joint_state.position = seed
                fut = self.ik_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut)
                res = fut.result()
                if res is None or res.error_code.val != 1:
                    continue
                js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                q = [js[n] for n in names]
                d = [math.degrees(x) for x in q]
                ok = (abs(d[4]) >= 40 and -140 <= d[3] <= -50 and -20 <= d[2] <= 110)
                self.get_logger().info(f'transfer IK: {[round(x) for x in d]} accepted={ok}')
                if ok:
                    return self.move_joints(names, q)
        self.get_logger().error('no safe-posture IK for the transfer pose')
        return False

    def move_pose(self, pose):
        sol = self.solve_ik(pose)
        if sol is None:
            self.get_logger().error('no IK solution in the front/elbow-up branch')
            return False
        return self.move_joints(*sol)

    def move_pose_old(self, pose):
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = GROUP
        req.num_planning_attempts = 12
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.25
        req.max_acceleration_scaling_factor = 0.25
        req.start_state.is_diff = True

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = 'world'
        position_constraint.link_name = TIP
        position_constraint.constraint_region = BoundingVolume(
            primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.01])],
            primitive_poses=[pose],
        )
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = 'world'
        orientation_constraint.link_name = TIP
        orientation_constraint.orientation = pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.08
        orientation_constraint.absolute_y_axis_tolerance = 0.08
        orientation_constraint.absolute_z_axis_tolerance = 0.08
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        constraints.joint_constraints.extend(
            self.branch_constraints(pose.position.x, pose.position.y))
        req.goal_constraints.append(constraints)

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('pose goal rejected by MoveIt')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        return result.result.error_code.val == 1

    def remove_cartons_overlapping(self, box, margin=0.06, idx=None):
        from moveit_msgs.srv import GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents
        if not hasattr(self, 'get_scene_client'):
            self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
            self.get_scene_client.wait_for_service()
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        fut = self.get_scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            return []
        hx = _snap(box['length']) / 2 + margin
        hy = _snap(box['width']) / 2 + margin
        scene = PlanningScene(is_diff=True)
        removed = []
        if idx is not None:
            rm = CollisionObject()
            rm.header.frame_id = 'world'
            rm.id = f'carton_{idx}'
            rm.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(rm)
            removed.append(rm.id)
        for o in res.scene.world.collision_objects:
            if o.id.startswith('carton_') and o.primitive_poses:
                self.get_logger().info(
                    f'scene {o.id}: obj_pose=({o.pose.position.x:.3f},{o.pose.position.y:.3f},{o.pose.position.z:.3f}) '
                    f'prim_pose=({o.primitive_poses[0].position.x:.3f},{o.primitive_poses[0].position.y:.3f},{o.primitive_poses[0].position.z:.3f})')
        for o in res.scene.world.collision_objects:
            if not o.id.startswith('carton_') or not o.primitive_poses or o.id in removed:
                continue
            pp = o.primitive_poses[0].position
            class _P: pass
            p = _P()
            p.x = o.pose.position.x + pp.x
            p.y = o.pose.position.y + pp.y
            p.z = o.pose.position.z + pp.z
            if abs(p.x - box['x']) < hx and abs(p.y - box['y']) < hy \
                    and p.z > box['z_top'] - box['height'] - 0.05:
                rm = CollisionObject()
                rm.header.frame_id = 'world'
                rm.id = o.id
                rm.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(rm)
                removed.append(o.id)
        if removed:
            self.apply_scene(scene, strict=False)
        self.get_logger().info(f'removed overlapping scene cartons: {removed}')
        return removed

    def reachable(self, pose):
        from moveit_msgs.srv import GetPositionIK
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'ik_client'):
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.ik_client.wait_for_service()
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        base = math.degrees(math.atan2(pose.position.y, pose.position.x))
        for dj in (-8.0, 0.0, 8.0):
            for j6 in (81.0, 120.0, 40.0):
                j1 = base + dj
                seed = [math.radians(v) for v in (j1, 37.0, 43.0, -95.0, j1, j6)]
                req = GetPositionIK.Request()
                r = req.ik_request
                r.group_name = GROUP
                r.avoid_collisions = True
                r.ik_link_name = TIP
                r.pose_stamped.header.frame_id = 'world'
                r.pose_stamped.pose = pose
                r.timeout = Duration(sec=0, nanosec=300000000)
                r.robot_state.joint_state.name = names
                r.robot_state.joint_state.position = seed
                fut = self.ik_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut)
                res = fut.result()
                if res is None or res.error_code.val != 1:
                    continue
                js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                d = [math.degrees(js[n]) for n in names]
                if abs(d[4]) >= 40 and -140 <= d[3] <= -50 and -20 <= d[2] <= 110:
                    return True
        return False

    def place_feasible(self, over, low):
        """Dry run: IK at the overhead pose, then Cartesian descent from that state."""
        from moveit_msgs.srv import GetPositionIK
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'ik_client'):
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.ik_client.wait_for_service()
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        base = math.degrees(math.atan2(over.position.y, over.position.x))
        ik_hits, best = 0, 0.0
        for dj in (-8.0, 0.0, 8.0):
            for j6 in (81.0, 120.0, 40.0):
                j1 = base + dj
                seed = [math.radians(v) for v in (j1, 37.0, 43.0, -95.0, j1, j6)]
                req = GetPositionIK.Request()
                r = req.ik_request
                r.group_name = GROUP
                r.avoid_collisions = True
                r.ik_link_name = TIP
                r.pose_stamped.header.frame_id = 'world'
                r.pose_stamped.pose = over
                r.timeout = Duration(sec=0, nanosec=300000000)
                r.robot_state.joint_state.name = names
                r.robot_state.joint_state.position = seed
                fut = self.ik_client.call_async(req)
                rclpy.spin_until_future_complete(self, fut)
                res = fut.result()
                if res is None or res.error_code.val != 1:
                    continue
                ik_hits += 1
                js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                q = [js[n] for n in names]
                cr = GetCartesianPath.Request()
                cr.header.frame_id = 'world'
                cr.start_state.joint_state.name = names
                cr.start_state.joint_state.position = q
                cr.group_name = GROUP
                cr.link_name = TIP
                cr.waypoints = [low]
                cr.max_step = 0.01
                cr.jump_threshold = 0.0
                cr.prismatic_jump_threshold = 0.005
                cr.revolute_jump_threshold = 0.35
                cr.avoid_collisions = True
                f2 = self.cartesian_client.call_async(cr)
                rclpy.spin_until_future_complete(self, f2)
                resp = f2.result()
                if resp is None:
                    continue
                best = max(best, resp.fraction)
                pts = resp.solution.joint_trajectory.points
                if resp.fraction >= 0.98 and len(pts) >= 2:
                    qq = [list(pt.positions) for pt in pts]
                    step = max(abs(b[j] - a[j]) for a, b in zip(qq, qq[1:]) for j in range(6))
                    if step <= 0.10:
                        self.reach_note = f'ik_hits={ik_hits} best_fraction={resp.fraction:.2f}'
                        return True
        self.reach_note = f'ik_hits={ik_hits} best_fraction={best:.2f}'
        return False

    def _cart_from(self, q, waypoints, max_j123=1.5):
        """Dry-run Cartesian path from joint state q. Returns (q_end, None) or (None, reason)."""
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        cr = GetCartesianPath.Request()
        cr.header.frame_id = 'world'
        cr.start_state.joint_state.name = names
        cr.start_state.joint_state.position = list(q)
        cr.group_name = GROUP
        cr.link_name = TIP
        cr.waypoints = waypoints
        cr.max_step = 0.005
        cr.jump_threshold = 0.0
        cr.prismatic_jump_threshold = 0.005
        cr.revolute_jump_threshold = 0.35
        cr.avoid_collisions = True
        fut = self.cartesian_client.call_async(cr)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if resp is None:
            return None, 'no response'
        if resp.fraction < 0.98:
            return None, f'fraction {resp.fraction:.2f}'
        jt = resp.solution.joint_trajectory
        if len(jt.points) < 2:
            return list(q), None
        order = [list(jt.joint_names).index(n) for n in names]
        qq = [[pt.positions[i] for i in order] for pt in jt.points]
        pairs = list(zip(qq, qq[1:]))
        travel = [sum(abs(b[j] - a[j]) for a, b in pairs) for j in range(6)]
        step, at, jj = max((abs(b[j] - a[j]), i, j)
                           for i, (a, b) in enumerate(pairs) for j in range(6))
        if step > MAX_JOINT_STEP:
            return None, (f'joint jump {step:.2f} rad in J{jj + 1} at point {at}/{len(pairs)}, '
                          f'J5={math.degrees(qq[at][4]):.0f} deg')
        if max(travel[1:3]) > max_j123:
            return None, (f'J2/J3 travel {max(travel[1:3]):.2f} > {max_j123} '
                          f'(J2 {math.degrees(qq[0][1]):.0f}->{math.degrees(qq[-1][1]):.0f}, '
                          f'J3 {math.degrees(qq[0][2]):.0f}->{math.degrees(qq[-1][2]):.0f})')
        if travel[0] > MAX_J1_SWING:
            return None, f'J1 travel {travel[0]:.2f} > {MAX_J1_SWING}'
        if max(travel[3:6]) > 3.5:
            return None, f'wrist travel {max(travel[3:6]):.2f}'
        if min(abs(pt[4]) for pt in qq) < math.radians(15):
            return None, 'wrist singularity'
        return qq[-1], None

    def ik_candidates(self, pose):
        """Other front-branch IK solutions (different wrist postures), best wrist first."""
        from moveit_msgs.srv import GetPositionIK
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'ik_client'):
            self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
            self.ik_client.wait_for_service()
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        j1 = math.atan2(pose.position.y, pose.position.x)
        sols = []
        for j5 in (-65.0, -90.0, -40.0):
            for j4 in (-95.0, -125.0, -155.0):
                for j6 in (130.0, 80.0, 180.0, 30.0, -30.0):
                    seed = [j1, math.radians(45), math.radians(30), math.radians(j4),
                            math.radians(j5), math.radians(j6)]
                    req = GetPositionIK.Request()
                    r = req.ik_request
                    r.group_name = GROUP
                    r.avoid_collisions = True
                    r.ik_link_name = TIP
                    r.pose_stamped.header.frame_id = 'world'
                    r.pose_stamped.pose = pose
                    r.timeout = Duration(sec=0, nanosec=80000000)
                    r.robot_state.joint_state.name = names
                    r.robot_state.joint_state.position = seed
                    fut = self.ik_client.call_async(req)
                    rclpy.spin_until_future_complete(self, fut)
                    res = fut.result()
                    if res is None or res.error_code.val != 1:
                        continue
                    js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                    q = [js[n] for n in names]
                    d = [math.degrees(x) for x in q]
                    if not (-175 <= d[3] <= -50 and -130 <= d[4] <= -30
                            and -60 <= d[5] <= 215 and -100 <= d[0] <= 100):
                        continue
                    if any(max(abs(x - y) for x, y in zip(d, e)) < 4.0 for e, _ in sols):
                        continue
                    sols.append((d, q))
        sols.sort(key=lambda t: -abs(t[0][4]))
        return [q for _, q in sols]

    def pick_chain(self, pre_grasp, grasp, lift, rise, quick=False):
        """Dry run of what the robot really does: IK at pre-grasp, then Cartesian
        descent, lift and rise. Tries the usual posture first, then other wrist postures.
        Returns (joint_state_at_top, None) or (None, reason); sets self.pre_q."""
        steps = [('descent', [grasp]), ('lift', [lift])]
        if rise is not None:
            steps.append(('rise', [rise]))
        self.pre_q = None

        def run(q0):
            q = q0
            for label, wps in steps:
                q, w = self._cart_from(q, wps)
                if q is None:
                    return None, f'{label}: {w}'
            return q, None

        why = 'no front-branch IK at the pre-grasp pose'
        first = self.solve_ik(pre_grasp)
        tried = [first[1]] if first is not None else []
        for q0 in list(tried):
            q, w = run(q0)
            if q is not None:
                self.pre_q = q0
                return q, None
            why = w
        if quick:
            return None, why
        for q0 in self.ik_candidates(pre_grasp):
            if any(max(abs(x - y) for x, y in zip(q0, t)) < 0.05 for t in tried):
                continue
            tried.append(q0)
            q, w = run(q0)
            if q is not None:
                self.get_logger().info(
                    f'pick: using wrist posture {[round(math.degrees(x)) for x in q0]}')
                self.pre_q = q0
                return q, None
            why = w
        # last resort: IK at the GRASP pose, then come straight UP to the pre-grasp
        # pose, so the descent is the same continuous path in reverse
        g_sols = []
        s0 = self.solve_ik(grasp)
        if s0 is not None:
            g_sols.append(s0[1])
        g_sols += self.ik_candidates(grasp)
        for q_g in g_sols:
            q_up, w = self._cart_from(q_g, [pre_grasp])
            if q_up is None:
                why = f'rise from grasp: {w}'
                continue
            q, w = run(q_up)
            if q is not None:
                self.get_logger().info('pick: using a posture found from the grasp pose')
                self.pre_q = q_up
                return q, None
            why = w
        return None, f'{why} (tried {len(tried)} wrist postures + {len(g_sols)} grasp-first)'

    def block(self, box):
        if not hasattr(self, 'blocked'):
            self.blocked = []
        self.blocked.append((box['x'], box['y'], box['z_top']))
        self.skip_ok = True

    def is_blocked(self, box):
        return any(abs(box['x'] - x) < 0.04 and abs(box['y'] - y) < 0.04
                   and abs(box['z_top'] - z) < 0.05
                   for x, y, z in getattr(self, 'blocked', []))

    def transfer_feasible(self, pick_over, c, z_over, z_place, q0=None, off=0.0):
        """Dry run of the transfer chain. Returns (mode, tool_yaw) or None."""
        if q0 is None:
            sol = self.solve_ik(pick_over)
            if sol is None:
                self.reach_note = 'no IK at pick height'
                return None
            q0 = sol[1]
        notes = []
        for mode in ('arc', 'line'):
            if mode == 'arc':
                wps, yaw_end = arc_waypoints(pick_over, c['x'], c['y'], z_over)
            else:
                wps = [pose_with(pick_over, x=c['x'], y=c['y'])]
                yaw_end = pose_yaw(pick_over)
            cy_ = c['yaw'] + off
            ty0 = cy_ + round((yaw_end - cy_) / math.pi) * math.pi
            tys = [ty0]
            if ALLOW_TOOL_FLIP:
                tys.append(ty0 + math.copysign(math.pi, yaw_end - ty0))
            for ty in tys:
                over = pose_yawed(pose_with(pick_over, x=c['x'], y=c['y']), ty)
                low = pose_yawed(pose_with(pick_over, x=c['x'], y=c['y'], z=z_place), ty)
                q = q0
                for label, wpl, lim in (('transfer', wps, 1.5), ('align', [over], 1.5),
                                        ('descent', [low], 5.2)):
                    q, why = self._cart_from(q, wpl, lim)
                    if q is None:
                        notes.append(f'{mode}@{ty:.2f} {label}: {why}')
                        break
                if q is not None:
                    self.reach_note = f'{mode}, tool_yaw={ty:.2f}'
                    return mode, ty
        self.reach_note = ' | '.join(notes)
        return None

    def log_contacts(self, label):
        req = GetStateValidity.Request()
        req.group_name = GROUP
        req.robot_state.is_diff = True
        fut = self.validity_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None:
            return
        pairs = sorted({(c.contact_body_1, c.contact_body_2) for c in res.contacts})
        self.get_logger().error(f'{label}: current state valid={res.valid}, contacts={pairs}')
        for c in res.contacts:
            self.get_logger().error(
                f'contact {c.contact_body_1} vs {c.contact_body_2} '
                f'at ({c.position.x:.3f},{c.position.y:.3f},{c.position.z:.3f}) '
                f'depth={c.depth:.4f}')
        for i, b in enumerate(self.boxes or []):
            self.get_logger().error(f'box[{i}] {b}')

    def cartesian_segment(self, waypoints, label='cartesian move', max_step=0.005,
                          min_fraction=0.98, max_j123=1.5):
        req = GetCartesianPath.Request()
        req.header.frame_id = 'world'
        req.start_state = RobotState(is_diff=True)
        req.group_name = GROUP
        req.link_name = TIP
        req.waypoints = waypoints
        req.max_step = max_step
        req.jump_threshold = 0.0
        req.prismatic_jump_threshold = 0.005
        req.revolute_jump_threshold = 0.35
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = CARTESIAN_SCALE
        req.max_acceleration_scaling_factor = CARTESIAN_SCALE


        future = self.cartesian_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().error(f'{label}: empty cartesian response')
            return False

        self.get_logger().info(
            f'{label}: cartesian fraction = {response.fraction:.2f}, '
            f'MoveIt error code = {response.error_code.val}'
        )
        if response.fraction < min_fraction:
            self.get_logger().warn(f'{label}: incomplete cartesian path; fraction={response.fraction:.2f}')
            self.log_contacts(label)
            return False


        if len(response.solution.joint_trajectory.points) < 2:
            self.get_logger().info(f'{label}: already at target, nothing to move')
            return True

        if not self.check_path(response.solution, label, max_j123=max_j123):
            self.get_logger().error(f'{label}: path rejected by joint-space collision check')
            return False
        
        pts = response.solution.joint_trajectory.points
        q = [list(p.positions) for p in pts]
        max_step = max(abs(b[j]-a[j]) for a, b in zip(q, q[1:]) for j in range(len(a)))
        travel = [abs(q[-1][j]-q[0][j]) for j in range(len(q[0]))]
        t_end = pts[-1].time_from_start
        self.get_logger().info(
            f'{label}: max joint step={max_step:.3f} rad, '
            f'per-joint travel={[round(x,2) for x in travel]}, '
            f'duration={t_end.sec + t_end.nanosec*1e-9:.1f}s')
        if max_step > MAX_JOINT_STEP:
            self.get_logger().warn(f'{label}: joint jump, likely IK branch flip; refusing')
            return False

        exec_goal = ExecuteTrajectory.Goal()
        exec_goal.trajectory = response.solution
        exec_future = self.execute_client.send_goal_async(exec_goal)
        rclpy.spin_until_future_complete(self, exec_future)
        handle = exec_future.result()
        if not handle or not handle.accepted:
            self.get_logger().warn(f'{label}: execution request rejected')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        return result.result.error_code.val == 1

    def _on_js(self, msg):
        self.js = dict(zip(msg.name, msg.position))

    def go_home(self, rise_pose=None):
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        home = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0]
        print('>> going home so the camera can see the pallet...')
        if rise_pose is not None:
            if not self.cartesian_segment([rise_pose], 'rise before home'):
                print('rise before home failed; not moving the arm sideways.')
                return False
        if self.move_joints(names, home):
            return True
        self.get_logger().warn('MoveIt home failed; folding arm up, then rotating')
        if not hasattr(self, 'js_sub'):
            self.js = {}
            self.js_sub = self.create_subscription(JointState, '/joint_states', self._on_js, 10)
            self.traj_pub = self.create_publisher(
                JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
            self.spin_for(1.0)
        self.spin_for(0.3)
        if 'J1' not in self.js:
            print('no /joint_states received; refusing blind home move.')
            return False
        msg = JointTrajectory()
        msg.joint_names = names
        p1 = JointTrajectoryPoint()
        p1.positions = [self.js['J1'], 0.0, 0.0, 0.0, 0.0, 0.0]
        p1.time_from_start = Duration(sec=8, nanosec=0)
        p2 = JointTrajectoryPoint()
        p2.positions = home
        p2.time_from_start = Duration(sec=16, nanosec=0)
        msg.points = [p1, p2]
        self.traj_pub.publish(msg)
        self.spin_for(17.0)
        return True

    def wait_fresh_scan(self, timeout=20.0):
        self.boxes_fresh = False
        self.boxes = None
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not self.boxes_fresh:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.spin_for(1.5)

    def apply_scene(self, scene, strict=True):
        req = ApplyPlanningScene.Request(scene=scene)
        future = self.scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            if strict:
                raise RuntimeError('MoveIt rejected the planning-scene update')
            self.get_logger().warn('scene update rejected (non-fatal)')
        self.spin_for(0.3)

    def add_environment(self):
        scene = PlanningScene(is_diff=True)

        for object_id, x_center, y_center in (
            ('source_pallet', SOURCE_CENTER_X, SOURCE_CENTER_Y),
            ('dest_pallet', DEST_CENTER_X, DEST_CENTER_Y),
        ):
            obj = CollisionObject()
            obj.header.frame_id = 'world'
            obj.id = object_id
            obj.primitives.append(
                SolidPrimitive(
                    type=SolidPrimitive.BOX,
                    dimensions=[PALLET_LENGTH, PALLET_WIDTH, PALLET_HEIGHT],
                )
            )
            pose = Pose()
            pose.position.x = x_center
            pose.position.y = y_center
            pose.position.z = PALLET_HEIGHT / 2.0
            pose.orientation.w = 1.0
            obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)

        self.apply_scene(scene)

    def remove_world_carton(self, idx):
        scene = PlanningScene(is_diff=True)
        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = f'carton_{idx}'
        obj.operation = CollisionObject.REMOVE
        scene.world.collision_objects.append(obj)
        self.apply_scene(scene, strict=False)

    def restore_world_carton(self, idx, box):
        scene = PlanningScene(is_diff=True)
        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = f'carton_{idx}'
        obj.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[box['length'], box['width'], box['height']],
            )
        )
        pose = Pose()
        pose.position.x = box['x']
        pose.position.y = box['y']
        pose.position.z = box['z_top'] - box['height'] / 2.0
        pose.orientation.z = math.sin(box['yaw'] / 2.0)
        pose.orientation.w = math.cos(box['yaw'] / 2.0)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        scene.world.collision_objects.append(obj)
        self.apply_scene(scene)

    def attach_carton(self, box):
        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = 'held_carton'
        obj.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[_snap(box['length']) - 0.08, _snap(box['width']) - 0.08, box['height'] - 0.02],
            )
        )

        pose = Pose()
        pose.position.x = box['x']
        pose.position.y = box['y']
        pose.position.z = box['z_top'] - box['height'] / 2.0
        pose.orientation.z = math.sin(box['yaw'] / 2.0)
        pose.orientation.w = math.cos(box['yaw'] / 2.0)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD

        attach = AttachedCollisionObject()
        attach.link_name = 'gripper_link'
        attach.object = obj
        attach.touch_links = TOUCH_LINKS

        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attach)
        self.apply_scene(scene)

    def detach_carton(self):
        detach = AttachedCollisionObject()
        detach.link_name = 'gripper_link'
        detach.object = CollisionObject(id='held_carton', operation=CollisionObject.REMOVE)

        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(detach)
        self.apply_scene(scene)

    def add_placed_carton_old(self, box, slot_index):
        slot = destination_slot(box, slot_index)
        if slot is None:
            raise ValueError('destination pallet slot does not fit this carton')
        layer, (offset_x, offset_y) = slot
        place_pose = Pose()
        place_pose.position.x = DEST_CENTER_X + offset_x
        place_pose.position.y = DEST_CENTER_Y + offset_y
        place_pose.position.z = PALLET_HEIGHT + layer * box['height'] + box['height'] / 2.0
        place_pose.orientation.z = 0.0
        place_pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = f'placed_carton_{slot_index}'
        obj.primitives.append(
            SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[box['length'], box['width'], box['height']],
            )
        )
        obj.primitive_poses.append(place_pose)
        obj.operation = CollisionObject.ADD

        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obj)
        self.apply_scene(scene)

    def add_placed_carton(self, box, slot_index, plan):
        pose = Pose()
        pose.position.x = plan['x']
        pose.position.y = plan['y']
        pose.position.z = PALLET_HEIGHT + plan['support_h'] + box['height'] / 2.0
        pose.orientation.z = math.sin(plan['yaw'] / 2.0)
        pose.orientation.w = math.cos(plan['yaw'] / 2.0)
        obj = CollisionObject()
        obj.header.frame_id = 'world'
        obj.id = f'placed_carton_{slot_index}'
        obj.primitives.append(SolidPrimitive(
            type=SolidPrimitive.BOX,
            dimensions=[_snap(box['length']), _snap(box['width']), box['height']]))
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obj)
        self.apply_scene(scene)

    def set_vacuum(self, enabled):
        self.spin_for(0.3)
        self.vacuum_pub.publish(Bool(data=enabled))
        self.spin_for(0.2)

    def wait_for_attachment(self, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.vacuum_attached is expected:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False


def _snap(v):
    for n in (0.2, 0.3, 0.4):
        if abs(v - n) < 0.03:
            return n
    return v


def pose_yawed(pose, yaw):
    q = Pose()
    q.position.x = pose.position.x
    q.position.y = pose.position.y
    q.position.z = pose.position.z
    q.orientation.x = math.cos(yaw / 2.0)
    q.orientation.y = math.sin(yaw / 2.0)
    q.orientation.z = 0.0
    q.orientation.w = 0.0
    return q


def destination_slot(box, slot_index):
    if slot_index < 0 or slot_index >= MAX_DEST_CARTONS:
        return None

    length = _snap(box['length'])
    width = _snap(box['width'])
    extent_x, extent_y = length, width      # every carton is placed at yaw 0
    gap_x, gap_y = 0.02, 0.04
    if (2.0 * extent_x + gap_x > PALLET_LENGTH + 0.04 or
            2.0 * extent_y + gap_y > PALLET_WIDTH + 0.04):
        return None

    layer, position = divmod(slot_index, 3)
    half_x = (extent_x + gap_x) / 2.0
    half_y = (extent_y + gap_y) / 2.0
    offsets = ((-half_x, -half_y), (-half_x, half_y), (half_x, 0.0))
    return layer, offsets[position]


class DestPlanner:
    """Height map of the destination pallet (metres above the pallet top)."""
    CELL = 0.01
    MARGIN = 0.02
    STEP = 2
    RING = 0.02

    def __init__(self):
        c = self.CELL
        self.nx = int(round((PALLET_LENGTH + 2 * self.MARGIN) / c))
        self.ny = int(round((PALLET_WIDTH + 2 * self.MARGIN) / c))
        self.x0 = DEST_CENTER_X - PALLET_LENGTH / 2 - self.MARGIN
        self.y0 = DEST_CENTER_Y - PALLET_WIDTH / 2 - self.MARGIN
        self.h = np.zeros((self.nx, self.ny))
        self.banned = set()

    def max_height(self):
        return float(self.h.max())

    def plan_all(self, box, limit=12):
        c = self.CELL
        L, W, H = _snap(box['length']), _snap(box['width']), box['height']
        ring = int(round(self.RING / c))
        orients = [(0.0, L, W)]
        if abs(L - W) > 0.01:
            orients.append((math.pi / 2.0, W, L))
        cands = []
        for yaw, ex, ey in orients:
            nx, ny = int(round(ex / c)), int(round(ey / c))
            for ix in range(0, self.nx - nx + 1, self.STEP):
                for iy in range(0, self.ny - ny + 1, self.STEP):
                    foot = self.h[ix:ix + nx, iy:iy + ny]
                    hs = float(foot.max())
                    if (foot >= hs - 0.01).mean() < 0.8:
                        continue
                    if hs + H > MAX_STACK:
                        continue
                    zone = self.h[max(ix - ring, 0):ix + nx + ring,
                                  max(iy - ring, 0):iy + ny + ring]
                    if zone.max() > hs + 0.01:
                        continue
                    cx = self.x0 + (ix + nx / 2.0) * c
                    cy = self.y0 + (iy + ny / 2.0) * c
                    dist = math.hypot(cx - BASE_XY[0], cy - BASE_XY[1])
                    if dist > MAX_REACH:
                        continue
                    key = (round(hs + H, 2), round(dist, 2), 0 if yaw == 0.0 else 1, ix, iy)
                    cands.append((key, dict(x=cx, y=cy, yaw=yaw, support_h=hs,
                                            ix=ix, iy=iy, nx=nx, ny=ny, top=hs + H)))
        cands.sort(key=lambda t: t[0])
        kept = []
        for _, cd in cands:
            if any(k['yaw'] == cd['yaw'] and abs(k['support_h'] - cd['support_h']) < 0.01
                   and abs(k['x'] - cd['x']) < 0.06 and abs(k['y'] - cd['y']) < 0.06
                   for k in kept):
                continue
            kept.append(cd)
            if len(kept) >= limit:
                break
        return kept

    def plan(self, box):
        r = self.plan_all(box, 1)
        return r[0] if r else None

    def commit(self, plan, box):
        self.h[plan['ix']:plan['ix'] + plan['nx'],
               plan['iy']:plan['iy'] + plan['ny']] = plan['top']


PLANNER = DestPlanner()


def pose_with(pose, x=None, y=None, z=None):
    new_pose = Pose()
    new_pose.position.x = pose.position.x if x is None else x
    new_pose.position.y = pose.position.y if y is None else y
    new_pose.position.z = pose.position.z if z is None else z
    new_pose.orientation = pose.orientation
    return new_pose


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose_yaw(pose):
    return 2.0 * math.atan2(pose.orientation.y, pose.orientation.x)


def arc_waypoints(start, x_end, y_end, z):
    """Arc about BASE_XY at height z. The tool turns with the base (J4-J6 stay put);
    the carton yaw is set afterwards by the alignment step."""
    bx, by = BASE_XY
    r0 = math.hypot(start.position.x - bx, start.position.y - by)
    t0 = math.atan2(start.position.y - by, start.position.x - bx)
    r1 = math.hypot(x_end - bx, y_end - by)
    t1 = math.atan2(y_end - by, x_end - bx)
    dt = wrap_pi(t1 - t0)
    n = max(2, int(math.ceil(abs(math.degrees(dt)) / ARC_STEP_DEG)))
    yaw0 = pose_yaw(start)
    wps = []
    for k in range(1, n + 1):
        f = k / n
        r = r0 + (r1 - r0) * f
        t = t0 + dt * f
        wps.append(pose_yawed(
            pose_with(start, x=bx + r * math.cos(t), y=by + r * math.sin(t), z=z),
            yaw0 + dt * f))
    return wps, yaw0 + dt


def pick_and_place_one(node, idx, box, target_pose, slot_index):
    # Tracks whether vacuum release was confirmed.
    node.last_release_confirmed = False
    plan = None
    top_z0 = target_pose.position.z
    src_top = max((item['z_top'] for item in node.boxes), default=top_z0)
    cands = PLANNER.plan_all(box)
    print(f'{len(cands)} candidate spots; testing the full pick->transfer->place chain...')
    z_over = max(top_z0 + LIFT_HEIGHT, src_top + box['height'] + 0.05,
                 PALLET_HEIGHT + PLANNER.max_height() + box['height'] + 0.05)
    base_yaw = pose_yaw(target_pose)
    q_pick, why, pick_off = None, '', 0.0
    for off in (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2):
        tp = pose_yawed(target_pose, base_yaw + off)
        pick_over = pose_with(tp, z=z_over)
        q_pick, why = node.pick_chain(
            pose_with(tp, z=top_z0 + APPROACH_HEIGHT),
            pose_with(tp, z=top_z0 + GRASP_OFFSET),
            pose_with(tp, z=top_z0 + LIFT_HEIGHT),
            pick_over if z_over > top_z0 + LIFT_HEIGHT + 0.005 else None,
            quick=(off != 0.0))
        if q_pick is not None:
            target_pose, pick_off = tp, off
            if off:
                print(f'  using the tool turned {math.degrees(off):.0f} deg from the carton axis for this pick')
            break
        print(f'  pick with tool offset {math.degrees(off):.0f} deg not feasible ({why})')
    if q_pick is None:
        print('  pick sequence not feasible at any tool offset; skipping this carton')
        node.block(box)
        return False
    print('  pick sequence OK (descent, lift and rise dry-run passed)')
    for c in cands:
        z_place = PALLET_HEIGHT + c['support_h'] + box['height'] + PLACE_CLEARANCE
        res = node.transfer_feasible(pick_over, c, z_over, z_place, q0=q_pick, off=pick_off)
        tag = 'OK ' if res else 'rejected '
        print(f"  spot x={c['x']:.3f} y={c['y']:.3f} h={c['support_h']:.2f} "
              f"carton_yaw={c['yaw']:.2f}: {tag}({node.reach_note})")
        if res:
            c['transfer'], c['tool_yaw'] = res
            plan = c
            break
    if plan is None:
        print('Pallet full: no supported spot left.' if not cands else f'{len(cands)} free spots exist, but none is reachable by the arm (see lines above).')
        return False
    slot_x, slot_y = plan['x'] - DEST_CENTER_X, plan['y'] - DEST_CENTER_Y
    support_h = plan['support_h']
    print(f"Planned spot x={plan['x']:.3f} y={plan['y']:.3f} support_h={support_h:.2f}")
    print(
        f'Target at x={target_pose.position.x:.3f} y={target_pose.position.y:.3f} '
        f'z={target_pose.position.z:.3f}, carton_{idx}, destination slot {slot_index + 1}, size '
        f'{box["length"]:.2f}x{box["width"]:.2f}x{box["height"]:.2f}'
    )

    top_z = target_pose.position.z
    pre_grasp = pose_with(target_pose, z=top_z + APPROACH_HEIGHT)
    grasp = pose_with(target_pose, z=top_z + GRASP_OFFSET)
    lift = pose_with(target_pose, z=top_z + LIFT_HEIGHT)

    source_stack_top = max((item['z_top'] for item in node.boxes), default=top_z)
    destination_clear_z = PALLET_HEIGHT + PLANNER.max_height() + box['height'] + 0.05
    transfer_z = max(
        lift.position.z,
        source_stack_top + box['height'] + 0.05,
        destination_clear_z,
    )
    transfer_height = pose_with(lift, z=transfer_z)
    transfer_high = pose_with(transfer_height, x=plan['x'], y=plan['y'])
    overhead_slot = pose_yawed(pose_with(
        transfer_high,
        x=DEST_CENTER_X + slot_x,
        y=DEST_CENTER_Y + slot_y,
    ), plan.get('tool_yaw', plan['yaw']))
    place_pose = pose_with(
        overhead_slot,
        z=PALLET_HEIGHT + support_h + box['height'] + PLACE_CLEARANCE,
    )
    # Retreat vertically after releasing the carton.
    retreat_a = pose_with(place_pose, z=place_pose.position.z + 0.12)

    node.spin_for(0.5)   # let the scene manager's latest update arrive
    input('Enter: move to pre-grasp...')
    q_pre = getattr(node, 'pre_q', None)
    pre_ok = (node.move_joints(['J1', 'J2', 'J3', 'J4', 'J5', 'J6'], q_pre)
              if q_pre is not None else node.move_pose(pre_grasp))
    if not pre_ok:
        print('pre-grasp pose failed; the arm could not find a collision-safe approach pose.')
        return False

    input('Enter: remove target from scene and descend to grasp...')
    node.remove_cartons_overlapping(box, idx=idx)
    if not node.cartesian_segment([pre_grasp, grasp], 'descent to grasp'):
        print('descent failed; the carton is still in the source region.')
        return False

    input('Enter: vacuum ON; wait for physical attachment confirmation...')
    node.set_vacuum(True)
    if not node.wait_for_attachment(True):
        node.set_vacuum(False)
        node.wait_for_attachment(False)
        node.restore_world_carton(idx, box)
        print('Vacuum did not attach the carton; stopped before lift. Check TCP contact and /vacuum/attached.')
        return False
    print('Vacuum attachment confirmed. Updating MoveIt scene and lifting...')
    node.attach_carton(box)
    node.spin_for(0.3)
    node.remove_cartons_overlapping(box, idx=idx)
    if not node.cartesian_segment([grasp, lift], 'lift with carton'):
        print('MoveIt refused or failed the lift; vacuum attachment was confirmed. Do not continue to transfer.')
        return False

    if transfer_z > lift.position.z + 0.005:
        input('Enter: rise vertically to carton clearance height...')
        if not node.cartesian_segment([lift, transfer_height], 'vertical carton clearance'):
            print('vertical clearance failed; carton remains held.')
            return False

    input('Enter: move across to the destination pallet...')
    if plan.get('transfer') == 'arc':
        arc_wps, _ = arc_waypoints(transfer_height, plan['x'], plan['y'], transfer_z)
        transfer_ok = node.cartesian_segment(arc_wps, 'destination transfer (arc)')
    else:
        transfer_ok = node.cartesian_segment([transfer_height, transfer_high], 'destination transfer')
    if not transfer_ok:
        print('transfer rejected; trying a joint-space transfer in a safe wrist posture...')
        if not node.transfer_joint(transfer_high):
            print('transfer to destination failed; carton remains held.')
            return False

    input('Enter: align over the next pallet slot...')
    if not node.cartesian_segment([overhead_slot], 'destination slot alignment'):
        print('slot alignment failed; carton remains held.')
        return False

    input('Enter: descend to the destination pallet...')
    if not node.cartesian_segment(
        [overhead_slot, place_pose], 'vertical place descent', max_j123=5.2
    ):
        print('place descent failed; the carton could not be lowered safely to the pallet.')
        return False

    input('Enter: vacuum OFF and release the carton...')
    node.set_vacuum(False)
    if not node.wait_for_attachment(False):
        print('Vacuum release was not confirmed; carton remains attached.')
        return False
    node.detach_carton()
    node.last_release_confirmed = True

    # Plan from the actual current robot state to the vertical retreat target.
    # Do not include place_pose as the first waypoint.
    retreat_ok = node.cartesian_segment(
        [retreat_a], 'vertical retreat after place'
    )

    # Fallback: collision-aware joint-space planning to the same target.
    if not retreat_ok:
        print('Cartesian retreat failed; trying joint-space retreat...')
        retreat_ok = node.move_pose(retreat_a)

    PLANNER.commit(plan, box)
    node.add_placed_carton(box, slot_index, plan)

    if not retreat_ok:
        print(
            'Carton was released and recorded, but the arm could not retreat. '
            'Stopping for manual recovery.'
        )
        return False

    print(f'Carton placed successfully in slot {slot_index + 1}.')
    node.rise_pose = retreat_a
    return True


def main():
    rclpy.init()
    node = SafeCartonPicker()
    node.add_environment()

    placed_count = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f'Starting at destination slot {placed_count + 1}')
    while rclpy.ok() and placed_count < MAX_DEST_CARTONS:
        idx, box, target_pose = node.get_next_carton()
        if box is None:
            if node.boxes_fresh:
                msg = f'No more stable pickable cartons. Placed {placed_count}.'
                if getattr(node, 'blocked', None):
                    msg += (f' {len(node.blocked)} carton(s) were skipped (pick dry run failed) '
                            f'(x, y, z_top): {node.blocked}')
                print(msg)
            else:
                print('No fresh carton detection received; stopping before another pick.')
            break

        if not pick_and_place_one(node, idx, box, target_pose, placed_count):
            if getattr(node, 'skip_ok', False):
                node.skip_ok = False
                print('>> carton skipped, looking for another candidate')
                node.spin_for(0.5)
                continue
            if getattr(node, 'last_release_confirmed', False):
                placed_count += 1
                print(
                    f'Carton released successfully. Counted {placed_count} placements; '
                    'stopping because the arm retreat failed.'
                )
            else:
                print(f'Stopping after {placed_count} completed carton placements.')
            break
        placed_count += 1
        if placed_count >= MAX_DEST_CARTONS:
            break
        node.wait_fresh_scan(timeout=6.0)
        if node.get_next_carton(timeout=0.5)[1] is None:
            print('>> no carton visible, going home to clear the camera view')
            if not node.go_home(getattr(node, 'rise_pose', None)):
                print('Home move failed; stopping.')
                break
            node.wait_fresh_scan()

    if placed_count == MAX_DEST_CARTONS:
        print(f'Destination slot capacity reached: {placed_count} cartons.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()