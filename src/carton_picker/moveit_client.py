"""Thin ROS 2 wrapper: services, actions, planning scene, perception, vacuum.
Everything here is plumbing; decisions live in planning.py and workflow.py."""
import json
import time

import rclpy
from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (AttachedCollisionObject, CollisionObject, Constraints,
                             JointConstraint, PlanningScene, PlanningSceneComponents,
                             RobotState)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetPlanningScene,
                             GetPositionIK, GetStateValidity)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory

from . import config
from .geometry import box_pose
from .models import Carton


class MoveItClient(Node):
    def __init__(self):
        super().__init__('safe_carton_picker')
        self._move_group = ActionClient(self, MoveGroup, '/move_action')
        self._execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._cartesian = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self._scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self._get_scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self._validity = self.create_client(GetStateValidity, '/check_state_validity')
        self._ik = self.create_client(GetPositionIK, '/compute_ik')
        self._move_group.wait_for_server()
        self._execute.wait_for_server()
        for c in (self._cartesian, self._scene, self._get_scene, self._validity, self._ik):
            c.wait_for_service()

        self._vacuum_pub = self.create_publisher(Bool, '/vacuum/enable', 10)
        self.vacuum_attached = None
        self.create_subscription(Bool, '/vacuum/attached', self._on_vacuum, 10)

        self.boxes = None            # list[Carton] from the last fresh detection
        self.boxes_fresh = False
        self.create_subscription(String, '/perception/boxes_json', self._on_boxes, 10)

        self.joints = {}
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

    # ---- callbacks / spinning ---------------------------------------------
    def _on_vacuum(self, msg):
        self.vacuum_attached = msg.data

    def _on_boxes(self, msg):
        decoded = json.loads(msg.data)
        self.boxes_fresh = not decoded.get('stale', False)
        if self.boxes_fresh:
            self.boxes = [Carton.from_dict(b) for b in decoded.get('boxes', [])]

    def _on_joint_state(self, msg):
        self.joints = dict(zip(msg.name, msg.position))

    def spin_for(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait(self, future):
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    # ---- perception --------------------------------------------------------
    def wait_for_boxes(self, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline and not self.boxes_fresh:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.boxes if self.boxes_fresh else None

    def wait_fresh_scan(self, timeout=20.0):
        """Discard the old detection and wait for a new one (after the arm moved away)."""
        self.boxes_fresh, self.boxes = False, None
        self.wait_for_boxes(timeout)
        self.spin_for(1.5)

    # ---- kinematics / planning --------------------------------------------
    def compute_ik(self, pose, seed_rad, timeout_s):
        """Collision-aware IK from a seed. Returns joints in J1..J6 order or None."""
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = config.GROUP
        r.avoid_collisions = True
        r.ik_link_name = config.TIP
        r.pose_stamped.header.frame_id = config.WORLD
        r.pose_stamped.pose = pose
        r.timeout = Duration(sec=0, nanosec=int(timeout_s * 1e9))
        r.robot_state.joint_state.name = list(config.JOINTS)
        r.robot_state.joint_state.position = list(seed_rad)
        res = self._wait(self._ik.call_async(req))
        if res is None or res.error_code.val != 1:
            return None
        js = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [js[n] for n in config.JOINTS]

    def compute_cartesian(self, waypoints, start_q=None, scale=None):
        """Raw GetCartesianPath. start_q=None plans from the current state (dry runs pass one)."""
        req = GetCartesianPath.Request()
        req.header.frame_id = config.WORLD
        if start_q is None:
            req.start_state = RobotState(is_diff=True)
        else:
            req.start_state.joint_state.name = list(config.JOINTS)
            req.start_state.joint_state.position = list(start_q)
        req.group_name = config.GROUP
        req.link_name = config.TIP
        req.waypoints = list(waypoints)
        req.max_step = config.CART_STEP
        req.jump_threshold = 0.0
        req.prismatic_jump_threshold = config.PRISMATIC_JUMP
        req.revolute_jump_threshold = config.REVOLUTE_JUMP
        req.avoid_collisions = True
        if scale is not None:
            req.max_velocity_scaling_factor = scale
            req.max_acceleration_scaling_factor = scale
        return self._wait(self._cartesian.call_async(req))

    def send_joint_goal(self, q):
        """Plan and execute to a joint target. Returns MoveIt's error code (1 = success) or None."""
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = config.GROUP
        req.num_planning_attempts = 12
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = config.JOINT_SCALE
        req.max_acceleration_scaling_factor = config.JOINT_SCALE
        req.start_state.is_diff = True
        c = Constraints()
        for name, v in zip(config.JOINTS, q):
            jc = JointConstraint()
            jc.joint_name, jc.position, jc.weight = name, v, 1.0
            jc.tolerance_above = jc.tolerance_below = 0.001
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)
        handle = self._wait(self._move_group.send_goal_async(goal))
        if not handle or not handle.accepted:
            return None
        return self._wait(handle.get_result_async()).result.error_code.val

    def execute(self, trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        handle = self._wait(self._execute.send_goal_async(goal))
        if not handle or not handle.accepted:
            return None
        return self._wait(handle.get_result_async()).result.error_code.val

    def state_validity(self, q=None):
        """GetStateValidity for joints q (or the current state if None). Response or None."""
        req = GetStateValidity.Request()
        req.group_name = config.GROUP
        req.robot_state.is_diff = True
        if q is not None:
            req.robot_state.joint_state.name = list(config.JOINTS)
            req.robot_state.joint_state.position = list(q)
        return self._wait(self._validity.call_async(req))

    def publish_trajectory(self, msg):
        self._traj_pub.publish(msg)

    # ---- planning scene ----------------------------------------------------
    def apply_scene(self, scene, strict=True):
        res = self._wait(self._scene.call_async(ApplyPlanningScene.Request(scene=scene)))
        if res is None or not res.success:
            if strict:
                raise RuntimeError('MoveIt rejected the planning-scene update')
            self.get_logger().warn('scene update rejected (non-fatal)')
        self.spin_for(0.3)

    @staticmethod
    def _box(object_id, dims, pose):
        obj = CollisionObject()
        obj.header.frame_id = config.WORLD
        obj.id = object_id
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(dims)))
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        return obj

    @staticmethod
    def _remove(object_id):
        obj = CollisionObject()
        obj.header.frame_id = config.WORLD
        obj.id = object_id
        obj.operation = CollisionObject.REMOVE
        return obj

    def add_environment(self):
        scene = PlanningScene(is_diff=True)
        for name, (cx, cy) in (('source_pallet', config.SOURCE_CENTER),
                               ('dest_pallet', config.DEST_CENTER)):
            scene.world.collision_objects.append(self._box(
                name, (config.PALLET_LENGTH, config.PALLET_WIDTH, config.PALLET_HEIGHT),
                box_pose(cx, cy, config.PALLET_HEIGHT / 2.0, 0.0)))
        self.apply_scene(scene)

    def remove_cartons_overlapping(self, c, idx, margin=0.06):
        """Remove scene carton `carton_<idx>` plus any scene carton sitting at the same
        footprint, so the one being picked cannot block its own grasp. (A box's pose is
        object pose + primitive pose; rotation is ignored: footprints are compared on x,y.)"""
        res = self._wait(self._get_scene.call_async(GetPlanningScene.Request(
            components=PlanningSceneComponents(
                components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY))))
        if res is None:
            return []
        hx = c.snapped_length / 2 + margin
        hy = c.snapped_width / 2 + margin
        removed = [f'carton_{idx}']
        for o in res.scene.world.collision_objects:
            if not o.id.startswith('carton_') or not o.primitive_poses or o.id in removed:
                continue
            p = o.primitive_poses[0].position
            x, y, z = (o.pose.position.x + p.x, o.pose.position.y + p.y, o.pose.position.z + p.z)
            if (abs(x - c.x) < hx and abs(y - c.y) < hy
                    and z > c.z_top - c.height - 0.05):   # centre above this carton's bottom
                removed.append(o.id)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.extend(self._remove(i) for i in removed)
        self.apply_scene(scene, strict=False)
        self.get_logger().info(f'removed scene cartons: {removed}')
        return removed

    def restore_carton(self, idx, c):
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(self._box(
            f'carton_{idx}', (c.length, c.width, c.height),
            box_pose(c.x, c.y, c.z_center, c.yaw)))
        self.apply_scene(scene)

    def attach_carton(self, c):
        """Attach a slightly shrunken box to the gripper (the suction cup sits on the lid,
        and perception edges are a few cm generous)."""
        obj = self._box('held_carton',
                        (c.snapped_length - 0.08, c.snapped_width - 0.08, c.height - 0.02),
                        box_pose(c.x, c.y, c.z_center, c.yaw))
        attach = AttachedCollisionObject()
        attach.link_name = config.GRIPPER_LINK
        attach.object = obj
        attach.touch_links = config.TOUCH_LINKS
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(attach)
        self.apply_scene(scene)

    def detach_carton(self):
        detach = AttachedCollisionObject()
        detach.link_name = config.GRIPPER_LINK
        detach.object = CollisionObject(id='held_carton', operation=CollisionObject.REMOVE)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(detach)
        self.apply_scene(scene)

    def add_placed_carton(self, c, slot_index, spot):
        z = config.PALLET_HEIGHT + spot.support_h + c.height / 2.0
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(self._box(
            f'placed_carton_{slot_index}',
            (c.snapped_length, c.snapped_width, c.height),
            box_pose(spot.x, spot.y, z, spot.yaw)))
        self.apply_scene(scene)

    # ---- vacuum ------------------------------------------------------------
    def set_vacuum(self, enabled):
        self.spin_for(0.3)
        self._vacuum_pub.publish(Bool(data=enabled))
        self.spin_for(0.2)

    def wait_for_attachment(self, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.vacuum_attached is expected:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False
