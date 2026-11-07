#!/usr/bin/env python3
#add cols to.py
import math, rclpy
from rclpy.node import Node
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

# id: (size_xyz, centre_xyz, yaw)  <- EDIT to match your Gazebo world
STATIC = {
    'floor':         ((8.0, 8.0, 0.1),  (0.0, 0.0, -0.05), 0.0),
    'wall_north':    ((8.4, 0.2, 3.0),  (0.0,  4.1, 1.5), 0.0),
    'wall_south':    ((8.4, 0.2, 3.0),  (0.0, -4.1, 1.5), 0.0),
    'wall_east':     ((0.2, 8.0, 3.0),  (4.1, 0.0, 1.5), 0.0),
    'wall_west':     ((0.2, 8.0, 3.0),  (-4.1, 0.0, 1.5), 0.0),
    'source_pallet': ((0.6, 0.6, 0.15), (0.80, -0.55, 0.075), 0.0),
    'dest_pallet':   ((0.6, 0.6, 0.15), (-0.35, -0.95, 0.075), 0.0),
}

def make_box(obj_id, size, xyz, yaw, frame='world'):
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = obj_id
    co.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(size)))
    p = Pose()
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.z = math.sin(yaw / 2); p.orientation.w = math.cos(yaw / 2)
    co.primitive_poses.append(p)
    co.operation = CollisionObject.ADD
    return co

def main():
    rclpy.init()
    n = Node('static_scene')
    c = n.create_client(ApplyPlanningScene, '/apply_planning_scene')
    c.wait_for_service()
    scene = PlanningScene(is_diff=True)
    for k, (s, xyz, yaw) in STATIC.items():
        scene.world.collision_objects.append(make_box(k, s, xyz, yaw))
    f = c.call_async(ApplyPlanningScene.Request(scene=scene))
    rclpy.spin_until_future_complete(n, f)
    print('applied:', f.result().success)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
