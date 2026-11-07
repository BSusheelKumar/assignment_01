#!/usr/bin/env python3
#scene manager.py
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import math

PALLET_TOP = 0.15   # world z of the source pallet surface; match the detector's pallet_top
MARGIN = 0.02       # extra size so the arm keeps a little clearance
PALLET_SIZE = (0.6, 0.6, 0.15)
PALLETS = {
    'source_pallet': (0.80, -0.55),
    'dest_pallet': (-0.35, -0.95),
}


class SceneManager(Node):
    def __init__(self):
        super().__init__('scene_manager')
        self.cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.cli.wait_for_service()
        self.old_ids = []
        self.create_subscription(String, '/perception/boxes_json', self.on_boxes, 10)
        self.get_logger().info('scene_manager ready')

    def on_boxes(self, msg):
        data = json.loads(msg.data)
        if data.get('stale'):          # arm is over the pallet: keep the previous scene
            return
        scene = PlanningScene(is_diff=True)

        # Gazebo has both pallets as static collision geometry. Mirror them in
        # MoveIt's world too, otherwise planning can send the arm through them.
        for object_id, (x, y) in PALLETS.items():
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.id = object_id
            co.primitives.append(SolidPrimitive(
                type=SolidPrimitive.BOX, dimensions=list(PALLET_SIZE)))
            p = Pose()
            p.position.x, p.position.y = x, y
            p.position.z = PALLET_TOP / 2.0
            p.orientation.w = 1.0
            co.primitive_poses.append(p)
            co.operation = CollisionObject.ADD
            scene.world.collision_objects.append(co)

        # remove what we added last time
        for oid in self.old_ids:
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.id = oid
            co.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(co)

        # add one solid column per detected carton, from pallet surface to its top face
        new_ids = []
        for i, b in enumerate(data['boxes']):
            h = b['z_top'] - PALLET_TOP
            if h <= 0.01:
                continue
            if max(b['length'], b['width']) > 0.5:   # unsegmented blob, not one carton
                continue
            co = CollisionObject()
            co.header.frame_id = 'world'
            co.id = f'carton_{i}'
            co.primitives.append(SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=[b['length'] + MARGIN, b['width'] + MARGIN, h]))
            p = Pose()
            p.position.x, p.position.y = b['x'], b['y']
            p.position.z = PALLET_TOP + h / 2.0
            p.orientation.z = math.sin(b['yaw'] / 2.0)
            p.orientation.w = math.cos(b['yaw'] / 2.0)
            co.primitive_poses.append(p)
            co.operation = CollisionObject.ADD
            scene.world.collision_objects.append(co)
            new_ids.append(co.id)

        self.old_ids = new_ids
        self.cli.call_async(ApplyPlanningScene.Request(scene=scene))


def main():
    rclpy.init()
    rclpy.spin(SceneManager())


if __name__ == '__main__':
    main()
