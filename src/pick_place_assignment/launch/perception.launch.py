from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # Match the camera link pose in warehouse.sdf: translation (0.7, 0, 2.2)
    # and +90 degrees pitch. Gazebo's point cloud frame is forward/left/up.
    camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=['--x', '0.7', '--y', '0', '--z', '2.2',
                   '--qx', '0', '--qy', '0.7071068', '--qz', '0', '--qw', '0.7071068',
                   '--frame-id', 'world', '--child-frame-id', 'overhead_camera/link/rgbd'])
    detector = Node(package='pick_place_assignment', executable='box_detector.py',
                    output='screen', parameters=[{'use_sim_time': True}])
    return LaunchDescription([camera_tf, detector])
