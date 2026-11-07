import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.actions import RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('pick_place_assignment')
    fanuc_share = os.path.dirname(get_package_share_directory('fanuc_crx_description'))
    xacro_file = os.path.join(pkg, 'urdf', 'depal_mobile_manipulator.urdf.xacro')
    world_file = os.path.join(pkg, 'worlds', 'warehouse.sdf')

    resource_path = fanuc_share + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    plugin_path = os.path.join(get_package_prefix('pick_place_assignment'), 'lib') + ':' + \
        os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world_file]}.items())

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
               output='screen')
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'depal_mobile_manipulator',
                            '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'),
                            '-z', '0.0', '-Y', LaunchConfiguration('yaw')])

    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  arguments=[
                      '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                      '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                      '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                      '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
                      '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
                      # Fortress publishes the color image on the sensor's
                      # <topic> value itself, not a suffixed /image topic.
                      '/mobile_camera@sensor_msgs/msg/Image[ignition.msgs.Image',
                      '/mobile_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
                      '/mobile_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
                      '/mobile_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
                      '/vacuum/enable@std_msgs/msg/Bool]ignition.msgs.Boolean',
                      '/vacuum/attached@std_msgs/msg/Bool[ignition.msgs.Boolean',
                  ])

    jsb = Node(package='controller_manager', executable='spawner', output='screen',
               arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'])
    jtc = Node(package='controller_manager', executable='spawner', output='screen',
               arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'])

    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='-1.5', description='Initial robot x position'),
        DeclareLaunchArgument('y', default_value='0.0', description='Initial robot y position'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='Initial robot yaw'),
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path),
        SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', plugin_path),
        gz_sim, rsp, bridge, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc])),
    ])
