import os
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            ExecuteProcess, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
#sim launch file

def box_sdf(name, dims, mass):
    dx, dy, dz = dims
    ixx = mass / 12.0 * (dy**2 + dz**2)
    iyy = mass / 12.0 * (dx**2 + dz**2)
    izz = mass / 12.0 * (dx**2 + dy**2)
    return f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia><ixx>{ixx}</ixx><iyy>{iyy}</iyy><izz>{izz}</izz>
                 <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="c">
        <geometry><box><size>{dx} {dy} {dz}</size></box></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
      </collision>
      <visual name="v">
        <geometry><box><size>{dx} {dy} {dz}</size></box></geometry>
        <material><ambient>0.75 0.55 0.3 1</ambient><diffuse>0.75 0.55 0.3 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


def spawn_boxes(context):
    pkg = get_package_share_directory('pick_place_assignment')
    layout = LaunchConfiguration('layout').perform(context)
    with open(os.path.join(pkg, 'config', 'layouts', layout + '.yaml')) as f:
        data = yaml.safe_load(f)
    cx, cy = data['pallet']['center']
    top = data['pallet']['top_height']
    mass = data.get('mass', 5.0)
    nodes = []
    for i, b in enumerate(data['boxes'], start=1):
        name = f'box_{i}'
        dx, dy, dz = b['dims']
        nodes.append(Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-string', box_sdf(name, b['dims'], mass), '-name', name,
                       '-x', str(cx + b['x']), '-y', str(cy + b['y']),
                       '-z', str(top + b['z'] + dz / 2.0 + 0.002),
                       '-Y', str(b.get('yaw', 0.0))]))
    return nodes


def generate_launch_description():
    pkg = get_package_share_directory('pick_place_assignment')
    world = os.path.join(pkg, 'worlds', 'warehouse.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'depal_robot.urdf.xacro')

    fanuc_share = os.path.dirname(get_package_share_directory('fanuc_crx_description'))
    resource_path = fanuc_share + ':' + os.environ.get('IGN_GAZEBO_RESOURCE_PATH', '')
    set_res = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', resource_path)

    plugin_path = os.path.join(get_package_prefix('pick_place_assignment'), 'lib') + ':' + \
        os.environ.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', '')
    set_plugin = SetEnvironmentVariable('IGN_GAZEBO_SYSTEM_PLUGIN_PATH', plugin_path)
    layout_arg = DeclareLaunchArgument('layout', default_value='default')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items())

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'moveit.launch.py')))
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'perception.launch.py')))
    scene_manager = Node(package='pick_place_assignment', executable='scene_manager.py',
                         output='screen', parameters=[{'use_sim_time': True}])
    static_scene = Node(package='pick_place_assignment', executable='add_cols_to.py', output='screen')

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_description, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create',
                 arguments=['-topic', 'robot_description', '-name', 'crx10ia_l'],
                 output='screen')
    clock_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
                                   '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
                                   '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
                                   '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
                                   '/vacuum/enable@std_msgs/msg/Bool]ignition.msgs.Boolean',
                                   '/vacuum/attached@std_msgs/msg/Bool[ignition.msgs.Boolean',
                                   '/camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
                                   ],
                        output='screen')
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'])
    jtc = Node(package='controller_manager', executable='spawner',
               arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'])
    load_jsb = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb]))
    load_jtc = RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc]))
    set_initial_arm_pose = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--once',
             '/joint_trajectory_controller/joint_trajectory',
             'trajectory_msgs/msg/JointTrajectory',
             '{joint_names: [J1, J2, J3, J4, J5, J6], '
             'points: [{positions: [1.5707963, 0.0, 0.0, 0.0, 0.0, 0.0], '
             'time_from_start: {sec: 3}}]}'],
        output='screen')
    initialize_arm = RegisterEventHandler(
        OnProcessExit(target_action=jtc, on_exit=[set_initial_arm_pose]))

    return LaunchDescription([set_res, set_plugin, layout_arg, gz_sim, rsp, clock_bridge,
                              spawn, load_jsb, load_jtc, initialize_arm,
                              OpaqueFunction(function=spawn_boxes),
                              moveit, perception, scene_manager, static_scene])
