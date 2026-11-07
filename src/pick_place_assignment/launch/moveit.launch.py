import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
#moveit2 launch file

def generate_launch_description():
    depal = get_package_share_directory('pick_place_assignment')
    fanuc_moveit = get_package_share_directory('fanuc_moveit_config')

    xacro_path = os.path.join(depal, 'urdf', 'depal_robot.urdf.xacro')

    mc = (
        MoveItConfigsBuilder('crx10ia_l', package_name='fanuc_moveit_config')
        .robot_description(file_path=xacro_path)
        .robot_description_semantic(file_path='srdf/crx10ia_l.srdf')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_scene_monitor(publish_robot_description=True,
                                publish_robot_description_semantic=True)
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )

    # FANUC's SRDF does not know our pedestal and gripper, so add
    # "these links touch by design" entries to a copy of it.
    with open(os.path.join(fanuc_moveit, 'srdf', 'crx10ia_l.srdf')) as f:
        srdf = f.read()
    extra = ''
    for a, b in [('pedestal', 'base_link'), ('pedestal', 'J1_link'),
                 ('gripper_link', 'J6_link'), ('gripper_link', 'J5_link'), ('gripper_link', 'J4_link'), ('gripper_link', 'flange'),
                 ('gripper_link', 'fanuc_flange'), ('gripper_link', 'ee_link')]:
        extra += f'  <disable_collisions link1="{a}" link2="{b}" reason="Adjacent"/>\n'
    srdf = srdf.replace('</robot>', extra + '</robot>')

    params = mc.to_dict()
    params['robot_description_semantic'] = srdf
    params['use_sim_time'] = True

    move_group = Node(package='moveit_ros_move_group', executable='move_group',
                      output='screen', parameters=[params])

    rviz = Node(
        package='rviz2', executable='rviz2', output='log',
        arguments=['--display-config',
                   os.path.join(fanuc_moveit, 'rviz', 'view_robot.rviz')],
        parameters=[mc.robot_description,
                    {'robot_description_semantic': srdf},
                    mc.planning_pipelines,
                    mc.robot_description_kinematics,
                    mc.joint_limits,
                    {'use_sim_time': True}])

    return LaunchDescription([move_group, rviz])