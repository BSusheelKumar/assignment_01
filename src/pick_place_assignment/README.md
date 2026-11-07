# FANUC CRX carton depalletization

`pick_place_assignment` is a ROS 2 package for simulating carton detection and pick-and-place with a FANUC CRX-10iA/L, an overhead RGB-D camera, and a simulated vacuum gripper. The default simulation contains five 5 kg cartons on the source pallet and an empty destination pallet.

The package directory is part of a larger colcon workspace. In this checkout, the sibling `fanuc_description` and `fanuc_driver` source trees provide `fanuc_crx_description` and `fanuc_moveit_config`. Those are runtime/build dependencies of the launch workflow; they are not copied into this package itself.

## Requirements

- Ubuntu 22.04 with ROS 2 Humble, including `ros-humble-desktop` and `ros-dev-tools`.
- Gazebo Fortress (`ignition-gazebo6`) and the ROS-Gazebo bridge packages.
- MoveIt 2 and the FANUC description and MoveIt configuration packages included in this workspace (or installed separately).
- A graphical desktop for Gazebo and RViz.

The package also uses PCL, OpenCV, NumPy, PyYAML, xacro, and ROS 2 control. The steps below use `rosdep` to install system dependencies declared by the workspace. If the FANUC packages are not present under `src/`, make sure they are installed and discoverable in the sourced ROS environment before building.

## Build the workspace

Run from the workspace root (the directory containing `src/`):

```bash
cd ~/Documents/assignment
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install -y python3-rosdep python3-colcon-common-extensions
sudo rosdep init  # only needed once on this machine; skip if already initialized
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to pick_place_assignment
source install/setup.bash
```

If `rosdep init` reports that it has already been initialized, continue with `rosdep update`. The build may compile the workspace packages needed by `pick_place_assignment`, including the FANUC packages.

## Launch the simulation and run pick-and-place

Use two terminals. In the first, start Gazebo, the robot, controllers, MoveIt/RViz, camera bridge, perception, and planning-scene nodes:

```bash
cd ~/Documents/assignment
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch pick_place_assignment sim.launch.py
```

Wait for the robot, controllers, MoveIt, and camera to finish starting. In the second terminal, source the same environment and start the pick-and-place application:

```bash
cd ~/Documents/assignment
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run pick_place_assignment pick_place_carton_safe.py
```

To start with destination slot numbering after two cartons have already been placed, pass `2`:

```bash
ros2 run pick_place_assignment pick_place_carton_safe.py 2
```

The static collision scene helper starts automatically with `sim.launch.py`. Stop the nodes with Ctrl+C in their respective terminals.

## Select a source layout

The default layout is `config/layouts/default.yaml`.

```bash
ros2 launch pick_place_assignment sim.launch.py 
```

Layouts specify pallet center/top height, carton mass, and each carton’s center offset, dimensions, and yaw. Keep cartons supported by the pallet and non-overlapping. The detector and motion application also contain fixed assumptions about carton sizes, pallet coordinates, frames, and robot group/tool names; changing the YAML alone does not update those assumptions.

## Other launch files

- `sim.launch.py`: complete fixed-base simulation and perception/planning stack (recommended entry point).
- `moveit.launch.py`: MoveIt and RViz for an already-running robot description and controllers.
- `perception.launch.py`: overhead camera transform and depth-image carton detector.
- `mobile_manipulator.launch.py`: separate mobile-manipulator simulation; accepts `x`, `y`, and `yaw` launch arguments. This is not the fixed-pallet `sim.launch.py` workflow.

## Package contents

- `config/`: controller settings, home pose, and source pallet layouts.
- `launch/`: simulation, MoveIt, perception, and mobile-manipulator launch descriptions.
- `scripts/`: Python depth detector, planning-scene synchronizer, pick-and-place application, and static collision-scene helper.
- `src/`: optional PCL carton detector and Gazebo vacuum-gripper plugin.
- `urdf/`: robot and mobile-manipulator descriptions.
- `worlds/`: warehouse, pallets, and overhead RGB-D camera.

The integrated launch workflow uses `box_detector.py`; the C++ point-cloud detector is an additional executable and is not started by `sim.launch.py`.

## Topics and frames

- Camera input: `/camera/depth_image`, `/camera/camera_info`
- Detection output: `/perception/boxes_json`; markers: `/perception/markers`; debug image: `/perception/debug_image`
- Gripper command/state: `/vacuum/enable`, `/vacuum/attached`
- Planning frame: `world`; tool frame: `tcp`

## Known limits

The detector estimates cartons from the overhead depth view and supports the carton dimensions configured in the detector. The motion code assumes this warehouse’s pallet positions, dimensions, world frame, and robot group/tool names. Update the related detector and motion constants when changing the scene. This package is a simulation deliverable; test changed layouts in simulation before using any resulting motion on a physical robot.

The assignment also requires a demonstration video. No video is included in this source package; record a simulation run and submit the video separately.
