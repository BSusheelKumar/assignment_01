# carton_picker

Collision-aware depalletizer for MoveIt 2. Picks cartons from the source pallet (using
`/perception/boxes_json`) and stacks them on the destination pallet.

    python3 -m carton_picker              # autonomous
    python3 -m carton_picker --step       # press Enter before each stage
    python3 -m carton_picker --start-slot 2   # 2 cartons already on the destination

## Layout
| file | job |
|---|---|
| `config.py` | every tunable number, with the reason for it |
| `models.py` | `Carton`, `Spot`, `PickPlan`, `Outcome` (no ROS) |
| `geometry.py` | tool-down poses, yaw helpers, base-centred arc |
| `dest_planner.py` | height map of the destination pallet, ranked placement spots (no ROS) |
| `path_checks.py` | joint-path sanity: jumps, elbow flips, wrist singularity (no ROS) |
| `postures.py` | IK seeded + filtered to a safe posture window |
| `planning.py` | dry run of pick -> transfer -> place before the arm moves |
| `motion.py` | joint moves, checked Cartesian segments, go-home |
| `moveit_client.py` | ROS plumbing: services, actions, scene, perception, vacuum |
| `workflow.py` | the pick-and-place sequence and main loop |

## Method
Long Cartesian moves fail midway, so motion is staged: pre-grasp -> descent -> lift -> rise ->
transfer (arc about the base) -> align -> descent -> release -> retreat. Each stage is dry-run
from the previous stage's end state; a spot or tool orientation is chosen only if the whole
chain passes. Executed segments are re-checked for joint jumps and collisions.

## Known limitations
- Posture windows (`postures.py`) and path limits (`config.py`) are tuned for this CRX and layout.
- Placement assumes cartons of 0.2 / 0.3 / 0.4 m edges (`CARTON_SIZES`).
- `ALLOW_TOOL_FLIP` assumes the gripper is symmetric under a 180 deg turn.
- Top-down depth cannot tell one lying carton from two standing ones (detector heuristic).
- Tests cover the ROS-free modules only; the motion stack needs the simulator.
