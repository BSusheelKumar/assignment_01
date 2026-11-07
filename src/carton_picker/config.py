"""Every tunable number, with the reason it exists. No logic lives here."""
import math

# --- MoveIt names -----------------------------------------------------------
GROUP = 'manipulator'
TIP = 'tcp'                      # suction point; GRASP_OFFSET is measured from here
WORLD = 'world'
JOINTS = ('J1', 'J2', 'J3', 'J4', 'J5', 'J6')
GRIPPER_LINK = 'gripper_link'
# Links allowed to touch a held carton (otherwise the attached box "collides" with the gripper).
TOUCH_LINKS = ['gripper_link', 'ee_link', 'tcp', 'J6_link', 'flange', 'fanuc_flange',
               'J5_link', 'J4_link']
HOME = [1.57, 0.0, 0.0, 0.0, 0.0, 0.0]   # arm folded up so the camera sees the pallet

# --- Layout (metres, world frame) -------------------------------------------
SOURCE_CENTER = (0.80, -0.55)
DEST_CENTER = (-0.35, -0.95)
PALLET_LENGTH = 0.60
PALLET_WIDTH = 0.60
PALLET_HEIGHT = 0.15
BASE_XY = (0.0, 0.0)             # world x,y of the J1 axis (read it from tf2_echo)
MAX_REACH = 1.20                 # horizontal base->TCP limit; CRX-10iA/L max is ~1.25
MAX_STACK = 0.85                 # max stack height above the pallet top
MAX_DEST_CARTONS = 6

# Carton edge lengths in the warehouse; measured sizes within SIZE_SNAP_TOL snap to these
# so depth-image noise does not leak into the destination layout.
CARTON_SIZES = (0.2, 0.3, 0.4)
SIZE_SNAP_TOL = 0.03

# --- Motion heights ---------------------------------------------------------
APPROACH_HEIGHT = 0.22           # pre-grasp height above carton top
GRASP_OFFSET = 0.0               # TCP height above carton top at grasp (0 = touching)
LIFT_HEIGHT = 0.45               # straight lift after grasp
PLACE_CLEARANCE = 0.015          # carton bottom stays this far above its support when released
RETREAT_HEIGHT = 0.12            # straight up after release
CARRY_MARGIN = 0.05              # extra clearance over the tallest carton while transferring

# --- Tool orientation -------------------------------------------------------
# Tool offsets tried (radians from the carton axis) when the aligned pick is infeasible.
TOOL_OFFSETS = (0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2)
# Only valid if the gripper looks the same after a 180 deg turn about the tool axis.
ALLOW_TOOL_FLIP = True
ARC_STEP_DEG = 6.0               # waypoint spacing of the transfer arc about the base

# --- Speed ------------------------------------------------------------------
JOINT_SCALE = 0.25               # velocity / acceleration scaling for joint-space moves
CARTESIAN_SCALE = 0.12           # slower for straight lines next to cartons
AUTO_PAUSE_S = 0.5               # pause between stages when not in --step mode

# --- Cartesian path sanity --------------------------------------------------
CART_STEP = 0.005                # interpolation resolution [m]
CART_MIN_FRACTION = 0.98         # accept a Cartesian path only if >= this much was solved
PRISMATIC_JUMP = 0.005
REVOLUTE_JUMP = 0.35             # MoveIt's own jump check; ours (below) is tighter
MAX_JOINT_STEP = 0.25            # rad between 5 mm samples; real IK branch flips are ~3 rad
MAX_J1_SWING = 3.0               # rad of total base travel in one move
MAX_J23_TRAVEL = 1.5             # rad; more means the elbow is folding / flipping
PLACE_DESCENT_J23_TRAVEL = 5.2   # the final descent legitimately moves J2/J3 a lot
MAX_WRIST_TRAVEL = 3.5           # rad per wrist joint
MIN_J5 = math.radians(15)        # closer to 0 than this is the wrist singularity
MAX_WINDING = 2.0                # joint-path length / net joint displacement
VALIDITY_STRIDE = 4              # collision-check every Nth trajectory point
