"""Tests for the parts that need no ROS. Run: python3 -m pytest tests -q (from the parent dir)."""
import math
import sys
import types

# geometry.py imports geometry_msgs; stub it when ROS is not installed.
try:
    import geometry_msgs.msg  # noqa: F401
except ImportError:
    class _V:  # attribute bag
        def __init__(self): self.x = self.y = self.z = self.w = 0.0
    class Pose:
        def __init__(self): self.position, self.orientation = _V(), _V()
    gm = types.ModuleType('geometry_msgs'); gmm = types.ModuleType('geometry_msgs.msg')
    gmm.Pose = Pose; gm.msg = gmm
    sys.modules['geometry_msgs'] = gm; sys.modules['geometry_msgs.msg'] = gmm

from carton_picker import config
from carton_picker.dest_planner import DestPlanner
from carton_picker.geometry import arc_waypoints, pose_yaw, tool_down_pose
from carton_picker.models import Carton, snap_size
from carton_picker.path_checks import PathLimits, violations
from carton_picker.postures import PICK, TRANSFER


def carton(l=0.4, w=0.3, h=0.2):
    return Carton(x=0.8, y=-0.55, z_top=0.35, yaw=0.0, length=l, width=w, height=h)


def test_snap():
    assert snap_size(0.41) == 0.4 and snap_size(0.25) == 0.25


def test_yaw_roundtrip():
    for yaw in (-1.2, 0.0, 0.7, 1.5):
        assert abs(pose_yaw(tool_down_pose(0, 0, 0, yaw)) - yaw) < 1e-9


def test_arc_ends_at_target():
    wps, yaw_end = arc_waypoints(tool_down_pose(0.8, -0.2, 0.7, 0.0), -0.35, -0.95, 0.7)
    assert abs(wps[-1].position.x + 0.35) < 1e-9 and abs(wps[-1].position.y + 0.95) < 1e-9
    assert all(abs(w.position.z - 0.7) < 1e-9 for w in wps)


def test_dest_planner_fills_and_stacks():
    p, c = DestPlanner(), carton()
    placed = 0
    while True:
        spots = p.candidates(c)
        if not spots:
            break
        p.commit(spots[0]); placed += 1
        assert placed < 20
    assert placed >= 4                      # 0.4x0.3 cartons on a 0.6x0.6 pallet, stacked
    assert p.max_height() + 0 <= config.MAX_STACK


def test_dest_planner_prefers_ground_level():
    p = DestPlanner()
    first = p.candidates(carton())[0]
    assert first.support_h == 0.0


def test_straight_path_ok_and_flip_rejected():
    ok = [[0, 0.5, 0.5, 0, -1.0, 0], [0, 0.51, 0.5, 0, -1.0, 0], [0, 0.52, 0.5, 0, -1.0, 0]]
    assert violations(ok, PathLimits()) == []
    flip = [[0, 0.5, 0.5, 0, -1.0, 0], [0, 0.5, 0.5, 3.0, -1.0, 0]]
    assert any('jump' in r for r in violations(flip))


def test_singularity_rejected():
    q = [[0, 0.5, 0.5, 0, -1.0, 0], [0, 0.5, 0.5, 0, 0.05, 0]]
    assert any('singularity' in r for r in violations(q, PathLimits(max_step=1.0)))


def test_posture_windows():
    good = [10, 40, 20, -100, -60, 100]
    assert PICK.accepts(good) and not PICK.accepts([10, 40, 20, -10, -60, 100])
    assert TRANSFER.accepts([0, 30, 40, -90, 50, 0]) and not TRANSFER.accepts([0, 30, 40, -90, 10, 0])
