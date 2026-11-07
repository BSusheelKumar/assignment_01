"""Pose helpers. The tool points straight down; orientation is a yaw about world Z."""
import math

from geometry_msgs.msg import Pose

from . import config


def tool_down_pose(x, y, z, yaw):
    """Tool Z pointing down. A 180 deg flip about the horizontal axis at angle yaw/2
    is the quaternion (cos(yaw/2), sin(yaw/2), 0, 0)."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.x = math.cos(yaw / 2.0)
    p.orientation.y = math.sin(yaw / 2.0)
    p.orientation.z = 0.0
    p.orientation.w = 0.0
    return p


def pose_yaw(pose):
    """Inverse of tool_down_pose's yaw."""
    return 2.0 * math.atan2(pose.orientation.y, pose.orientation.x)


def pose_yawed(pose, yaw):
    return tool_down_pose(pose.position.x, pose.position.y, pose.position.z, yaw)


def pose_with(pose, x=None, y=None, z=None):
    """Copy of pose with some position components replaced."""
    p = Pose()
    p.position.x = pose.position.x if x is None else x
    p.position.y = pose.position.y if y is None else y
    p.position.z = pose.position.z if z is None else z
    p.orientation = pose.orientation
    return p


def box_pose(x, y, z, yaw):
    """Pose of a collision box: yaw about world Z."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = x, y, z
    p.orientation.z = math.sin(yaw / 2.0)
    p.orientation.w = math.cos(yaw / 2.0)
    return p


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def arc_waypoints(start, x_end, y_end, z):
    """Arc about the robot base at height z, from start to (x_end, y_end).

    The tool turns with the base so J4-J6 hardly move; the carton yaw is fixed
    afterwards by a separate alignment step. Returns (waypoints, final_tool_yaw).
    """
    bx, by = config.BASE_XY
    r0 = math.hypot(start.position.x - bx, start.position.y - by)
    t0 = math.atan2(start.position.y - by, start.position.x - bx)
    r1 = math.hypot(x_end - bx, y_end - by)
    t1 = math.atan2(y_end - by, x_end - bx)
    dt = wrap_pi(t1 - t0)
    n = max(2, int(math.ceil(abs(math.degrees(dt)) / config.ARC_STEP_DEG)))
    yaw0 = pose_yaw(start)
    wps = []
    for k in range(1, n + 1):
        f = k / n
        r = r0 + (r1 - r0) * f
        t = t0 + dt * f
        wps.append(tool_down_pose(bx + r * math.cos(t), by + r * math.sin(t), z, yaw0 + dt * f))
    return wps, yaw0 + dt
