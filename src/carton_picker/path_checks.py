"""Sanity checks on a joint-space path. Pure functions, no ROS imports.

A Cartesian path can be collision-free and still be a bad idea: IK may flip branch
mid-way, fold the elbow, or pass through the J5 wrist singularity. These checks
reject such paths before anything moves.
"""
import math
from dataclasses import dataclass, replace
from typing import Optional

from . import config


@dataclass(frozen=True)
class PathLimits:
    max_step: float = config.MAX_JOINT_STEP
    max_j1: float = config.MAX_J1_SWING
    max_j23: float = config.MAX_J23_TRAVEL
    max_wrist: float = config.MAX_WRIST_TRAVEL
    min_j5: float = config.MIN_J5
    max_winding: Optional[float] = config.MAX_WINDING   # None = skip the winding test

    def with_(self, **kw):
        return replace(self, **kw)


def trajectory_matrix(joint_trajectory):
    """Joint positions as rows ordered J1..J6, whatever order MoveIt used."""
    names = list(joint_trajectory.joint_names)
    order = [names.index(n) for n in config.JOINTS]
    return [[p.positions[i] for i in order] for p in joint_trajectory.points]


def travel_per_joint(q):
    return [sum(abs(b[j] - a[j]) for a, b in zip(q, q[1:])) for j in range(len(q[0]))]


def largest_step(q):
    """(step, point_index, joint_index) of the biggest single-sample joint jump."""
    return max((abs(b[j] - a[j]), i, j)
               for i, (a, b) in enumerate(zip(q, q[1:])) for j in range(len(a)))


def winding(q):
    """Joint-path length over net displacement; ~1 for a direct move, large for a detour."""
    n = len(q[0])
    length = sum(math.sqrt(sum((b[j] - a[j]) ** 2 for j in range(n))) for a, b in zip(q, q[1:]))
    net = math.sqrt(sum((q[-1][j] - q[0][j]) ** 2 for j in range(n)))
    return length / max(net, 1e-3), net


def violations(q, limits=PathLimits()):
    """Human-readable reasons this joint path is unsafe (empty list = fine)."""
    if len(q) < 2:
        return []
    out = []
    ratio, net = winding(q)
    if limits.max_winding is not None and net >= 0.5 and ratio > limits.max_winding:
        out.append(f'winding {ratio:.2f} > {limits.max_winding}')
    step, at, joint = largest_step(q)
    if step > limits.max_step:
        out.append(f'joint jump {step:.2f} rad in J{joint + 1} at point {at}/{len(q) - 1}, '
                   f'J5={math.degrees(q[at][4]):.0f} deg (IK branch flip?)')
    travel = travel_per_joint(q)
    if max(travel[1:3]) > limits.max_j23:
        out.append(f'J2/J3 travel {max(travel[1:3]):.2f} > {limits.max_j23} (elbow flip / fold)')
    if travel[0] > limits.max_j1:
        out.append(f'J1 travel {travel[0]:.2f} > {limits.max_j1}')
    if max(travel[3:6]) > limits.max_wrist:
        out.append(f'wrist travel {max(travel[3:6]):.2f} > {limits.max_wrist}')
    if min(abs(p[4]) for p in q) < limits.min_j5:
        out.append(f'J5 within {math.degrees(limits.min_j5):.0f} deg of the wrist singularity')
    return out
