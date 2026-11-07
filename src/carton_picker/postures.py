"""IK with posture control.

A 6-axis arm usually has several IK solutions for one pose (elbow up/down, wrist
flipped or not). MoveIt may return any of them, and some make the next Cartesian
move impossible. So we (1) seed the solver near a good posture and (2) reject
answers outside a posture window. The numbers are specific to the CRX at this
cell layout: they keep the arm in front of the base, elbow up, wrist away from J5=0.
"""
import math
from dataclasses import dataclass, field

from .config import JOINTS


@dataclass(frozen=True)
class PostureWindow:
    """Allowed joint ranges in degrees, by joint index (0 = J1). Missing = unrestricted."""
    ranges: dict = field(default_factory=dict)
    min_abs_j5: float = 0.0

    def accepts(self, deg):
        if abs(deg[4]) < self.min_abs_j5:
            return False
        return all(lo <= deg[i] <= hi for i, (lo, hi) in self.ranges.items())


# Front branch, elbow up: used for picking and as the default for everything.
PICK = PostureWindow({0: (-100, 100), 3: (-175, -50), 4: (-130, -20), 5: (-60, 215)})
# Same, but keep J5 further from the singularity (alternative wrist postures).
ALTERNATE = PostureWindow({0: (-100, 100), 3: (-175, -50), 4: (-130, -30), 5: (-60, 215)})
# Safe posture for a joint-space transfer across the cell.
TRANSFER = PostureWindow({2: (-20, 110), 3: (-140, -50)}, min_abs_j5=40.0)


def _rad(values):
    return [math.radians(v) for v in values]


class IKSolver:
    """The three IK strategies the planner needs, sharing one service call."""

    def __init__(self, client):
        self.client = client
        self.log = client.get_logger()

    def _solve(self, pose, seed_deg, timeout_s, window):
        q = self.client.compute_ik(pose, _rad(seed_deg), timeout_s)
        if q is None:
            return None
        deg = [math.degrees(x) for x in q]
        return q if window.accepts(deg) else None

    def pick_posture(self, pose):
        """Best front-branch solution for a pick pose, or None."""
        j1 = math.degrees(math.atan2(pose.position.y, pose.position.x))
        for j6 in (110, 150, 70, 30, 190):
            q = self._solve(pose, [j1, 38, 21, -106, -45, j6], 0.3, PICK)
            if q is not None:
                self.log.info(f'IK J6-seed={j6}: {[round(math.degrees(x)) for x in q]}')
                return q
        return None

    def alternatives(self, pose):
        """Other front-branch solutions with different wrist postures, best J5 first."""
        j1 = math.degrees(math.atan2(pose.position.y, pose.position.x))
        sols = []
        for j5 in (-65, -90, -40):
            for j4 in (-95, -125, -155):
                for j6 in (130, 80, 180, 30, -30):
                    q = self._solve(pose, [j1, 45, 30, j4, j5, j6], 0.08, ALTERNATE)
                    if q is None:
                        continue
                    deg = [math.degrees(x) for x in q]
                    if any(max(abs(a - b) for a, b in zip(deg, d)) < 4.0 for d, _ in sols):
                        continue
                    sols.append((deg, q))
        sols.sort(key=lambda t: -abs(t[0][4]))
        return [q for _, q in sols]

    def transfer_posture(self, pose):
        """Solution in the safe transfer posture. J5 is seeded with the base angle on
        purpose: it keeps the wrist roughly aligned with the arm plane."""
        base = math.degrees(math.atan2(pose.position.y, pose.position.x))
        for dj in (-8.0, 0.0, 8.0):
            for j6 in (81.0, 120.0, 40.0):
                j1 = base + dj
                q = self._solve(pose, [j1, 37, 43, -95, j1, j6], 0.5, TRANSFER)
                if q is not None:
                    self.log.info(f'transfer IK: {[round(math.degrees(x)) for x in q]}')
                    return q
        return None
