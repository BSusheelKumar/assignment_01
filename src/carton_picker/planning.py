"""Dry-run planning: decide whether a whole pick -> transfer -> place chain will work
BEFORE the arm moves. Each stage is simulated from the joint state the previous stage
ends in, exactly as the real motion will run."""
import math

from . import config
from .geometry import arc_waypoints, pose_with, pose_yaw, pose_yawed
from .models import PickPlan
from .path_checks import PathLimits, trajectory_matrix, violations

DRY_LIMITS = PathLimits(max_winding=None)


class Planner:
    def __init__(self, client, solver):
        self.client = client
        self.solver = solver
        self.log = client.get_logger()

    def dry_cartesian(self, q, waypoints, max_j23=config.MAX_J23_TRAVEL):
        """Simulate a Cartesian move from joints q. Returns (q_end, None) or (None, reason)."""
        resp = self.client.compute_cartesian(waypoints, start_q=q)
        if resp is None:
            return None, 'no response'
        if resp.fraction < config.CART_MIN_FRACTION:
            return None, f'fraction {resp.fraction:.2f}'
        rows = trajectory_matrix(resp.solution.joint_trajectory)
        if len(rows) < 2:
            return list(q), None
        bad = violations(rows, DRY_LIMITS.with_(max_j23=max_j23))
        if bad:
            return None, '; '.join(bad)
        return rows[-1], None

    # ---- pick ---------------------------------------------------------------
    def pick_chain(self, pre_grasp, grasp, lift, rise, quick=False):
        """Dry run of: pre-grasp posture -> descent -> lift -> (rise). Tries the usual IK
        posture first, then other wrist postures, then IK at the grasp pose and back up.
        Returns (q_pre, q_top, None) or (None, None, reason)."""
        steps = [('descent', [grasp]), ('lift', [lift])]
        if rise is not None:
            steps.append(('rise', [rise]))

        def run(q0):
            q = q0
            for label, wps in steps:
                q, why = self.dry_cartesian(q, wps)
                if q is None:
                    return None, f'{label}: {why}'
            return q, None

        why = 'no front-branch IK at the pre-grasp pose'
        tried = []
        first = self.solver.pick_posture(pre_grasp)
        if first is not None:
            tried.append(first)
            q, why = run(first)
            if q is not None:
                return first, q, None
        if quick:
            return None, None, why

        for q0 in self.solver.alternatives(pre_grasp):
            if any(max(abs(a - b) for a, b in zip(q0, t)) < 0.05 for t in tried):
                continue
            tried.append(q0)
            q, why = run(q0)
            if q is not None:
                self.log.info(f'pick: using wrist posture {[round(math.degrees(x)) for x in q0]}')
                return q0, q, None

        # Last resort: solve at the GRASP pose, go straight up to the pre-grasp pose, then
        # replay the pick. The descent is then the same continuous path in reverse.
        grasp_sols = [s for s in [self.solver.pick_posture(grasp)] if s is not None]
        grasp_sols += self.solver.alternatives(grasp)
        for q_g in grasp_sols:
            q_up, w = self.dry_cartesian(q_g, [pre_grasp])
            if q_up is None:
                why = f'rise from grasp: {w}'
                continue
            q, why = run(q_up)
            if q is not None:
                self.log.info('pick: using a posture found from the grasp pose')
                return q_up, q, None
        return None, None, f'{why} (tried {len(tried)} wrist postures + {len(grasp_sols)} grasp-first)'

    def find_pick(self, target, top_z, z_over):
        """Try tool orientations until one gives a feasible pick. Returns (PickPlan|None, why)."""
        base_yaw = pose_yaw(target)
        why = ''
        for off in config.TOOL_OFFSETS:
            tp = pose_yawed(target, base_yaw + off)
            over = pose_with(tp, z=z_over)
            rise = over if z_over > top_z + config.LIFT_HEIGHT + 0.005 else None
            q_pre, q_top, why = self.pick_chain(
                pose_with(tp, z=top_z + config.APPROACH_HEIGHT),
                pose_with(tp, z=top_z + config.GRASP_OFFSET),
                pose_with(tp, z=top_z + config.LIFT_HEIGHT),
                rise,
                quick=(off != 0.0))   # full search only for the aligned orientation
            if q_pre is not None:
                if off:
                    self.log.info(f'pick: tool turned {math.degrees(off):.0f} deg from the carton axis')
                return PickPlan(tp, over, off, q_pre, q_top), None
            self.log.info(f'pick with tool offset {math.degrees(off):.0f} deg not feasible ({why})')
        return None, why

    # ---- transfer + place ---------------------------------------------------
    def transfer_feasible(self, pick, spot, z_over, z_place):
        """Dry run: transfer (arc, else straight line) -> align -> descent.
        Returns ((mode, tool_yaw), note) or (None, note)."""
        notes = []
        for mode in ('arc', 'line'):
            if mode == 'arc':
                wps, yaw_end = arc_waypoints(pick.over, spot.x, spot.y, z_over)
            else:
                wps = [pose_with(pick.over, x=spot.x, y=spot.y)]
                yaw_end = pose_yaw(pick.over)
            # Carton yaw in the world must be spot.yaw; the tool yaw is that modulo a half
            # turn (the carton is symmetric), picked closest to where the transfer ends.
            carton_yaw = spot.yaw + pick.offset
            ty0 = carton_yaw + round((yaw_end - carton_yaw) / math.pi) * math.pi
            tool_yaws = [ty0]
            if config.ALLOW_TOOL_FLIP:
                tool_yaws.append(ty0 + math.copysign(math.pi, yaw_end - ty0))
            for ty in tool_yaws:
                over = pose_yawed(pose_with(pick.over, x=spot.x, y=spot.y), ty)
                low = pose_with(over, z=z_place)
                q = pick.q_top
                for label, wp, j23 in (('transfer', wps, config.MAX_J23_TRAVEL),
                                       ('align', [over], config.MAX_J23_TRAVEL),
                                       ('descent', [low], config.PLACE_DESCENT_J23_TRAVEL)):
                    q, why = self.dry_cartesian(q, wp, j23)
                    if q is None:
                        notes.append(f'{mode}@{ty:.2f} {label}: {why}')
                        break
                if q is not None:
                    return (mode, ty), f'{mode}, tool_yaw={ty:.2f}'
        return None, ' | '.join(notes)
