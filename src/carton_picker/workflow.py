"""The pick-and-place sequence, one carton at a time. Reads top to bottom as the task:
plan everything (dry run) -> pre-grasp -> descend -> vacuum -> lift -> transfer ->
align -> place -> release -> retreat."""
import math
import time

from . import config
from .dest_planner import DestPlanner
from .geometry import arc_waypoints, pose_with, pose_yawed, tool_down_pose
from .models import Outcome
from .motion import Motion
from .planning import Planner
from .postures import IKSolver


class Pauses:
    """Optional pause between stages. --step waits for Enter; otherwise a short delay."""

    def __init__(self, log, interactive=False):
        self.log, self.interactive = log, interactive

    def __call__(self, text):
        if self.interactive:
            input(f'Enter: {text}... ')
        else:
            self.log.info(text)
            time.sleep(config.AUTO_PAUSE_S)


class Depalletizer:
    def __init__(self, client, pauses):
        self.client = client
        self.log = client.get_logger()
        self.pause = pauses
        self.solver = IKSolver(client)
        self.planner = Planner(client, self.solver)
        self.motion = Motion(client, self.solver)
        self.dest = DestPlanner()
        self.blocked = []        # (x, y, z_top) of cartons whose pick dry run failed
        self.rise_pose = None    # last retreat pose; the arm rises from here before going home

    # ---- carton selection ---------------------------------------------------
    def _is_blocked(self, c):
        return any(abs(c.x - x) < 0.04 and abs(c.y - y) < 0.04 and abs(c.z_top - z) < 0.05
                   for x, y, z in self.blocked)

    def next_carton(self, timeout=8.0):
        """Nearest pickable carton in the top layer. Returns (index, Carton, pose) or Nones."""
        boxes = self.client.wait_for_boxes(timeout)
        if boxes is None:
            return None, None, None
        cands = [(i, b) for i, b in enumerate(boxes) if b.pickable and not self._is_blocked(b)]
        if not cands:
            return None, None, None
        top = max(b.z_top for _, b in cands)
        layer = [(i, b) for i, b in cands if top - b.z_top < 0.02]
        idx, box = min(layer, key=lambda t: math.hypot(t[1].x, t[1].y))
        return idx, box, tool_down_pose(box.x, box.y, box.z_top, box.yaw)

    # ---- one carton ---------------------------------------------------------
    def pick_and_place_one(self, idx, c, target, slot_index):
        log, motion, client = self.log, self.motion, self.client
        top_z = target.position.z
        src_top = max((b.z_top for b in client.boxes), default=top_z)

        # Carry height: clears the lift, every source carton, and the destination stack.
        z_over = max(top_z + config.LIFT_HEIGHT,
                     src_top + c.height + config.CARRY_MARGIN,
                     config.PALLET_HEIGHT + self.dest.max_height() + c.height + config.CARRY_MARGIN)

        spots = self.dest.candidates(c)
        log.info(f'{len(spots)} candidate spots; testing the full pick->transfer->place chain')

        pick, why = self.planner.find_pick(target, top_z, z_over)
        if pick is None:
            log.warn(f'pick not feasible at any tool offset ({why}); skipping this carton')
            self.blocked.append((c.x, c.y, c.z_top))
            return Outcome.SKIPPED

        spot = None
        for s in spots:
            z_place = config.PALLET_HEIGHT + s.support_h + c.height + config.PLACE_CLEARANCE
            result, note = self.planner.transfer_feasible(pick, s, z_over, z_place)
            log.info(f'spot x={s.x:.3f} y={s.y:.3f} h={s.support_h:.2f} '
                     f'yaw={s.yaw:.2f}: {"OK" if result else "rejected"} ({note})')
            if result:
                s.transfer, s.tool_yaw = result
                spot = s
                break
        if spot is None:
            log.error('pallet full: no supported spot left' if not spots else
                      f'{len(spots)} free spots exist, but none is reachable by the arm')
            return Outcome.FAILED

        log.info(f'carton_{idx} ({c.length:.2f}x{c.width:.2f}x{c.height:.2f}) '
                 f'-> slot {slot_index + 1} at x={spot.x:.3f} y={spot.y:.3f} h={spot.support_h:.2f}')

        # Key poses.
        pre_grasp = pose_with(pick.target, z=top_z + config.APPROACH_HEIGHT)
        grasp = pose_with(pick.target, z=top_z + config.GRASP_OFFSET)
        lift = pose_with(pick.target, z=top_z + config.LIFT_HEIGHT)
        carry_end = pose_with(pick.over, x=spot.x, y=spot.y)
        overhead = pose_yawed(carry_end, spot.tool_yaw)
        place = pose_with(overhead, z=config.PALLET_HEIGHT + spot.support_h + c.height
                          + config.PLACE_CLEARANCE)
        retreat = pose_with(place, z=place.position.z + config.RETREAT_HEIGHT)

        client.spin_for(0.5)   # let the scene manager's latest update arrive

        self.pause('move to pre-grasp')
        if not motion.move_joints(pick.q_pre):
            log.error('pre-grasp failed: no collision-safe approach')
            return Outcome.FAILED

        self.pause('remove target from scene and descend to grasp')
        client.remove_cartons_overlapping(c, idx)
        if not motion.cartesian([pre_grasp, grasp], 'descent to grasp'):
            log.error('descent failed; the carton is still in the source region')
            return Outcome.FAILED

        self.pause('vacuum ON, wait for physical attachment')
        client.set_vacuum(True)
        if not client.wait_for_attachment(True):
            client.set_vacuum(False)
            client.wait_for_attachment(False)
            client.restore_carton(idx, c)
            log.error('vacuum did not attach; stopped before lift. Check TCP contact and /vacuum/attached')
            return Outcome.FAILED
        client.attach_carton(c)
        client.spin_for(0.3)
        client.remove_cartons_overlapping(c, idx)
        if not motion.cartesian([grasp, lift], 'lift with carton'):
            log.error('lift failed with the carton attached; not continuing')
            return Outcome.FAILED

        if z_over > lift.position.z + 0.005:
            self.pause('rise vertically to carry height')
            if not motion.cartesian([lift, pick.over], 'rise to carry height'):
                log.error('rise failed; carton remains held')
                return Outcome.FAILED

        self.pause('move across to the destination pallet')
        if spot.transfer == 'arc':
            wps, _ = arc_waypoints(pick.over, spot.x, spot.y, z_over)
        else:
            wps = [carry_end]
        if not motion.cartesian(wps, f'transfer ({spot.transfer})'):
            log.warn('transfer rejected; trying a joint-space transfer in a safe wrist posture')
            q = self.solver.transfer_posture(carry_end)
            if q is None or not motion.move_joints(q):
                log.error('transfer failed; carton remains held')
                return Outcome.FAILED

        self.pause('align over the slot')
        if not motion.cartesian([overhead], 'slot alignment'):
            log.error('alignment failed; carton remains held')
            return Outcome.FAILED

        self.pause('descend to the destination pallet')
        if not motion.cartesian([overhead, place], 'place descent',
                                max_j23=config.PLACE_DESCENT_J23_TRAVEL):
            log.error('place descent failed; carton remains held')
            return Outcome.FAILED

        self.pause('vacuum OFF and release')
        client.set_vacuum(False)
        if not client.wait_for_attachment(False):
            log.error('vacuum release not confirmed; carton remains attached')
            return Outcome.FAILED
        client.detach_carton()

        # From here the carton IS on the pallet; record it even if the retreat fails.
        retreat_ok = motion.cartesian([retreat], 'retreat after place')
        if not retreat_ok:
            log.warn('cartesian retreat failed; trying joint-space retreat')
            retreat_ok = motion.move_to_pose(retreat)
        self.dest.commit(spot)
        client.add_placed_carton(c, slot_index, spot)
        if not retreat_ok:
            log.error('carton placed but the arm could not retreat; stopping for manual recovery')
            return Outcome.PLACED_NO_RETREAT
        self.rise_pose = retreat
        log.info(f'carton placed in slot {slot_index + 1}')
        return Outcome.PLACED

    # ---- main loop ----------------------------------------------------------
    def run(self, start_slot=0):
        log, client = self.log, self.client
        client.add_environment()
        placed = start_slot
        log.info(f'starting at destination slot {placed + 1}')

        while placed < config.MAX_DEST_CARTONS:
            idx, carton, target = self.next_carton()
            if carton is None:
                if client.boxes_fresh:
                    msg = f'no more pickable cartons; placed {placed}'
                    if self.blocked:
                        msg += f'; skipped (x, y, z_top): {self.blocked}'
                    log.info(msg)
                else:
                    log.error('no fresh carton detection; stopping before another pick')
                break

            outcome = self.pick_and_place_one(idx, carton, target, placed)
            if outcome is Outcome.SKIPPED:
                log.info('carton skipped, looking for another candidate')
                client.spin_for(0.5)
                continue
            if outcome is Outcome.PLACED_NO_RETREAT:
                placed += 1
                log.error(f'{placed} cartons placed; stopping because the retreat failed')
                break
            if outcome is Outcome.FAILED:
                log.error(f'stopping after {placed} completed placements')
                break

            placed += 1
            if placed >= config.MAX_DEST_CARTONS:
                break
            client.wait_fresh_scan(timeout=6.0)
            if self.next_carton(timeout=0.5)[1] is None:
                log.info('no carton visible; going home to clear the camera view')
                if not self.motion.go_home(self.rise_pose):
                    log.error('home move failed; stopping')
                    break
                client.wait_fresh_scan()

        if placed == config.MAX_DEST_CARTONS:
            log.info(f'destination capacity reached: {placed} cartons')
