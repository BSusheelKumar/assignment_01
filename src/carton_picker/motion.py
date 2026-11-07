"""Executing motion: joint goals and checked Cartesian segments."""
import math

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from . import config
from .path_checks import PathLimits, trajectory_matrix, violations


class Motion:
    def __init__(self, client, solver):
        self.client = client
        self.solver = solver
        self.log = client.get_logger()

    def move_joints(self, q):
        code = self.client.send_joint_goal(q)
        if code is None:
            self.log.warn('joint goal rejected by MoveIt')
        elif code != 1:
            self.log.warn(f'joint goal failed, MoveIt error code = {code}')
        return code == 1

    def move_to_pose(self, pose):
        """Joint-space move to a pose, using a front-branch IK solution."""
        q = self.solver.pick_posture(pose)
        if q is None:
            self.log.error('no IK solution in the front/elbow-up branch')
            return False
        return self.move_joints(q)

    def cartesian(self, waypoints, label, max_j23=config.MAX_J23_TRAVEL):
        """Straight-line move through waypoints from the current state.
        Planned, checked (fraction, joint jumps, singularity, collisions), then executed."""
        resp = self.client.compute_cartesian(waypoints, scale=config.CARTESIAN_SCALE)
        if resp is None:
            self.log.error(f'{label}: empty cartesian response')
            return False
        self.log.info(f'{label}: fraction={resp.fraction:.2f}, MoveIt error={resp.error_code.val}')
        if resp.fraction < config.CART_MIN_FRACTION:
            self.log.warn(f'{label}: incomplete cartesian path')
            self._log_contacts(label)
            return False
        traj = resp.solution
        if len(traj.joint_trajectory.points) < 2:
            self.log.info(f'{label}: already at target')
            return True

        rows = trajectory_matrix(traj.joint_trajectory)
        self.log.info(f'{label}: q_start={[round(math.degrees(x)) for x in rows[0]]} '
                      f'q_end={[round(math.degrees(x)) for x in rows[-1]]} deg')
        bad = violations(rows, PathLimits().with_(max_j23=max_j23))
        if bad:
            self.log.error(f'{label}: REJECTED: ' + '; '.join(bad))
            return False
        if not self._collision_free(rows, label):
            return False

        return self.client.execute(traj) == 1

    def _collision_free(self, rows, label):
        idx = list(range(0, len(rows), config.VALIDITY_STRIDE)) + [len(rows) - 1]
        for i in idx:
            res = self.client.state_validity(rows[i])
            if res is None or not res.valid:
                pairs = sorted({(c.contact_body_1, c.contact_body_2) for c in res.contacts}) if res else []
                self.log.error(f'{label}: collision at point {i}: {pairs}')
                return False
        return True

    def _log_contacts(self, label):
        res = self.client.state_validity()
        if res is None:
            return
        self.log.error(f'{label}: current state valid={res.valid}')
        for c in res.contacts:
            self.log.error(f'contact {c.contact_body_1} vs {c.contact_body_2} '
                           f'at ({c.position.x:.3f},{c.position.y:.3f},{c.position.z:.3f}) '
                           f'depth={c.depth:.4f}')
        for b in self.client.boxes or []:
            self.log.error(f'box {b}')

    def go_home(self, rise_pose=None):
        """Fold the arm to HOME so the camera sees the pallet. Rises vertically first."""
        self.log.info('going home so the camera can see the pallet')
        if rise_pose is not None and not self.cartesian([rise_pose], 'rise before home'):
            self.log.error('rise before home failed; not moving the arm sideways')
            return False
        if self.move_joints(config.HOME):
            return True

        # Fallback: straight to the controller. Fold up first, then rotate J1.
        self.log.warn('MoveIt home failed; folding arm up, then rotating')
        self.client.spin_for(0.3)
        if 'J1' not in self.client.joints:
            self.log.error('no /joint_states received; refusing blind home move')
            return False
        msg = JointTrajectory()
        msg.joint_names = list(config.JOINTS)
        fold = JointTrajectoryPoint()
        fold.positions = [self.client.joints['J1'], 0.0, 0.0, 0.0, 0.0, 0.0]
        fold.time_from_start = Duration(sec=8)
        home = JointTrajectoryPoint()
        home.positions = list(config.HOME)
        home.time_from_start = Duration(sec=16)
        msg.points = [fold, home]
        self.client.publish_trajectory(msg)
        self.client.spin_for(17.0)
        return True
