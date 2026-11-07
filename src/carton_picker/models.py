"""Plain data types. No ROS imports, so these are unit-testable anywhere."""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from . import config


def snap_size(v):
    """Snap a measured edge length to the nearest known carton size."""
    for n in config.CARTON_SIZES:
        if abs(v - n) < config.SIZE_SNAP_TOL:
            return n
    return v


@dataclass
class Carton:
    """One detected carton, as published on /perception/boxes_json."""
    x: float
    y: float
    z_top: float
    yaw: float
    length: float
    width: float
    height: float
    pickable: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(x=d['x'], y=d['y'], z_top=d['z_top'], yaw=d['yaw'],
                   length=d['length'], width=d['width'], height=d['height'],
                   pickable=d.get('pickable', False))

    @property
    def snapped_length(self):
        return snap_size(self.length)

    @property
    def snapped_width(self):
        return snap_size(self.width)

    @property
    def z_center(self):
        return self.z_top - self.height / 2.0


@dataclass
class Spot:
    """A candidate placement on the destination pallet."""
    x: float
    y: float
    yaw: float          # carton yaw on the pallet (0 or pi/2)
    support_h: float    # height of whatever it rests on, above the pallet top
    ix: int
    iy: int
    nx: int
    ny: int
    top: float          # height of the carton top above the pallet top
    transfer: Optional[str] = None     # 'arc' or 'line', set once the dry run passes
    tool_yaw: Optional[float] = None   # tool yaw to use at the spot


@dataclass
class PickPlan:
    target: Any         # geometry_msgs/Pose of the carton top, with the chosen tool offset
    over: Any           # same x,y/yaw at carry height
    offset: float       # tool offset from the carton axis [rad]
    q_pre: list         # joint posture at the pre-grasp pose
    q_top: list         # joint posture at the top of the lift (dry-run end state)


class Outcome(Enum):
    PLACED = auto()
    SKIPPED = auto()            # carton not pickable here; try another
    PLACED_NO_RETREAT = auto()  # released fine, arm could not retreat: stop for manual recovery
    FAILED = auto()
