"""Height-map planner for the destination pallet. No ROS imports."""
import math

import numpy as np

from . import config
from .models import Spot


class DestPlanner:
    """Tracks stack heights on the destination pallet (metres above the pallet top)
    and proposes supported, reachable spots for the next carton."""

    CELL = 0.01        # grid resolution
    MARGIN = 0.02      # grid extends this far past the pallet edge
    STEP = 2           # try every 2nd cell as a footprint origin
    RING = 0.02        # keep-out ring: nothing taller than the support within this distance
    SUPPORT_FILL = 0.8 # at least this fraction of the footprint must sit on the support height
    FLAT_TOL = 0.01    # heights within 1 cm count as the same level

    def __init__(self):
        c = self.CELL
        self.nx = int(round((config.PALLET_LENGTH + 2 * self.MARGIN) / c))
        self.ny = int(round((config.PALLET_WIDTH + 2 * self.MARGIN) / c))
        self.x0 = config.DEST_CENTER[0] - config.PALLET_LENGTH / 2 - self.MARGIN
        self.y0 = config.DEST_CENTER[1] - config.PALLET_WIDTH / 2 - self.MARGIN
        self.h = np.zeros((self.nx, self.ny))

    def max_height(self):
        return float(self.h.max())

    def candidates(self, carton, limit=12):
        """Spots ranked: lowest resulting stack, then closest to the base, then unrotated."""
        c = self.CELL
        L, W, H = carton.snapped_length, carton.snapped_width, carton.height
        ring = int(round(self.RING / c))
        orients = [(0.0, L, W)]
        if abs(L - W) > 0.01:
            orients.append((math.pi / 2.0, W, L))

        found = []
        for yaw, ex, ey in orients:
            nx, ny = int(round(ex / c)), int(round(ey / c))
            for ix in range(0, self.nx - nx + 1, self.STEP):
                for iy in range(0, self.ny - ny + 1, self.STEP):
                    foot = self.h[ix:ix + nx, iy:iy + ny]
                    support = float(foot.max())
                    if (foot >= support - self.FLAT_TOL).mean() < self.SUPPORT_FILL:
                        continue                      # not enough support underneath
                    if support + H > config.MAX_STACK:
                        continue
                    zone = self.h[max(ix - ring, 0):ix + nx + ring,
                                  max(iy - ring, 0):iy + ny + ring]
                    if zone.max() > support + self.FLAT_TOL:
                        continue                      # a taller neighbour is too close
                    cx = self.x0 + (ix + nx / 2.0) * c
                    cy = self.y0 + (iy + ny / 2.0) * c
                    dist = math.hypot(cx - config.BASE_XY[0], cy - config.BASE_XY[1])
                    if dist > config.MAX_REACH:
                        continue
                    key = (round(support + H, 2), round(dist, 2), 0 if yaw == 0.0 else 1, ix, iy)
                    found.append((key, Spot(cx, cy, yaw, support, ix, iy, nx, ny, support + H)))
        found.sort(key=lambda t: t[0])

        kept = []
        for _, s in found:   # drop near-duplicates
            if any(k.yaw == s.yaw and abs(k.support_h - s.support_h) < 0.01
                   and abs(k.x - s.x) < 0.06 and abs(k.y - s.y) < 0.06 for k in kept):
                continue
            kept.append(s)
            if len(kept) >= limit:
                break
        return kept

    def commit(self, spot):
        self.h[spot.ix:spot.ix + spot.nx, spot.iy:spot.iy + spot.ny] = spot.top
