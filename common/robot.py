"""
common/robot.py
===============

A lightweight differential-drive robot for the sandbox.

The robot carries a continuous pose ``(x, y, theta)`` in pixels/radians and is
driven either by teleoperation (WASD) or by following a planner path
(pure-pursuit over grid waypoints). Collision against the rasterised occupancy
is checked through a caller-supplied ``passable(px, py)`` predicate so this
module stays independent of the feature/grid representation.

Controls convention (handled by the demo, realised here):
  * forward / back  -> differential linear velocity
  * left / right    -> differential angular velocity (rotate in place allowed)
  * speed scale     -> Q increases, E decreases (a multiplier on both)

Pure standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

Point = Tuple[float, float]

# Per-robot colours (id 1..5).
ROBOT_COLOURS = [
    (90, 160, 245),   # 1 blue
    (90, 220, 140),   # 2 green
    (235, 110, 110),  # 3 red
    (80, 200, 210),   # 4 cyan
    (205, 130, 230),  # 5 purple
]

BASE_LIN = 95.0   # px/s at scale 1.0
BASE_ANG = 2.4    # rad/s at scale 1.0
ARRIVE_TOL = 7.0  # px waypoint arrival tolerance

# Robot body outline in local units (forward = +y). The polygon inscribes a
# circular footprint of radius FOOTPRINT_UNITS; the body is drawn scaled so
# that footprint maps to ``radius`` px, then rotated to the heading.
# (User-specified shape + footprint radius.)
SHAPE: List[Point] = [
    (2.0, -2.0), (2.0, 1.0), (1.0, 2.0),
    (-1.0, 2.0), (-2.0, 1.0), (-2.0, -2.0),
]
FOOTPRINT_UNITS: float = 2.0   # collision-circle radius in local SHAPE units


def _wrap(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Robot:
    """A differential-drive robot with optional planner-path following."""

    rid: int
    x: float
    y: float
    theta: float = 0.0
    radius: float = 11.0          # circular footprint the body polygon inscribes
    waypoints: List[Point] = field(default_factory=list)
    wp_i: int = 0

    @property
    def color(self) -> Tuple[int, int, int]:
        return ROBOT_COLOURS[(self.rid - 1) % len(ROBOT_COLOURS)]

    def polygon_world(self) -> List[Point]:
        """Body outline in world coordinates, oriented to the heading.

        SHAPE's forward axis is +y, so we align +y with ``theta`` (rotate by
        ``theta - pi/2``). Local units are scaled by ``radius / FOOTPRINT_UNITS``
        so the SHAPE's footprint circle (radius FOOTPRINT_UNITS) maps to the
        robot's ``radius`` px collision footprint.
        """
        a = self.theta - math.pi / 2.0
        ca, sa = math.cos(a), math.sin(a)
        s = self.radius / FOOTPRINT_UNITS
        pts: List[Point] = []
        for lx, ly in SHAPE:
            pts.append((self.x + s * (ca * lx - sa * ly),
                        self.y + s * (sa * lx + ca * ly)))
        return pts

    # ----- navigation goals -------------------------------------------
    def set_path(self, waypoints: List[Point]) -> None:
        self.waypoints = list(waypoints)
        self.wp_i = 0

    def clear_path(self) -> None:
        self.waypoints = []
        self.wp_i = 0

    @property
    def has_path(self) -> bool:
        return bool(self.waypoints) and self.wp_i < len(self.waypoints)

    # ----- low-level drive --------------------------------------------
    def drive(self, fwd: float, turn: float, scale: float, dt: float,
              passable: Callable[[float, float], bool]) -> None:
        """Apply a differential-drive command for ``dt`` seconds.

        ``fwd`` and ``turn`` are in {-1, 0, +1}. Rotation always applies;
        translation only commits if the destination is passable (so the robot
        slides to a stop against obstacles instead of clipping through them).
        """
        self.theta = _wrap(self.theta + turn * BASE_ANG * scale * dt)
        if fwd != 0.0:
            v = fwd * BASE_LIN * scale
            nx = self.x + v * math.cos(self.theta) * dt
            ny = self.y + v * math.sin(self.theta) * dt
            if passable(nx, ny):
                self.x, self.y = nx, ny

    def teleop(self, forward: int, rotate: int, scale: float, dt: float,
               passable: Callable[[float, float], bool]) -> None:
        """WASD teleoperation; cancels any active planner path."""
        if forward or rotate:
            self.clear_path()
        self.drive(float(forward), float(rotate), scale, dt, passable)

    # ----- autopilot (follow a planner path) --------------------------
    def follow(self, scale: float, dt: float,
               passable: Callable[[float, float], bool],
               lookahead: float = 28.0) -> None:
        """Lookahead pure-pursuit along the path (smooth differential drive).

        Instead of aiming at the immediate next waypoint (which makes the robot
        stutter on every grid corner), it aims at a point ``lookahead`` px down
        the path and arcs toward it — turning and advancing together.
        """
        if not self.has_path:
            return
        # consume waypoints we've already reached
        while (self.wp_i < len(self.waypoints) - 1
               and math.hypot(self.waypoints[self.wp_i][0] - self.x,
                              self.waypoints[self.wp_i][1] - self.y) <= ARRIVE_TOL):
            self.wp_i += 1
        # pick the farthest waypoint within the lookahead radius
        j = self.wp_i
        while (j < len(self.waypoints) - 1
               and math.hypot(self.waypoints[j + 1][0] - self.x,
                              self.waypoints[j + 1][1] - self.y) <= lookahead):
            j += 1
        tx, ty = self.waypoints[j]
        dtheta = _wrap(math.atan2(ty - self.y, tx - self.x) - self.theta)
        turn = max(-1.0, min(1.0, dtheta / 0.6))
        # advance while turning unless the heading error is very large; speed
        # eases off as misalignment grows for a smooth arc
        fwd = max(0.0, math.cos(dtheta)) if abs(dtheta) < 1.4 else 0.0
        self.theta = _wrap(self.theta + turn * BASE_ANG * scale * dt)
        if fwd > 0.0:
            v = fwd * BASE_LIN * scale
            nx = self.x + v * math.cos(self.theta) * dt
            ny = self.y + v * math.sin(self.theta) * dt
            if passable(nx, ny):
                self.x, self.y = nx, ny
            elif self.wp_i < len(self.waypoints) - 1:
                self.wp_i += 1          # blocked: skip ahead so we keep moving
