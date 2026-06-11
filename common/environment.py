"""
common/environment.py
=====================

Shared grid world and scenario library for the mosquito-control
robot sandbox.

A :class:`GridWorld` is a discrete occupancy grid (``0`` free, ``1`` obstacle)
with three optional annotations used by the two planners:

  * ``start``     — the robot's docking / start cell.
  * ``goal``      — the *prey*: the mosquito hotspot the point-to-point planner
                    is dispatched to (``ppstar``).
  * ``predator``  — the centre of a *predator-dominance* zone (a human crowd)
                    the point-to-point planner must route around.
  * ``humans``    — discrete human-activity cells used by the coverage planner
                    (``hfaccpp``) to decide which regions to service first.

Pure standard library — no numpy, no rendering dependencies — so it imports
cleanly in a headless test or inside your own nav stack.

The named maze scenarios are ported verbatim from the original research
benchmark (``mazefinder`` / ``setting``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A cell is a (row, col) integer coordinate.
Cell = Tuple[int, int]

# 8-connected neighbour offsets (row, col) used across the package.
_OFFSETS_8: List[Tuple[int, int]] = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


@dataclass
class GridWorld:
    """A discrete occupancy grid with optional planner annotations."""

    grid: List[List[int]]                 # 0 = free, 1 = obstacle
    start: Optional[Cell] = None
    goal: Optional[Cell] = None           # prey (mosquito hotspot)
    predator: Optional[Cell] = None       # predator-dominance centre (crowd)
    humans: List[Cell] = field(default_factory=list)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    # ----- queries -----------------------------------------------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def passable(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r][c] == 0

    def neighbors(self, cell: Cell, diagonal: bool = True) -> List[Cell]:
        """Passable 8- (or 4-) connected neighbours, no obstacle corner-cutting."""
        r, c = cell
        out: List[Cell] = []
        offsets = _OFFSETS_8 if diagonal else _OFFSETS_8[:4]
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if not self.passable(nr, nc):
                continue
            if dr != 0 and dc != 0:
                if not self.passable(r + dr, c) or not self.passable(r, c + dc):
                    continue
            out.append((nr, nc))
        return out

    def free_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == 0]

    def n_free(self) -> int:
        return sum(row.count(0) for row in self.grid)


# ----------------------------------------------------------------------
#  Point-to-point (PP*) maze scenarios — ported from the research benchmark
# ----------------------------------------------------------------------
# Each scenario is (grid, start, goal, predator). ``start`` is the robot,
# ``goal`` the prey (mosquito hotspot), ``predator`` the crowd-zone centre.
# A predator "setting" of [a, b, radius] yields dominance strength (a - b)
# and avoidance radius ``radius`` (see ppstar.open_ppstar).

_PP_MAZES: Dict[int, Tuple[List[List[int]], Cell, Cell, Cell]] = {
    # 1 — rooms with pillars (12x12)
    1: ([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ], (0, 1), (11, 6), (4, 2)),
    # 2 — open field (12x12)
    2: ([[0] * 12 for _ in range(12)], (0, 1), (11, 6), (4, 2)),
    # 3 — dense maze (12x12)
    3: ([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    ], (0, 1), (11, 6), (4, 2)),
}

# Predator "setting" presets keyed by grid size (rows). [a, b, radius].
_PP_SETTINGS: Dict[int, List[int]] = {
    12: [10, 4, 3],
    24: [20, 8, 6],
    36: [30, 12, 9],
}


def pp_scenario(n: int = 1) -> GridWorld:
    """Return a point-to-point (PP*) scenario as a :class:`GridWorld`."""
    if n not in _PP_MAZES:
        raise ValueError(f"unknown pp scenario {n}; choose from {sorted(_PP_MAZES)}")
    grid, start, goal, predator = _PP_MAZES[n]
    grid = [row[:] for row in grid]  # defensive copy
    return GridWorld(grid=grid, start=start, goal=goal,
                     predator=predator, humans=[predator])


def predator_setting(rows: int) -> Tuple[float, float]:
    """Return ``(strength, radius)`` for a grid with ``rows`` rows.

    Strength is ``a - b`` and radius is the third element of the preset; both
    follow the original benchmark's ``setting()`` table, scaling with size.
    """
    preset = _PP_SETTINGS.get(rows, [10, 4, 3])
    return float(preset[0] - preset[1]), float(preset[2])


# ----------------------------------------------------------------------
#  Coverage (HFA-CCPP) scenario
# ----------------------------------------------------------------------
def coverage_scenario(rows: int = 12, cols: int = 12, seed: int = 0,
                      n_humans: int = 3) -> GridWorld:
    """A walled room with interior obstacles and a few human-activity cells.

    The robot starts at the first free cell; human cells bias the coverage
    order (human-dense regions serviced first).
    """
    grid = [[0] * cols for _ in range(rows)]
    # outer walls
    for c in range(cols):
        grid[0][c] = grid[rows - 1][c] = 1
    for r in range(rows):
        grid[r][0] = grid[r][cols - 1] = 1
    # a couple of interior obstacles
    for c in range(2, cols // 2):
        grid[rows // 2][c] = 1
    for r in range(rows // 4, rows // 4 + 2):
        grid[r][3 * cols // 4] = 1

    world = GridWorld(grid=grid)
    # start: first free cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                world.start = (r, c)
                break
        if world.start:
            break

    rng = random.Random(seed)
    free = [cell for cell in world.free_cells() if cell != world.start]
    humans: List[Cell] = []
    # cluster humans toward the lower-right to make "human-first" visible
    free.sort(key=lambda cell: (cell[0] + cell[1]), reverse=True)
    pool = free[: max(n_humans * 4, n_humans)]
    rng.shuffle(pool)
    for cell in pool:
        if all(abs(cell[0] - h[0]) + abs(cell[1] - h[1]) >= 2 for h in humans):
            humans.append(cell)
        if len(humans) >= n_humans:
            break
    world.humans = humans
    return world


def start_cell(world: GridWorld) -> Cell:
    """The world's start cell, or the first free cell if unset."""
    if world.start is not None:
        return world.start
    for r in range(world.rows):
        for c in range(world.cols):
            if world.grid[r][c] == 0:
                return (r, c)
    raise ValueError("no free cell in world")
