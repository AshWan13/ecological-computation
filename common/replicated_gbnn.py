"""
common/replicated_gbnn.py
=========================

Reference implementation of the **Glasius Bio-inspired Neural Network (GBNN)**
on a 2-D grid, ported faithfully from the original research code.

This is *prior art* — the shunting neural dynamics are due to Glasius, Komoda &
Gielen (1995), and the grid-coverage application follows Sun, Zhu, Tian & Luo
(2018) and Yi et al. (2023). It is the unmodified base that the ``hfaccpp``
module extends with its human-activity awareness. It is **not** a contribution
of this repository.

Dynamics (per neuron i):

    x_i = G( sum_j w_ij * [x_j]^+  +  I_i )

with the piecewise-linear transfer function

    G(x) = -1            if x < 0
            1            if x >= 1
            b * x        otherwise            (0 < b < 1)

and the local connection weights

    w_ij = exp(-alpha * d_ij^2)   if d_ij < r   else   0

External input I_i is +E for uncovered free targets, -E for obstacles, and 0
for covered cells. Because uncovered cells are the only positive sources and
obstacles are sinks, the relaxed activity has no interior local maxima — the
coverage walk is deadlock-free.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

Cell = Tuple[int, int]


class GBNN:
    """Discrete Glasius Bio-inspired Neural Network over an occupancy grid."""

    def __init__(
        self,
        rows: int,
        cols: int,
        grid: List[List[int]],
        *,
        E: float = 100.0,    # external input magnitude
        r: float = 2.0,      # connection radius (cells)
        alpha: float = 2.0,  # connection decay
        b: float = 0.7,      # transfer slope in the linear region
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.grid = grid
        self.E = E
        self.r = r
        self.alpha = alpha
        self.b = b
        self.activity: List[List[float]] = [[0.0] * cols for _ in range(rows)]
        # Pre-compute neighbours + weights for every free cell.
        self._nbrs: List[List[Tuple[int, int, float]]] = [
            [] for _ in range(rows * cols)
        ]
        rad = int(math.ceil(r))
        for x in range(rows):
            for c in range(cols):
                if grid[x][c] != 0:
                    continue
                lst = self._nbrs[x * cols + c]
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = x + dr, c + dc
                        if not (0 <= nr < rows and 0 <= nc < cols):
                            continue
                        if grid[nr][nc] != 0:
                            continue
                        w = self.w_ij(dr, dc)
                        if w > 0.0:
                            lst.append((nr, nc, w))

    # ------------------------------------------------------------------
    def G_x(self, x: float) -> float:
        """Piecewise-linear transfer function G(x)."""
        if x < 0.0:
            return -1.0
        if x >= 1.0:
            return 1.0
        return self.b * x

    def w_ij(self, dr: int, dc: int) -> float:
        """Connection weight between cells offset by (dr, dc)."""
        d = math.hypot(dr, dc)
        if d < self.r:
            return math.exp(-self.alpha * d * d)
        return 0.0

    def external_input(
        self,
        covered,
        boost: Optional[Callable[[int, int], float]] = None,
    ) -> List[List[float]]:
        """Build the external input field I.

        +E for uncovered free cells, -E for obstacles, 0 for covered cells.
        ``boost(r, c) -> multiplier`` lets the human-aware module amplify the
        input of uncovered cells; the base network passes ``boost=None``.
        """
        I = [[0.0] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != 0:
                    I[r][c] = -self.E
                elif (r, c) in covered:
                    I[r][c] = 0.0
                else:
                    m = boost(r, c) if boost is not None else 1.0
                    I[r][c] = self.E * m
        return I

    def relax(self, I: List[List[float]], iters: int = 1) -> List[List[float]]:
        """Propagate the network ``iters`` times given external input ``I``."""
        x = self.activity
        for _ in range(iters):
            new = [[0.0] * self.cols for _ in range(self.rows)]
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.grid[r][c] != 0:
                        new[r][c] = -1.0
                        continue
                    lateral = 0.0
                    for nr, nc, w in self._nbrs[r * self.cols + c]:
                        xn = x[nr][nc]
                        if xn > 0.0:
                            lateral += w * xn
                    new[r][c] = self.G_x(lateral + I[r][c])
            x = new
        self.activity = x
        return x

    def activity_at(self, r: int, c: int) -> float:
        return self.activity[r][c]
