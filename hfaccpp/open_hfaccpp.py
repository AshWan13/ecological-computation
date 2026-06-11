#!/usr/bin/env python3
"""
hfaccpp/open_hfaccpp.py — Human-First Approach to Complete Coverage Path
Planning (HFA-CCPP).

    Wan Sang, Veerajagadheswar, Elara & Le, "Human activity-aware coverage
    path planning for robot-based mosquito control", Scientific Reports
    15:31009 (2025). https://doi.org/10.1038/s41598-025-16114-1

Two stacked neural layers over the same occupancy grid drive each robot step:

  * a coverage layer (paper symbol ν), a Glasius Bio-inspired Neural Network
    (GBNN) where every uncovered free cell is an excitatory source and every
    obstacle is an inhibitory sink, so its activity gradient always points to
    the nearest unswept region (deadlock-free complete coverage); and
  * a social layer (paper symbol μ), a GBNN whose interest targets are the
    cells surrounding each person (the mosquito-risk zone); the robot's area of
    effect mitigates this risk as it works an area.

At each step the next waypoint is the neighbour with the highest SUM of the two
layers, wp_{i+1} = max(N_kμ + N_kν)  (paper Eqn 10).

Notation — paper symbol → code name
-----------------------------------
    N_i  neuron value ............. gbnn_coverage.activity / gbnn_social.activity
    α_i  external input (±E, 0) ... GBNN.external_input (Eqn 2 / 5)
    E    excitation constant ...... E
    β    lateral-weight decay ...... beta   (paper Eqn 3: exp(-β(i-j)^2))
    R    connection radius ........ R       (paper Eqn 3: 0 < i-j ≤ R)
    ε    transfer-function slope ... epsilon (paper Eqn 4: f(x)=εx)
    f(x) transfer function ........ GBNN.G_x (Eqn 4 / 6)
    w_ij lateral weight ........... GBNN.w_ij (Eqn 3)
    φ    area-of-effect radius ..... aoe_radius (paper Fig. 3 / Eqn 7-8)
    d    distance to a person ...... d   (paper: Manhattan distance)
    r*_r residual risk ............ _residual_risk (per-cell field, Eqn 7)
    r*_c carrying risk ............ _carrying_risk (per-person, Eqn 8)
    μ    social layer ............. gbnn_social
    ν    coverage layer ........... gbnn_coverage
    N_k  neighbour list ........... world.neighbors(pos)
    wp   next waypoint ............ step() selection (Eqn 10)

Both layers are refreshed every step from the live simulation (the human
count and positions, the feature obstacle map, and the serviced state).
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

try:
    from common.environment import GridWorld, Cell, coverage_scenario, start_cell
    from common.replicated_gbnn import GBNN
except ImportError:  # pragma: no cover - direct-script fallback
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.environment import GridWorld, Cell, coverage_scenario, start_cell
    from common.replicated_gbnn import GBNN


@dataclass
class CoverageResult:
    """Outcome of a coverage run."""

    path: List[Cell]
    covered: int
    free_total: int
    steps: int
    human_weighted_delay: float   # Σ visit_order × human_density; lower = crowded areas first

    @property
    def coverage_fraction(self) -> float:
        return self.covered / self.free_total if self.free_total else 0.0

    @property
    def complete(self) -> bool:
        return self.covered >= self.free_total


class HFACoveragePlanner:
    """HFA-CCPP coverage planner: a GBNN coverage layer (ν) plus a robot-serviced
    social layer (μ), both fed from the live simulation each step. Variable names
    follow the paper's notation — see the module docstring's mapping table."""

    def __init__(
        self,
        world: GridWorld,
        *,
        human_first: bool = True,
        footprint: int = 1,        # cells the robot marks covered per step
        aoe_radius: float = 4.0,   # φ — area-of-effect radius (cells)
        risk_sigma: float = 1.0,   # narrow Gaussian spread (cells) of a person's risk
        trail_decay: float = 0.95,  # per-step survival of vacated risk (lingering trail)
        risk_threshold: float = 0.15,  # a cell counts as human-vicinity above this
        risk_bias: float = 1.0,    # weight of the social layer in the Eqn-10 sum
        turn_penalty: float = 0.30,
        relax_iters: int = 8,      # GBNN relaxation iterations per step
        E: float = 100.0,          # GBNN excitation constant E
        R: float = 2.0,            # GBNN connection radius R  (paper Eqn 3)
        beta: float = 2.0,         # GBNN lateral-weight decay β  (paper Eqn 3)
        epsilon: float = 0.7,      # GBNN transfer-function slope ε  (paper Eqn 4)
    ) -> None:
        self.world = world
        self.rows, self.cols = world.rows, world.cols
        self.human_first = human_first
        self.footprint = footprint
        self.aoe_radius = aoe_radius          # φ
        self.risk_sigma = risk_sigma
        self.trail_decay = trail_decay
        self.risk_threshold = risk_threshold
        self.risk_bias = risk_bias
        self.turn_penalty = turn_penalty
        self.relax_iters = relax_iters

        # GBNN hyper-parameters (paper E, R, β, ε) mapped to the prior-art GBNN
        # class's kwargs (E, r, alpha, b). Kept so both layers can be rebuilt
        # against a live occupancy grid (see update_world) without touching the
        # prior-art GBNN class.
        self._gbnn_kw = dict(E=E, r=R, alpha=beta, b=epsilon)

        # Coverage layer ν — a GBNN over the occupancy grid (feature obstacles
        # are already marked as 1 in world.grid and become inhibitory sinks that
        # settle to -1).
        self.gbnn_coverage = GBNN(self.rows, self.cols, world.grid, **self._gbnn_kw)

        # Social layer μ — a GBNN with the SAME dynamics but whose excitatory
        # sources are only the uncovered cells in a person's vicinity. Its
        # activity gradient therefore points to the nearest un-serviced human
        # area across the whole map, so summing it with ν yields human-first
        # ordering through pure neural propagation (no beeline).
        self.gbnn_social = GBNN(self.rows, self.cols, world.grid, **self._gbnn_kw)

        # Per-cell residual-risk field r*_r and the instantaneous human-density
        # field. r*_r is PERSISTENT: each person stamps a narrow bump at their
        # CURRENT cell, the field decays by ``trail_decay`` so vacated cells keep
        # a fading trail, and the robot's AoE scrubs it. ``_density`` (no trail)
        # marks the human-surround interest targets and feeds the delay metric.
        self._residual_risk = [[0.0] * self.cols for _ in range(self.rows)]
        self._density = [[0.0] * self.cols for _ in range(self.rows)]

        # Live-people state (populated by update_humans()/begin()). Each person
        # carries a carrying risk r*_c keyed by id.
        self._humans: List[Tuple[int, Cell]] = []
        self._carrying_risk: Dict[int, float] = {}

        # Coverage-session state (populated by begin()).
        self._region: Set[Cell] = set()
        self._covered: Set[Cell] = set()
        self._order: Dict[Cell, int] = {}
        self._pos: Optional[Cell] = None
        self._heading_vec: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    #  Social layer (fed by the simulation each step)
    # ------------------------------------------------------------------
    def update_world(self, world: GridWorld) -> None:
        """Adopt a freshly-rasterised occupancy grid (live obstacles + people)
        so both neural layers and the movement graph reflect the CURRENT map.
        Call once per step before ``step()``.

        When the grid actually changed (a person crossing a cell boundary, or an
        obstacle added/removed), each GBNN layer is reconstructed against the new
        grid — its ``__init__`` rebuilds the lateral-connection table — while the
        relaxed ``activity`` is carried over so propagation continues smoothly.
        Newly-occupied nodes settle to -1 on the next relax and freed nodes
        rejoin the field. The prior-art GBNN class is left untouched."""
        if world.grid != self.world.grid:
            self.gbnn_coverage = self._rebuild_gbnn(self.gbnn_coverage, world.grid)
            self.gbnn_social = self._rebuild_gbnn(self.gbnn_social, world.grid)
        self.world = world

    def _rebuild_gbnn(self, old: GBNN, grid) -> GBNN:
        """A fresh GBNN over ``grid`` (rebuilds neighbours) carrying ``old``'s
        activity forward."""
        fresh = GBNN(self.rows, self.cols, grid, **self._gbnn_kw)
        fresh.activity = old.activity
        return fresh

    def update_humans(self, humans) -> None:
        """Adopt the live people from the simulation as ``(id, row, col)`` and
        rebuild the social layer. Each person keeps a carrying risk r*_c keyed
        by id, so the risk follows the person as they move; new people start at
        full risk and departed ones are dropped."""
        parsed: List[Tuple[int, Cell]] = []
        for hid, r, c in humans:
            hid = int(hid)
            parsed.append((hid, (int(r), int(c))))
            self._carrying_risk.setdefault(hid, 1.0)
        self._humans = parsed
        live = {hid for hid, _ in parsed}
        for gone in [h for h in self._carrying_risk if h not in live]:
            del self._carrying_risk[gone]
        self._recompute_risk()

    def _service_people_near(self, pos: Cell) -> None:
        """Carrying risk r*_c (paper Eqn 8): the robot's proximity lowers each
        person's risk — ~0 when the robot is adjacent, fading to none at the AoE
        edge φ — monotonically (it never rises). ``d`` is the distance from the
        robot to the person."""
        phi = self.aoe_radius
        for hid, (hr, hc) in self._humans:
            d = math.hypot(hr - pos[0], hc - pos[1])
            if d < phi:
                r_c = max(d - 1.0, 0.0) / phi          # Eqn 8
                if r_c < self._carrying_risk[hid]:
                    self._carrying_risk[hid] = r_c

    def _scrub_trail(self, pos: Cell) -> None:
        """The robot's area of effect (radius φ == ``aoe_radius``) scrubs the
        persistent residual-risk field around the robot, so serviced ground goes
        quiet even where a person's lingering trail had been deposited. Full
        clear at the centre, fading to no effect at the AoE edge."""
        phi = self.aoe_radius
        pr, pc = pos
        rad = int(math.ceil(phi))
        for r in range(max(0, pr - rad), min(self.rows, pr + rad + 1)):
            risk_row = self._residual_risk[r]
            for c in range(max(0, pc - rad), min(self.cols, pc + rad + 1)):
                d = math.hypot(r - pr, c - pc)
                if d < phi:
                    keep = max(d - 1.0, 0.0) / phi      # 0 at centre, 1 at edge
                    if keep < 1.0:
                        risk_row[c] *= keep

    def _recompute_risk(self) -> None:
        """Refresh the persistent residual-risk field r*_r for this step:

          1. decay the whole field by ``trail_decay`` so cells a person has
             left keep a fading, lingering risk (the residual-risk trail);
          2. stamp each live person's narrow Gaussian bump at their CURRENT
             cell, scaled by their carrying risk r*_c, via max — so the live
             area around a person stays fully "hot"; and
          3. recompute the instantaneous human-density field (no trail) for the
             social-layer interest targets and the delay metric.

        Obstacle cells are forced to zero. Stamping is windowed to ~3σ around
        each person for speed, since the bump is negligible beyond that.
        """
        grid = self.world.grid
        humans = self._humans
        two_sigma_sq = 2.0 * self.risk_sigma * self.risk_sigma
        decay = self.trail_decay
        reach = max(1, int(math.ceil(3.0 * self.risk_sigma)))

        # (1) decay the lingering trail; (3a) reset the instantaneous density.
        for r in range(self.rows):
            risk_row, density_row = self._residual_risk[r], self._density[r]
            for c in range(self.cols):
                if grid[r][c] != 0:
                    risk_row[c] = 0.0
                else:
                    risk_row[c] *= decay
                    if risk_row[c] < 1e-3:
                        risk_row[c] = 0.0
                density_row[c] = 0.0

        if not humans:
            return

        # (2)+(3b) stamp each person's current bump into a 3σ window.
        for hid, (hr, hc) in humans:
            r_c = self._carrying_risk.get(hid, 1.0)
            for r in range(max(0, hr - reach), min(self.rows, hr + reach + 1)):
                risk_row, density_row = self._residual_risk[r], self._density[r]
                for c in range(max(0, hc - reach), min(self.cols, hc + reach + 1)):
                    if grid[r][c] != 0:
                        continue
                    falloff = math.exp(-((r - hr) ** 2 + (c - hc) ** 2) / two_sigma_sq)
                    if falloff > density_row[c]:
                        density_row[c] = falloff
                    weighted = falloff * r_c
                    if weighted > risk_row[c]:
                        risk_row[c] = weighted

    # ------------------------------------------------------------------
    #  Coverage helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _heading(a: Cell, b: Cell) -> Tuple[int, int]:
        return ((b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1]))

    def _cover_footprint(self, cell: Cell) -> None:
        if self.footprint <= 1:
            cells = [cell]
        else:
            r0, c0 = cell
            cells = [(r0 + dr, c0 + dc)
                     for dr in range(-self.footprint + 1, self.footprint)
                     for dc in range(-self.footprint + 1, self.footprint)
                     if math.hypot(dr, dc) < self.footprint]
        for rc in cells:
            if rc in self._region and rc not in self._covered:
                self._order[rc] = len(self._covered)
                self._covered.add(rc)

    def _nearest_uncovered(self) -> Optional[Cell]:
        """First step toward the nearest uncovered in-region cell, returned as an
        immediate NEIGHBOUR of the current node (never a far cell). Like the
        original reference, the robot only ever advances to a node adjacent
        to its current one — it walks the route one neuron at a time rather than
        teleporting to a disconnected uncovered cell. Returns None when nothing
        uncovered is reachable."""
        prev: Dict[Cell, Optional[Cell]] = {self._pos: None}
        queue = deque([self._pos])
        target: Optional[Cell] = None
        while queue:
            cell = queue.popleft()
            if (cell != self._pos and cell in self._region
                    and cell not in self._covered):
                target = cell
                break
            for nbr in self.world.neighbors(cell):
                # stay strictly inside the selected region — the escape route
                # never leaves the selected area
                if nbr not in prev and nbr in self._region:
                    prev[nbr] = cell
                    queue.append(nbr)
        if target is None:
            return None
        step = target
        while prev[step] is not None and prev[step] != self._pos:
            step = prev[step]
        return step

    # ------------------------------------------------------------------
    #  Public coverage API
    # ------------------------------------------------------------------
    def begin(self, start: Cell, region: Optional[Set[Cell]] = None) -> Cell:
        """Start an online coverage session from ``start``, optionally limited
        to ``region`` (a selected area).

        When a ``region`` is given it is taken as-is — the robot's STATIC
        coverage responsibility — and is NOT intersected with the currently-free
        cells. This matters because people are dynamic obstacles: a cell where
        someone happens to stand at drop-time must stay part of the region so it
        gets covered once they move off it (rather than being permanently carved
        out). Cells momentarily blocked by a person are simply skipped each step
        (they are -1 in the live grid) and swept later."""
        free = set(self.world.free_cells())
        self._region = free if region is None else set(region)
        self._covered = set()
        self._order = {}
        self._pos = start
        self._heading_vec = (0, 0)
        # fresh episode: everyone starts at full carrying risk r*_c and the
        # residual-risk field is wiped clean (no trail carried over).
        self._carrying_risk = {hid: 1.0 for hid, _ in self._humans}
        for r in range(self.rows):
            for c in range(self.cols):
                self._residual_risk[r][c] = 0.0
                self._density[r][c] = 0.0
        # Fresh neural state: both GBNN layers start from rest so the field
        # propagates cleanly from this episode's covered/human configuration.
        self.gbnn_coverage.activity = [[0.0] * self.cols for _ in range(self.rows)]
        self.gbnn_social.activity = [[0.0] * self.cols for _ in range(self.rows)]
        self._cover_footprint(start)
        self._service_people_near(start)
        self._recompute_risk()
        self._scrub_trail(start)
        return start

    @property
    def covered_fraction(self) -> float:
        return len(self._covered) / len(self._region) if self._region else 1.0

    def step(self) -> Optional[Cell]:
        """Advance coverage by one cell, refreshing both neural layers from the
        current state; returns the cell stepped to, or None when complete.

        Paper Eqn 10 — the next waypoint is the free neighbour with the highest
        SUM of the two layers:

            wp_{i+1} = max( N_kν  +  risk_bias · N_kμ )

        where ν is the coverage layer and μ the social layer. Each layer's
        activity is ``-1`` on occupied cells — but occupied cells are already
        excluded from the neighbour list, so a ``-1`` node can never be selected.
        Because the layers are *summed* (not a beeline) the robot keeps sweeping
        while leaning toward human areas. Set ``human_first=False`` to drop μ and
        run the bare GBNN. A small turn penalty and covered-cell penalty break
        ties. (risk_bias defaults to 1.0, i.e. the plain Eqn-10 sum.)
        """
        if not self._region or len(self._covered) >= len(self._region):
            return None

        self._service_people_near(self._pos)
        self._recompute_risk()
        self._scrub_trail(self._pos)

        def in_region(r: int, c: int) -> float:
            return 1.0 if (r, c) in self._region else 0.0

        # Coverage layer ν — relax one update this step. Activity persists
        # between steps (gbnn_coverage.activity), so the field keeps propagating
        # across the grid exactly as in the reference dynamics.
        nu = self.gbnn_coverage.relax(
            self.gbnn_coverage.external_input(self._covered, boost=in_region),
            iters=self.relax_iters)

        # Social layer μ — relax the same way. Its sources (interest targets)
        # are the uncovered cells in a person's vicinity (density >= threshold);
        # every other free cell gets neutral input, obstacles -E. As human cells
        # are covered the sources vanish and μ fades, so the robot moves on to
        # the next crowd and finally to plain coverage.
        mu = None
        if self.human_first:
            def human_src(r: int, c: int) -> float:
                return 1.0 if (self._density[r][c] >= self.risk_threshold
                               and (r, c) in self._region) else 0.0
            mu = self.gbnn_social.relax(
                self.gbnn_social.external_input(self._covered, boost=human_src),
                iters=self.relax_iters)

        nxt: Optional[Cell] = None
        best_score = -math.inf
        for nbr in self.world.neighbors(self._pos):       # N_k, excludes -1 nodes
            if nbr not in self._region:                   # never leave the area
                continue
            r, c = nbr
            # Eqn 10: sum of both layers at this neighbour, N_kν + risk_bias·N_kμ
            score = nu[r][c]
            if mu is not None:
                score += self.risk_bias * mu[r][c]
            if self._heading(self._pos, nbr) != self._heading_vec:
                score -= self.turn_penalty
            if nbr in self._covered:
                score -= 0.05
            if score > best_score:
                best_score, nxt = score, nbr
        if nxt is None or nxt in self._covered:
            nxt = self._nearest_uncovered()
            if nxt is None:
                return None

        self._heading_vec = self._heading(self._pos, nxt)
        self._pos = nxt
        self._cover_footprint(nxt)
        return nxt

    def plan(self, start: Optional[Cell] = None,
             region: Optional[Set[Cell]] = None) -> CoverageResult:
        """Batch coverage: run the online loop to completion using the world's
        own human cells. Returns coverage statistics."""
        if start is None:
            start = start_cell(self.world)
        people = getattr(self.world, "humans", []) or []
        self.update_humans([(i, r, c) for i, (r, c) in enumerate(people)])
        self.begin(start, region)
        path = [start]
        max_steps = len(self._region) * 12
        steps = 0
        while len(self._covered) < len(self._region) and steps < max_steps:
            steps += 1
            nxt = self.step()
            if nxt is None:
                break
            path.append(nxt)
        # Delay metric uses the un-serviced human density so it measures whether
        # crowded cells were covered early, independent of risk mitigation.
        delay = sum(order * self._density[cell[0]][cell[1]]
                    for cell, order in self._order.items())
        return CoverageResult(path, len(self._covered), len(self._region),
                              steps, delay)


# ----------------------------------------------------------------------
def _demo(rows: int, cols: int, seed: int) -> None:
    world = coverage_scenario(rows, cols, seed)
    start = start_cell(world)
    hfa = HFACoveragePlanner(world, human_first=True).plan(start)
    base = HFACoveragePlanner(world, human_first=False).plan(start)

    print(f"HFA-CCPP coverage — {rows}x{cols} grid, seed {seed}")
    print(f"  free cells : {world.n_free()}   humans : {world.humans}")
    print()
    print(f"  base GBNN  : coverage={base.coverage_fraction*100:5.1f}%  "
          f"steps={base.steps:4d}  human_weighted_delay={base.human_weighted_delay:8.1f}")
    print(f"  HFA-CCPP   : coverage={hfa.coverage_fraction*100:5.1f}%  "
          f"steps={hfa.steps:4d}  human_weighted_delay={hfa.human_weighted_delay:8.1f}")
    if base.human_weighted_delay > 0:
        drop = 100.0 * (base.human_weighted_delay - hfa.human_weighted_delay) \
            / base.human_weighted_delay
        print()
        print(f"  earlier human-area coverage (delay reduction): {drop:+5.1f}%")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HFA-CCPP coverage demo")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    _demo(args.rows, args.cols, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
