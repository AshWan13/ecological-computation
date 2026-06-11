#!/usr/bin/env python3
"""
ppstar/open_ppstar.py  --  NOVELTY of this module
======================================================

Predator-Dominance and Prey-Approach path planner (PP*) for a
mosquito vector-control robot.

    Wan Sang, Veerajagadheswar, Elara & Le, "Efficient Path Planner via Predator
    Dominance and Prey Approach for a Vector Surveillance Robot",
    ICSR+AI 2025, LNCS/LNAI vol. 16131, pp. 281-294.
    https://doi.org/10.1007/978-981-95-2379-5_19

Idea
----
The robot lures-and-traps mosquitoes with pheromones. It must reach the
mosquito hotspot (the *prey*) but must not drag its mosquito cloud through a
crowd. PP* augments a grid A* with two terms on the cost-to-go of a cell:

    q(n) = g(n) + h1(n) + h0(n)            (paper Eqn 6)

  * ``g``  — accumulated Euclidean path cost (8-connected, diagonal = sqrt2).
  * ``h1`` — Euclidean distance to the goal: the **prey-approach** pull (Eqn 4).
  * ``h0`` — the **predator-dominance** penalty around a crowd (Eqn 3):

        h0(n) = (Cs − Ct) · max(Cr² − dist(n, P0)², 0)

    i.e. zero outside the dominance radius Cr and rising quadratically toward
    the crowd centre P0 inside it.

Predator = a CLUSTER of people, not a single person
---------------------------------------------------
The crowd is **not** one-predator-per-person. People standing close together are
grouped into a proximity cluster (single-linkage within ``link_dist``), and the
cluster's SIZE sets its characteristics, exactly as the paper's crowd size Cs:

  * dominance strength  (Cs − Ct)  scales with the number of people N, so the
    robot threads through a lone person but takes a wider berth around a crowd;
  * dominance radius Cr is the base radius plus the cluster's spatial spread.

``h0`` is deliberately *non-admissible*: it is a repulsion field, not a lower
bound, so the optimal-w.r.t-distance path is intentionally bent away from the
crowd. With ``strength = 0`` the planner reduces to plain A* (``astar``); drop
``h1`` as well and it is Dijkstra (``dijkstra``) — both kept as baselines.

``ppstar_maze(maze, start, end, crowd, setnow)`` is the module's main entry
(mirroring the reference code's function name).

Pure standard library. Run standalone:

    python -m ppstar.open_ppstar --scenario 1
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from common.environment import GridWorld, Cell, pp_scenario, predator_setting
except ImportError:  # pragma: no cover - direct-script fallback
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.environment import GridWorld, Cell, pp_scenario, predator_setting


@dataclass
class Predator:
    """A proximity cluster of people modelled as one predator, described by the
    paper's crowd variables (Eqn 3):

      * ``Cs`` — crowd size: the population weight, ``cs_per_person · N``, so it
        scales with the number of people in the cluster;
      * ``Ct`` — crowd tolerance: the robot's allowance toward the crowd;
      * ``Cr`` — crowd radius: the dispersion radius (base radius + the cluster's
        spatial spread).

    The dominance penalty it contributes is ``(Cs − Ct) · max(Cr² − d², 0)``."""

    center: Cell        # geometric centre P0 of the cluster
    count: int          # N — number of people in the cluster
    Cs: float           # crowd size  (population weight = cs_per_person · N)
    Ct: float           # crowd tolerance
    Cr: float           # crowd radius (base radius + cluster spread)

    @property
    def strength(self) -> float:
        """Dominance weight (Cs − Ct), clamped at zero (paper Eqn 3)."""
        return max(self.Cs - self.Ct, 0.0)

    @property
    def radius(self) -> float:
        """Alias for the crowd radius Cr (the drawn dominance circle)."""
        return self.Cr


@dataclass
class PlanResult:
    """Outcome of a single plan."""

    path: List[Cell]
    start: Cell
    goal: Cell
    length: float        # geometric path length (cells; diagonal = sqrt2)
    steps: int           # number of cells on the path
    expanded: int        # nodes expanded (the original benchmark's "loop")
    crowd_exposure: float  # accumulated predator-proximity along the path

    @property
    def reached(self) -> bool:
        return bool(self.path) and self.path[-1] == self.goal


def cluster_predators(points: Sequence[Tuple[float, float]],
                      link_dist: float,
                      base_radius: float,
                      cs_per_person: float,
                      tolerance: float = 0.0) -> List[Predator]:
    """Group ``points`` (people positions, any consistent unit) into proximity
    clusters and return one :class:`Predator` per cluster, described by the
    paper's crowd variables Cs, Ct, Cr.

    Single-linkage: two people join the same cluster when they are within
    ``link_dist`` of each other (chains transitively). For each cluster:

      * ``Cs`` = ``cs_per_person · N`` — crowd size scaling with the member
        count N (the population);
      * ``Ct`` = ``tolerance`` — crowd tolerance;
      * ``Cr`` = ``base_radius`` + the cluster's spread (max member distance
        from the centre) — the dispersion radius.

    A lone person reproduces the single-predator behaviour (Cs = cs_per_person,
    Cr = base_radius)."""
    pts = [(float(a), float(b)) for a, b in points]
    n = len(pts)
    if n == 0:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ld2 = link_dist * link_dist
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            if dx * dx + dy * dy <= ld2:
                parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    predators: List[Predator] = []
    for members in groups.values():
        N = len(members)
        cx = sum(pts[i][0] for i in members) / N
        cy = sum(pts[i][1] for i in members) / N
        spread = (max(math.hypot(pts[i][0] - cx, pts[i][1] - cy) for i in members)
                  if N > 1 else 0.0)
        predators.append(Predator(
            (int(round(cx)), int(round(cy))),
            N,                              # count
            cs_per_person * N,             # Cs — crowd size (population weight)
            tolerance,                     # Ct — crowd tolerance
            base_radius + spread))         # Cr — crowd radius (dispersion)
    return predators


class PPStar:
    """PP* — Predator-Dominance & Prey-Approach planner (Wan Sang et al.,
    ICSR+AI 2025) over a :class:`GridWorld`. ``q = g + h1(prey) + h0(predators)``,
    where the predators are proximity clusters of the crowd (see
    :func:`cluster_predators`)."""

    # 8-connected moves, matching the research code's neighbour order.
    _MOVES = [(0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(
        self,
        world: GridWorld,
        *,
        cs_per_person: Optional[float] = None,  # Cs contributed per person
        tolerance: float = 0.0,                 # crowd tolerance Ct
        radius: Optional[float] = None,         # base crowd radius Cr
        link_dist: Optional[float] = None,      # proximity threshold for clustering
    ) -> None:
        self.world = world
        s, rad = predator_setting(world.rows)
        # Default Cs/person from the scenario preset so a single crowd point
        # reproduces the benchmark strength (Cs − Ct with Ct = 0).
        self.cs_per_person = s if cs_per_person is None else cs_per_person
        self.tolerance = tolerance               # Ct
        self.radius = rad if radius is None else radius   # Cr base
        # People within ~2/3 of the dominance radius (~2 m at the demo's 3 m
        # radius) count as one crowd.
        self.link_dist = self.radius * 0.66 if link_dist is None else link_dist
        # The crowd: the scenario's predator point (if any) plus every human
        # cell, de-duplicated. Used both to build the predator clusters and to
        # report exposure across all algorithms.
        crowd: List[Cell] = []
        if world.predator is not None:
            crowd.append(world.predator)
        crowd.extend(world.humans)
        self.crowd: List[Cell] = list(dict.fromkeys(crowd))
        # Proximity-clustered predators — each cluster's Cs scales with its
        # member count, NOT one predator per person.
        self.predators: List[Predator] = cluster_predators(
            self.crowd, self.link_dist, self.radius, self.cs_per_person,
            self.tolerance)

    # ------------------------------------------------------------------
    def _h0(self, cell: Cell, predators: List[Predator]) -> float:
        """Predator-dominance penalty (paper Eqn 3), summed over the crowd
        clusters: each adds ``strength · max(radius² − dist², 0)``."""
        if not predators:
            return 0.0
        total = 0.0
        cr, cc = cell
        for p in predators:
            weight = p.Cs - p.Ct                 # (Cs − Ct)
            if weight <= 0.0:
                continue
            d2 = (cr - p.center[0]) ** 2 + (cc - p.center[1]) ** 2
            total += weight * max(p.Cr * p.Cr - d2, 0.0)
        return total

    def _search(
        self,
        start: Cell,
        goal: Cell,
        *,
        use_h1: bool,
        predators: List[Predator],
    ) -> PlanResult:
        """A*/Dijkstra/PP* core (selected by use_h1 and predators)."""
        w = self.world
        # g accumulates the TRUE cost = step distance + predator penalty h0, so
        # the reconstructed path minimises distance *and* crowd exposure and
        # therefore routes around the predator clusters. h1 is the admissible A*
        # heuristic (Euclidean-to-goal) and only steers the search order.
        g: Dict[Cell, float] = {start: 0.0}
        came: Dict[Cell, Cell] = {}
        open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
        closed = set()
        expanded = 0

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            closed.add(cur)
            expanded += 1
            if cur == goal:
                break
            cx, cy = cur
            for dx, dy in self._MOVES:
                nx, ny = cx + dx, cy + dy
                if not w.passable(nx, ny):
                    continue
                if dx != 0 and dy != 0:  # no obstacle corner-cutting
                    if not w.passable(cx + dx, cy) or not w.passable(cx, cy + dy):
                        continue
                nbr = (nx, ny)
                step = math.hypot(dx, dy)
                tentative = g[cur] + step + self._h0(nbr, predators)
                if nbr not in g or tentative < g[nbr]:
                    g[nbr] = tentative
                    came[nbr] = cur
                    h1 = math.hypot(nx - goal[0], ny - goal[1]) if use_h1 else 0.0
                    heapq.heappush(open_heap, (tentative + h1, nbr))

        if goal not in came and goal != start:
            return PlanResult([], start, goal, 0.0, 0, expanded, 0.0)

        path: List[Cell] = [goal]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()

        length = sum(math.hypot(a[0] - b[0], a[1] - b[1])
                     for a, b in zip(path, path[1:]))
        # Exposure is always measured against the actual crowd of people, so the
        # baselines (which ignore the crowd for cost) still report how much human
        # proximity their path incurs.
        exposure = sum(self._proximity(cell) for cell in path)
        return PlanResult(path, start, goal, length, len(path), expanded, exposure)

    def _proximity(self, cell: Cell) -> float:
        """Summed 0..1 closeness-to-people score, for reporting exposure."""
        if not self.crowd or not self.radius:
            return 0.0
        total = 0.0
        for (pr, pc) in self.crowd:
            d = math.hypot(cell[0] - pr, cell[1] - pc)
            total += max(0.0, 1.0 - d / self.radius)
        return total

    # ----- public planners --------------------------------------------
    def dijkstra(self, start: Cell, goal: Cell) -> PlanResult:
        """Uniform-cost shortest path (baseline)."""
        return self._search(start, goal, use_h1=False, predators=[])

    def astar(self, start: Cell, goal: Cell) -> PlanResult:
        """Plain A* with Euclidean heuristic (baseline)."""
        return self._search(start, goal, use_h1=True, predators=[])

    def ppstar(self, start: Cell, goal: Cell,
               predators: Optional[List[Predator]] = None) -> PlanResult:
        """PP*: A* with prey approach and predator-dominance avoidance.

        ``predators`` defaults to the proximity-clustered crowd.
        """
        if predators is None:
            predators = self.predators
        return self._search(start, goal, use_h1=True, predators=predators)

    def plan(self, start: Cell, goal: Cell, algorithm: str = "ppstar") -> PlanResult:
        """Dispatch by name: 'dijkstra' | 'astar' | 'ppstar'."""
        algo = algorithm.lower()
        if algo == "dijkstra":
            return self.dijkstra(start, goal)
        if algo == "astar":
            return self.astar(start, goal)
        if algo == "ppstar":
            return self.ppstar(start, goal)
        raise ValueError(f"unknown algorithm {algorithm!r}")


# ----------------------------------------------------------------------
def ppstar_maze(maze: List[List[int]], start: Cell, end: Cell,
                crowd: Sequence[Cell],
                setnow: Optional[Sequence[float]] = None) -> PlanResult:
    """Main PP* entry (mirrors the reference ``ppstar_maze``).

    Plan from ``start`` to ``end`` on ``maze`` (0 = free, 1 = obstacle), avoiding
    ``crowd`` — a list of *people* cells that are clustered by proximity into
    predators whose crowd size Cs scales with the cluster's member count.
    ``setnow`` optionally overrides the crowd model as
    ``[Cs_per_person, Ct (tolerance), Cr (radius)]`` (the paper's ``setting()``
    triple); when omitted the scenario default is used. Returns a
    :class:`PlanResult`."""
    grid = [list(row) for row in maze]
    world = GridWorld(grid=grid, humans=list(crowd))
    if setnow is None:
        planner = PPStar(world)
    else:
        cs, ct, cr = float(setnow[0]), float(setnow[1]), float(setnow[2])
        planner = PPStar(world, cs_per_person=cs, tolerance=ct, radius=cr)
    return planner.ppstar(start, end)


def _demo(scenario: int) -> None:
    world = pp_scenario(scenario)
    planner = PPStar(world)
    start, goal, pred = world.start, world.goal, world.predator

    print(f"PP* planner — scenario {scenario}  ({world.rows}x{world.cols})")
    print(f"  start={start}  prey(goal)={goal}  crowd={pred}  "
          f"predators(clusters)={len(planner.predators)}  "
          f"Cs/person={planner.cs_per_person:.0f}  Ct={planner.tolerance:.0f}  "
          f"Cr={planner.radius:.0f}")
    for p in planner.predators:
        print(f"    cluster centre={p.center}  N={p.count}  "
              f"Cs={p.Cs:.0f}  Ct={p.Ct:.0f}  Cr={p.Cr:.1f}  "
              f"strength(Cs-Ct)={p.strength:.0f}")
    print()
    print(f"  {'algorithm':<10} {'len':>6} {'steps':>6} {'expanded':>9} "
          f"{'crowd_exposure':>15}")
    for name in ("dijkstra", "astar", "ppstar"):
        t0 = time.perf_counter()
        res = planner.plan(start, goal, name)
        ms = (time.perf_counter() - t0) * 1000.0
        label = "PP*" if name == "ppstar" else name
        print(f"  {label:<10} {res.length:6.1f} {res.steps:6d} {res.expanded:9d} "
              f"{res.crowd_exposure:15.2f}   ({ms:.2f} ms)")

    base = planner.astar(start, goal)
    pd = planner.ppstar(start, goal)
    if base.crowd_exposure > 0:
        drop = 100.0 * (base.crowd_exposure - pd.crowd_exposure) / base.crowd_exposure
        extra = 100.0 * (pd.length - base.length) / base.length if base.length else 0.0
        print()
        print(f"  PP* vs A*: crowd-exposure {drop:+.1f}%  "
              f"(for {extra:+.1f}% path length)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PP* point-to-point planner demo")
    ap.add_argument("--scenario", type=int, default=1, choices=(1, 2, 3))
    args = ap.parse_args(argv)
    _demo(args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
