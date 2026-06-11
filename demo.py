#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py — Continuous park sandbox for Ecological Computation.

Unified pygame host for both path-planning modules:

    ppstar/open_ppstar.py   point-to-point  (PP*, ICSR+AI 2025)
    hfaccpp/open_hfaccpp.py       area coverage    (HFA-CCPP, Sci. Rep. 2025)

Layered after the sibling-repo demos:
  * common/features.py  — the park feature palette + occupancy rasteriser.
  * common/robot.py     — differential-drive robots with a polygon body.
  * the planners        — consume the rasterised grid; humans come from the
                          live human agents in the scene.

Iterative planning
------------------
Both planners run *every tick* against the current human positions, so paths
adapt as people move:
  * PP*  re-plans the full point-to-point path each tick on a FINE grid.
  * HFA-CCPP advances one coverage step each tick on a coarse Cartesian BOX
    grid, with the human-first ordering refreshed from current humans.

Run from the repo root:
    python demo.py            # interactive sandbox
    python demo.py --test     # headless sanity check (no display)

Controls
--------
  1 .. 5               spawn-or-select robot with that id
  W / S                drive the selected robot forward / backward
  A / D                rotate counter-clockwise / clockwise (differential)
  Q / E                increase / decrease speed scale
  P                    cycle point-to-point planner: PP* -> A* -> Dijkstra
  H                    toggle human-first coverage (HFA-CCPP vs base GBNN)
  SHIFT (hold)         overlay active planner layers: PP* predator zones +
                       prey goal; HFA-CCPP mosquito-risk / occupied / open
  left-click (empty)   dispatch the selected robot point-to-point (PP*) there
  left-drag (empty)    select a rectangle -> HFA-CCPP coverage of that area
  right palette        pick a feature, then click / click-drag on the park
  double-click         despawn the feature under the cursor
  SPACE                stop the selected robot
  X                    cancel the selected robot's plan
  C                    clear all features
  R                    reset the scenario
  ESC                  quit
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

from common.environment import GridWorld, Cell, coverage_scenario, start_cell
from common.features import (
    FeatureManager, FeatureKind, PALETTE, DRAG_SPAWN, LASSO_SPAWN, WALKABLE,
    FEATURE_LABELS, draw_feature, draw_human, draw_mosquito,
    Species, SPECIES_LABEL, species_activity,
)
from common.robot import Robot
from ppstar.open_ppstar import PPStar, cluster_predators
from hfaccpp.open_hfaccpp import HFACoveragePlanner


# ---- world geometry --------------------------------------------------------
CANVAS_W, CANVAS_H = 962, 650
HUD_H = 92
TOOLBAR_W = 150
WINDOW_W, WINDOW_H = CANVAS_W + TOOLBAR_W, CANVAS_H + HUD_H
FPS = 30

GP_FINE = 13                      # PP* fine grid pitch (px)
GP_BOX = 26                       # HFA-CCPP Cartesian box pitch (px)
COLS_F, ROWS_F = CANVAS_W // GP_FINE, CANVAS_H // GP_FINE
COLS_B, ROWS_B = CANVAS_W // GP_BOX, CANVAS_H // GP_BOX

# ---- mosquito attraction (exponential kernel, grounded in cited refs) ------
# Park scale assumption: canvas ~30 m wide -> ~32 px/m (tunable).
PX_PER_M = 32.0
# lambda (e-folding length) in px; effective reach ~ 3*lambda.
LAM_CO2 = 6.0 * PX_PER_M      # human CO2 plume  (Gillies & Wilkes 1969, ~18 m)
LAM_PHERO = 0.6 * PX_PER_M    # pheromone/odour  (Bernier 2015 / Davis 1984, <2 m)
LAM_UV = 1.7 * PX_PER_M       # UV light         (Moore 2001, ~5 m), light-gated
LAM_BUSH = 0.8 * PX_PER_M     # bush harborage   (uncited; resting pull)
# Robot lure reaches (px), effective reach ≈ 3·λ for the short pheromone plume
# and 2.2·λ for the UV lamp (matches the drawn glow). The robot's risk-clearing
# area-of-effect is whichever reaches FURTHER — UV or pheromone — so AOE_PX is
# the single source of truth for both the service radius and the rendered wave.
UV_REACH_PX = LAM_UV * 2.2        # ≈ 120 px (~3.7 m)
PHERO_REACH_PX = LAM_PHERO * 3.0  # ≈ 58 px (~1.8 m)
AOE_PX = max(UV_REACH_PX, PHERO_REACH_PX)   # robot AoE = the further of the two
# Cue strengths (free gain — the grounded part is the lambda *ratios* above).
S_CO2, S_PHERO, S_UV, S_BUSH = 3.5, 6.0, 8.0, 2.5
MOZ_LURE_CAP = 0.10           # max per-tick probability of a lure step (10%)
MOZ_STEP = 2.5                # mosquito flight step (px/tick)
MOZ_CAPTURE_R = 16.0          # robot trap capture radius (px)
MOZ_CATCH = 0.05              # capture probability per SECOND while on the robot
MOZ_CAP = 200                 # population ceiling
MOZ_INITIAL = 20              # mosquitoes at sim start
PP_REPLAN_TICKS = 2           # PP* re-plans every 2 ticks on the live crowd
PP_CLUSTER_LINK_PX = 2.0 * PX_PER_M  # ~2 m: proximity to merge people into one crowd
# PP* predator (human-avoidance) tuning on the FINE grid. The reference code
# scales the avoidance radius with grid size (~rows/4); here we set a physical
# per-human radius instead so humans actually bend the path on the big grid.
PP_AVOID_M = 3.0                                  # human-avoidance radius (m)
PP_RADIUS_CELLS = PP_AVOID_M * PX_PER_M / GP_FINE  # ≈ 7.4 fine cells (~3 m), Cr base
# Paper crowd model: Cs (crowd size) = PP_CS_PER_PERSON·N, Ct (tolerance), and
# Cr (radius). Dominance weight = (Cs − Ct), so the robot tolerates a lone
# person more and avoids larger crowds (here a single person → Cs−Ct = 6).
PP_CS_PER_PERSON = 10.0                             # Cs contributed per person
PP_TOLERANCE = 4.0                                 # Ct — crowd tolerance
# (robot risk-servicing radius now == UV_REACH_PX above, shared with the glow)

# Feature draw z-order (lower drawn first).
_ZORDER = {
    FeatureKind.PAVEMENT: 0, FeatureKind.POND: 1,
    FeatureKind.SHELTER: 2, FeatureKind.PAVILION: 2, FeatureKind.BUILDING: 2,
    FeatureKind.BUSH: 3, FeatureKind.EATERY: 3, FeatureKind.TOILET: 3,
    FeatureKind.TREE: 4,
}

# ---- environment: time-of-day + weather (tunable on the side panel) --------
TIMES = ["Dawn", "Day", "Dusk", "Night"]
WEATHERS = ["Clear", "Rain", "Windy"]
# Ambient light 0..1 (gates UV), temperature (°C, Singapore-like), and a
# mosquito-activity multiplier — all discrete by (time, weather).
_ENV_LIGHT = {"Dawn": 0.50, "Day": 1.00, "Dusk": 0.40, "Night": 0.12}
_ENV_TEMP = {"Dawn": 26.0, "Day": 32.0, "Dusk": 29.0, "Night": 26.0}
_ENV_ACT_TIME = {"Dawn": 1.0, "Day": 0.65, "Dusk": 1.0, "Night": 0.8}
_ENV_ACT_WX = {"Clear": 1.0, "Rain": 0.5, "Windy": 0.7}


def env_state(tod: str, wx: str):
    """Return (light 0..1, temperature °C, activity multiplier) for the
    discrete time-of-day + weather combination."""
    light, temp = _ENV_LIGHT[tod], _ENV_TEMP[tod]
    if wx == "Rain":
        light *= 0.6
        temp -= 3.0
    elif wx == "Windy":
        temp -= 1.0
    return light, temp, _ENV_ACT_TIME[tod] * _ENV_ACT_WX[wx]


# ============================================================================
#  HEADLESS SELF-TEST
# ============================================================================
def _run_headless_test() -> int:
    from common.environment import pp_scenario
    ok = True

    for scenario in (1, 2, 3):
        world = pp_scenario(scenario)
        planner = PPStar(world)
        s, g = world.start, world.goal
        dij, ast, pdp = (planner.dijkstra(s, g), planner.astar(s, g),
                         planner.ppstar(s, g))
        if not (dij.reached and ast.reached and pdp.reached):
            print(f"[FAIL] scenario {scenario}: a planner failed"); ok = False
        if pdp.crowd_exposure > ast.crowd_exposure + 1e-9:
            print(f"[FAIL] scenario {scenario}: PP* exposure exceeds A*"); ok = False
        print(f"PP*  scenario {scenario}: A* exposure {ast.crowd_exposure:.2f} "
              f"| PP* exposure {pdp.crowd_exposure:.2f}")

    for seed in (0, 1, 2):
        world = coverage_scenario(12, 12, seed)
        start = start_cell(world)
        hfa = HFACoveragePlanner(world, human_first=True).plan(start)
        base = HFACoveragePlanner(world, human_first=False).plan(start)
        if not (hfa.complete and base.complete):
            print(f"[FAIL] seed {seed}: coverage incomplete"); ok = False
        if hfa.human_weighted_delay > base.human_weighted_delay + 1e-9:
            print(f"[FAIL] seed {seed}: HFA-CCPP not human-first"); ok = False
        drop = (100.0 * (base.human_weighted_delay - hfa.human_weighted_delay)
                / base.human_weighted_delay) if base.human_weighted_delay else 0.0
        print(f"HFA-CCPP seed {seed}: coverage 100% | human-delay {drop:+.1f}%")

    # online coverage step API
    w = coverage_scenario(12, 12, 0)
    p = HFACoveragePlanner(w, human_first=True)
    p.update_humans([(i, r, c) for i, (r, c) in enumerate(w.humans)])
    p.begin(start_cell(w))
    n = 0
    while p.step() is not None:
        n += 1
    if p.covered_fraction < 0.999:
        print("[FAIL] online coverage incomplete"); ok = False
    print(f"online coverage: {n} steps, {p.covered_fraction*100:.0f}% covered")

    # features rasterise: no-entry blocks, walkable does not
    fm = FeatureManager()
    fm.spawn(FeatureKind.EATERY, 100, 100)
    fm.spawn(FeatureKind.PAVEMENT, 300, 300, 40, 40)
    fm.spawn(FeatureKind.TREE, 200, 200)
    grid = fm.build_occupancy_grid(ROWS_F, COLS_F, GP_FINE)
    if not fm.blocks_circle(100, 100, 8):
        print("[FAIL] eatery should block"); ok = False
    if fm.blocks_circle(300, 300, 8):
        print("[FAIL] pavement should be walkable"); ok = False
    if fm.blocks_circle(260, 200, 8):
        print("[FAIL] tree canopy edge should be walkable (stump-only block)"); ok = False
    print(f"features: {sum(r.count(1) for r in grid)} obstacle cells "
          f"(fine grid {ROWS_F}x{COLS_F})")

    if ok:
        print("\nAll headless tests PASSED.")
        return 0
    print("\nHeadless tests FAILED.")
    return 1


# ============================================================================
#  PYGAME SANDBOX (lazy-loaded pygame)
# ============================================================================
def _run_pygame_sandbox() -> int:
    try:
        import pygame
    except ImportError:
        print("pygame is not installed.  Install with:  pip install pygame",
              file=sys.stderr)
        return 2

    HUD_BG = (18, 22, 24)
    HUD_TEXT = (224, 230, 224)
    HUD_ACCENT = (255, 206, 110)
    TOOLBAR_BG = (28, 32, 34)
    TOOLBAR_BTN = (48, 54, 56)
    TOOLBAR_SEL = (90, 130, 180)
    SEL_DRAG = (255, 230, 120)
    PATH_COL = (255, 238, 90)   # bright path line (with a dark underlay)
    DRAG_THRESH = 8
    DBL_MS = 380

    PLANNERS = ("ppstar", "astar", "dijkstra")
    PLABEL = {"ppstar": "PP*", "astar": "A*", "dijkstra": "Dijkstra"}

    def make_grass() -> "pygame.Surface":
        """Non-checkered grass: a solid base textured with short, horizontal
        zig-zag strokes in a few green shades. Fully opaque (no alpha
        compositing) so there are no dark/black patches."""
        surf = pygame.Surface((CANVAS_W, CANVAS_H))
        surf.fill((70, 120, 64))
        rng = random.Random(7)
        shades = [(60, 108, 56), (80, 134, 72), (64, 114, 60), (86, 142, 78)]
        for _ in range(2200):
            x = rng.randint(0, CANVAS_W - 18)
            y = rng.randint(2, CANVAS_H - 3)
            col = rng.choice(shades)
            seg = rng.randint(3, 4)               # 3-4 segments...
            step = rng.randint(3, 4)              # ...~3-4 px each (short)
            amp = rng.choice([1, 2, 2])           # small vertical wobble
            pts = [(x + i * step, y + (amp if i % 2 else -amp))
                   for i in range(seg + 1)]
            pygame.draw.lines(surf, col, False, pts, 1)
        return surf

    class Sandbox:
        def __init__(self) -> None:
            pygame.init()
            self.fm = FeatureManager()
            self.robots: Dict[int, Robot] = {}
            self.selected = 1
            self.scale = 1.0
            self.planner_idx = 0
            self.human_first = True
            self.tool: Optional[FeatureKind] = None
            self.captured = 0           # mosquitoes trapped by robots (cumulative)
            self.time = "Day"           # time-of-day (tunable on the side panel)
            self.weather = "Clear"      # weather (tunable on the side panel)
            self.dropdown = None        # None | "time" | "weather" (open selector)
            self.light = 1.0            # ambient light 0..1 (gates UV)
            self.temp = 32.0            # temperature °C (Singapore-like)
            # missions[rid] = ("p2p", (gx, gy)) | ("cover",)
            self.missions: Dict[int, tuple] = {}
            # cover sessions: rid -> {"planner":HFACoveragePlanner, "target":Cell|None}
            self.cover: Dict[int, dict] = {}
            self.msg = "1-5 spawn robot | WASD drive | click=PP* | drag=coverage"

            self.down: Optional[Tuple[float, float]] = None
            self.down_screen: Optional[Tuple[int, int]] = None
            self.drag_now: Optional[Tuple[float, float]] = None
            self.lasso: List[Tuple[float, float]] = []   # pond freeform path
            self.dragging = False
            self.suppress_up = False
            self.last_click_ms = 0
            self.last_click_pos = (0, 0)
            self.move_feature = None                      # feature being dragged
            self.move_offset = (0.0, 0.0)

            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
            pygame.display.set_caption("Ecological Computation — park sandbox")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("menlo", 12)
            self.font_b = pygame.font.SysFont("menlo", 15, bold=True)
            self.grass = make_grass()
            self.running = True
            self.fm.spawn_mosquitoes(MOZ_INITIAL, (4, 4, CANVAS_W - 4, CANVAS_H - 4))
            self._spawn_robot(1)

        @property
        def planner_name(self) -> str:
            return PLANNERS[self.planner_idx]

        # ----- coordinates -------------------------------------------
        def to_world(self, sx, sy):
            return (float(sx), float(sy - HUD_H))

        def in_canvas(self, sx, sy):
            return 0 <= sx < CANVAS_W and HUD_H <= sy < WINDOW_H

        def in_toolbar(self, sx, sy):
            return sx >= CANVAS_W and sy >= HUD_H

        def passable(self, wx, wy):
            r = 11.0
            if not (r <= wx <= CANVAS_W - r and r <= wy <= CANVAS_H - r):
                return False
            return not self.fm.blocks_circle(wx, wy, r)

        # ----- grids -------------------------------------------------
        # Planning is restricted to walkable (paved) surfaces — autonomous
        # planners never route across open grass, though teleop still can.
        # ``block_humans=True`` also stamps each person's cell as occupied, so a
        # person's node is a -1 obstacle in both the movement graph and the
        # neural layers.
        def fine_world(self) -> GridWorld:
            grid = self.fm.build_occupancy_grid(ROWS_F, COLS_F, GP_FINE,
                                                inflate=1, plannable_only=True,
                                                block_humans=True)
            humans = self.fm.human_cells(ROWS_F, COLS_F, GP_FINE)
            return GridWorld(grid=grid, humans=humans)

        def box_world(self) -> GridWorld:
            # inflate=1 keeps a one-cell clearance around STATIC obstacles so the
            # robot moves smoothly around them (people are stamped afterwards as
            # single cells and are NOT inflated).
            grid = self.fm.build_occupancy_grid(ROWS_B, COLS_B, GP_BOX,
                                                inflate=1, plannable_only=True,
                                                block_humans=True)
            humans = self.fm.human_cells(ROWS_B, COLS_B, GP_BOX)
            return GridWorld(grid=grid, humans=humans)

        def _on_plannable(self, wx, wy) -> bool:
            """True only if the world point lies on a plannable (paved) cell.
            Grass and obstacles are not plannable, so a robot on grass cannot
            begin any autonomous path — only teleoperation may leave pavement."""
            grid = self.fine_world().grid
            r, c = int(wy // GP_FINE), int(wx // GP_FINE)
            return 0 <= r < ROWS_F and 0 <= c < COLS_F and grid[r][c] == 0

        def _on_grass(self, wx, wy) -> bool:
            """True if the point is open grass — i.e. not on any walkable surface
            (pavement / shelter / pavilion). Cheap feature test, used to slow the
            robot down off-pavement."""
            return not any(f.kind in WALKABLE and f.contains(wx, wy)
                           for f in self.fm.features.values())

        @staticmethod
        def _snap_free(grid, rows, cols, cell, max_radius=10 ** 9) -> Optional[Cell]:
            """Nearest free (plannable) cell within ``max_radius`` cells, or None.

            A small radius means a robot/goal sitting on grass won't be snapped
            across the grass onto a far-away path — it simply can't auto-plan.
            """
            r0, c0 = cell
            if 0 <= r0 < rows and 0 <= c0 < cols and grid[r0][c0] == 0:
                return cell
            seen = {cell}
            q = deque([(r0, c0, 0)])
            while q:
                r, c, d = q.popleft()
                if d >= max_radius:
                    continue
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen):
                            if grid[nr][nc] == 0:
                                return (nr, nc)
                            seen.add((nr, nc))
                            q.append((nr, nc, d + 1))
            return None

        # ----- robots ------------------------------------------------
        def _spawn_robot(self, rid: int) -> None:
            if rid in self.robots:
                self.selected = rid
                self.msg = f"selected robot {rid}"
                return
            slot = len(self.robots)
            self.robots[rid] = Robot(rid, 40 + slot * 46, CANVAS_H - 40,
                                     theta=-math.pi / 2)
            self.selected = rid
            self.msg = f"spawned robot {rid}"

        # ----- planning helpers --------------------------------------
        @staticmethod
        def _box_center(cell):
            return ((cell[1] + 0.5) * GP_BOX, (cell[0] + 0.5) * GP_BOX)

        @staticmethod
        def _fine_center(cell):
            return ((cell[1] + 0.5) * GP_FINE, (cell[0] + 0.5) * GP_FINE)

        def _fine_path(self, start_cell, goal_cell):
            """PP* path on the FINE grid, returned as NODE-TO-NODE world
            waypoints (one per grid cell, no line-of-sight smoothing), using the
            live human crowd. The fine grid (GP_FINE) is finer than the HFA-CCPP
            coverage grid (GP_BOX). None if unreachable."""
            world = self.fine_world()
            # The START must already be on a plannable (paved) cell — never snap
            # a path start across grass. The goal keeps a small tolerance for
            # click imprecision near paving edges.
            s = self._snap_free(world.grid, ROWS_F, COLS_F, start_cell, max_radius=0)
            g = self._snap_free(world.grid, ROWS_F, COLS_F, goal_cell, max_radius=2)
            if s is None or g is None:
                return None
            res = PPStar(world, cs_per_person=PP_CS_PER_PERSON,
                         tolerance=PP_TOLERANCE, radius=PP_RADIUS_CELLS,
                         link_dist=PP_CLUSTER_LINK_PX / GP_FINE
                         ).plan(s, g, self.planner_name)
            if not res.reached:
                return None
            # node-to-node: every grid cell on the path becomes a waypoint.
            return [self._fine_center(c) for c in res.path]

        # ----- dispatch ----------------------------------------------
        def _set_p2p(self, wx, wy) -> None:
            rob = self.robots.get(self.selected)
            if rob is None:
                return
            if not self._on_plannable(rob.x, rob.y):
                self.msg = "robot is on grass — drive onto pavement to plan a path"
                return
            self.missions[self.selected] = {"type": "p2p", "goal": (wx, wy),
                                            "hcount": -1, "tick": 0}
            self.msg = f"R{self.selected} dispatched (PP*)"

        def _set_cover(self, x0, y0, x1, y1) -> None:
            rob = self.robots.get(self.selected)
            if rob is None:
                return
            if not self._on_plannable(rob.x, rob.y):
                self.msg = "robot is on grass — drive onto pavement to start coverage"
                return
            lo_x, hi_x = sorted((x0, x1))
            lo_y, hi_y = sorted((y0, y1))
            # Snap the drag OUT to whole box-grid nodes, so the selection is an
            # exact set of grid cells: every covered node is then fully inside
            # the (snapped) area — no node pokes out — and any drag that touches
            # pavement yields at least one whole node (so the robot can move).
            lo_x = (int(lo_x) // GP_BOX) * GP_BOX
            lo_y = (int(lo_y) // GP_BOX) * GP_BOX
            hi_x = -(-int(hi_x) // GP_BOX) * GP_BOX          # ceil to grid line
            hi_y = -(-int(hi_y) // GP_BOX) * GP_BOX
            # The coverage region is the STATIC pavement inside the selection —
            # built WITHOUT block_humans, so a person standing here at drop-time
            # doesn't carve a permanent hole. People are dynamic obstacles: such
            # a node is skipped while occupied and swept once they move off it.
            static_grid = self.fm.build_occupancy_grid(ROWS_B, COLS_B, GP_BOX,
                                                       inflate=1,
                                                       plannable_only=True)
            region = {(r, c) for r in range(ROWS_B) for c in range(COLS_B)
                      if static_grid[r][c] == 0
                      and lo_x <= c * GP_BOX and (c + 1) * GP_BOX <= hi_x
                      and lo_y <= r * GP_BOX and (r + 1) * GP_BOX <= hi_y}
            if not region:
                self.msg = "no pavement in selection — draw Pavement, then drag over it"
                return
            # nearest node of the selected area to the robot
            nb = min(region, key=lambda rc: (self._box_center(rc)[0] - rob.x) ** 2
                     + (self._box_center(rc)[1] - rob.y) ** 2)
            # plan a paved approach to that node; if there is no path on pavement
            # (it would have to cross grass), refuse rather than move.
            ncx, ncy = self._box_center(nb)
            path = self._fine_path((int(rob.y // GP_FINE), int(rob.x // GP_FINE)),
                                   (int(ncy // GP_FINE), int(ncx // GP_FINE)))
            if path is None:
                self.msg = "no paved route to that area — coverage needs a path on pavement"
                return
            m = {"type": "cover", "phase": "goto", "region": region,
                 "start_box": nb, "rect": (lo_x, lo_y, hi_x, hi_y),
                 "trace": [], "planner": None, "target": None}
            self.missions[self.selected] = m
            rob.set_path(path)
            tag = "HFA-CCPP" if self.human_first else "base GBNN"
            self.msg = f"R{self.selected} {tag}: heading to area ({len(region)} boxes)"

        # ----- per-tick mission stepping ------------------------------
        def _tick_p2p(self, rid, m) -> None:
            rob = self.robots[rid]
            gx, gy = m["goal"]
            if math.hypot(gx - rob.x, gy - rob.y) < GP_FINE * 0.8:
                self.missions.pop(rid, None)
                rob.clear_path()
                self.msg = f"R{rid} reached goal"
                return
            m["tick"] += 1
            # Re-plan every 2 ticks (and on spawn/despawn) so PP* tracks the
            # live, moving crowd — its predators are re-clustered from the
            # current human positions each replan.
            hc = len(self.fm.humans)
            if (not rob.has_path) or hc != m["hcount"] or m["tick"] % PP_REPLAN_TICKS == 0:
                path = self._fine_path((int(rob.y // GP_FINE), int(rob.x // GP_FINE)),
                                       (int(gy // GP_FINE), int(gx // GP_FINE)))
                if path is None:
                    rob.clear_path()
                    self.missions.pop(rid, None)
                    self.msg = "auto-nav runs on paths only — drive onto Pavement"
                    return
                rob.set_path(path)
                m["hcount"] = hc

        def _tick_cover(self, rid, m) -> None:
            rob = self.robots[rid]
            if m["phase"] == "goto":
                ncx, ncy = self._box_center(m["start_box"])
                if math.hypot(ncx - rob.x, ncy - rob.y) < GP_BOX * 0.6 or not rob.has_path:
                    world = self.box_world()
                    # AoE that clears mosquito risk == the robot's lure radius
                    # (UV or pheromone, whichever reaches further), in box cells.
                    planner = HFACoveragePlanner(
                        world, human_first=self.human_first,
                        aoe_radius=AOE_PX / GP_BOX)
                    planner.update_humans(self.fm.human_cells_id(ROWS_B, COLS_B, GP_BOX))
                    planner.begin(m["start_box"], m["region"])
                    m["planner"] = planner
                    m["phase"] = "run"
                    m["target"] = m["start_box"]
                    m["trace"] = [self._box_center(m["start_box"])]
                    rob.clear_path()
                return
            # run phase: refresh the live occupancy grid (moving people +
            # obstacles) into both neural layers, then feed live humans so the
            # planner rebuilds its risk layer.
            planner = m["planner"]
            planner.update_world(self.box_world())
            planner.update_humans(self.fm.human_cells_id(ROWS_B, COLS_B, GP_BOX))
            tcx, tcy = self._box_center(m["target"])
            if math.hypot(tcx - rob.x, tcy - rob.y) < GP_BOX * 0.55:
                nxt = planner.step()
                if nxt is None:
                    # Don't declare "complete" if uncovered region nodes are
                    # merely blocked by a person right now — hold position and
                    # sweep them once they step off (their cell is -1 only while
                    # occupied). Only finish when every reachable node is done.
                    blocked = any(cell not in planner._covered
                                  and planner.world.grid[cell[0]][cell[1]] == 1
                                  for cell in planner._region)
                    if blocked:
                        rob.clear_path()
                        self.msg = f"R{rid} waiting for a person to move"
                        return
                    self.missions.pop(rid, None)
                    rob.clear_path()
                    self.msg = f"R{rid} coverage complete"
                    return
                m["target"] = nxt
                c = self._box_center(nxt)
                m["trace"].append(c)
                rob.set_path([c])

        # ----- toolbar ------------------------------------------------
        def toolbar_kind_at(self, sx, sy) -> Optional[FeatureKind]:
            if not self.in_toolbar(sx, sy):
                return None
            idx = (sy - HUD_H - 10) // 40
            return PALETTE[idx] if 0 <= idx < len(PALETTE) else None

        def _panel_y(self):
            return HUD_H + 10 + len(PALETTE) * 40 + 12

        def _env_rows(self):
            y = self._panel_y() + 16
            x, w = CANVAS_W + 8, TOOLBAR_W - 16
            return {"time": pygame.Rect(x, y, w, 28),
                    "weather": pygame.Rect(x, y + 34, w, 28)}

        def _dropdown_rects(self):
            base = self._env_rows()[self.dropdown]
            opts = TIMES if self.dropdown == "time" else WEATHERS
            return [(v, pygame.Rect(base.x, base.y + 30 + i * 24, base.w, 22))
                    for i, v in enumerate(opts)]

        def _toolbar_click(self, sx, sy) -> None:
            # an open dropdown captures the next click
            if self.dropdown is not None:
                for val, rect in self._dropdown_rects():
                    if rect.collidepoint(sx, sy):
                        if self.dropdown == "time":
                            self.time = val
                        else:
                            self.weather = val
                        self.msg = f"{self.dropdown} = {val}"
                        self.dropdown = None
                        return
                self.dropdown = None      # click elsewhere closes it
                return
            rows = self._env_rows()
            if rows["time"].collidepoint(sx, sy):
                self.dropdown = "time"
                return
            if rows["weather"].collidepoint(sx, sy):
                self.dropdown = "weather"
                return
            kind = self.toolbar_kind_at(sx, sy)
            if kind is not None:
                self.tool = None if self.tool == kind else kind
                self.msg = f"tool: {FEATURE_LABELS[self.tool] if self.tool else 'none'}"

        # ----- events -------------------------------------------------
        def on_key(self, key) -> None:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif pygame.K_1 <= key <= pygame.K_5:
                self._spawn_robot(key - pygame.K_0)
            elif key == pygame.K_q:
                self.scale = min(3.0, self.scale + 0.1)
                self.msg = f"speed scale {self.scale:.1f}"
            elif key == pygame.K_e:
                self.scale = max(0.2, self.scale - 0.1)
                self.msg = f"speed scale {self.scale:.1f}"
            elif key == pygame.K_p:
                self.planner_idx = (self.planner_idx + 1) % len(PLANNERS)
                self.msg = f"planner: {PLABEL[self.planner_name]}"
            elif key == pygame.K_h:
                self.human_first = not self.human_first
                self.msg = f"human-first coverage = {self.human_first}"
            elif key in (pygame.K_x, pygame.K_SPACE):
                self.missions.pop(self.selected, None)
                self.cover.pop(self.selected, None)
                if self.selected in self.robots:
                    self.robots[self.selected].clear_path()
                self.msg = f"R{self.selected} stopped"
            elif key == pygame.K_c:
                self.fm.clear()
                self.msg = "features cleared"
            elif key == pygame.K_r:
                self.fm.clear()
                self.fm.spawn_mosquitoes(MOZ_INITIAL, (4, 4, CANVAS_W - 4, CANVAS_H - 4))
                self.robots.clear()
                self.missions.clear()
                self.cover.clear()
                self._spawn_robot(1)
                self.msg = "reset"

        def on_mousedown(self, pos) -> None:
            sx, sy = pos
            now = pygame.time.get_ticks()
            self.suppress_up = False
            if self.in_toolbar(sx, sy):
                self._toolbar_click(sx, sy)
                self.suppress_up = True
                return
            if not self.in_canvas(sx, sy):
                self.dropdown = None
                return
            wx, wy = self.to_world(sx, sy)
            dbl = (now - self.last_click_ms < DBL_MS
                   and abs(sx - self.last_click_pos[0]) < 6
                   and abs(sy - self.last_click_pos[1]) < 6)
            self.last_click_ms, self.last_click_pos = now, (sx, sy)
            if dbl and self.fm.despawn_at(wx, wy):   # any feature/person, any tool
                self.msg = "despawned feature"
                self.suppress_up = True
                return
            # A press on an existing feature/person operates on IT — never spawns
            # on top — so double-click-despawn works even with a tool selected,
            # and a feature can be dragged to move it.
            feat = self.fm.feature_at(wx, wy)
            if feat is not None:
                self.move_feature = feat
                self.move_offset = (wx - feat.x, wy - feat.y)
                return
            if self.fm.human_at(wx, wy) is not None:
                self.suppress_up = True       # don't spawn over a person
                return
            self.down = (wx, wy)
            self.down_screen = (sx, sy)
            self.drag_now = (wx, wy)
            self.lasso = [(wx, wy)] if self.tool in LASSO_SPAWN else []
            self.dragging = True

        def on_mousemotion(self, pos) -> None:
            if self.move_feature is not None:           # dragging a feature
                sx, sy = pos
                wx, wy = self.to_world(max(0, min(sx, CANVAS_W)),
                                       max(HUD_H, min(sy, WINDOW_H)))
                nx = max(0.0, min(float(CANVAS_W), wx - self.move_offset[0]))
                ny = max(0.0, min(float(CANVAS_H), wy - self.move_offset[1]))
                self.fm.move_feature(self.move_feature, nx, ny)
                return
            if self.dragging:
                sx, sy = pos
                self.drag_now = self.to_world(max(0, min(sx, CANVAS_W)),
                                              max(HUD_H, min(sy, WINDOW_H)))
                if self.tool in LASSO_SPAWN:
                    px, py = self.drag_now
                    if not self.lasso or math.hypot(
                            px - self.lasso[-1][0], py - self.lasso[-1][1]) >= 5:
                        self.lasso.append((px, py))

        def on_mouseup(self, pos) -> None:
            if self.move_feature is not None:           # finished moving a feature
                self.msg = f"moved {FEATURE_LABELS[self.move_feature.kind]}"
                self.move_feature = None
                return
            if self.suppress_up or not self.dragging:
                self.dragging = False
                return
            sx, sy = pos
            self.dragging = False
            if self.down is None:
                return
            wx0, wy0 = self.down
            wx1, wy1 = self.to_world(max(0, min(sx, CANVAS_W)),
                                     max(HUD_H, min(sy, WINDOW_H)))
            moved = math.hypot(sx - self.down_screen[0], sy - self.down_screen[1])
            if self.tool in LASSO_SPAWN:
                if len(self.lasso) >= 3:
                    self.fm.spawn_poly(self.tool, self.lasso)
                    self.msg = f"drew {FEATURE_LABELS[self.tool]} ({len(self.lasso)} pts)"
                else:
                    self.msg = "drag to lasso a water body outline"
                self.lasso = []
            elif self.tool is not None:
                if self.tool in DRAG_SPAWN and moved >= DRAG_THRESH:
                    self.fm.spawn(self.tool, (wx0 + wx1) / 2, (wy0 + wy1) / 2,
                                  abs(wx1 - wx0) / 2, abs(wy1 - wy0) / 2)
                else:
                    self.fm.spawn(self.tool, wx0, wy0)
                self.msg = f"placed {FEATURE_LABELS[self.tool]}"
            elif moved < DRAG_THRESH:
                self._set_p2p(wx0, wy0)
            else:
                self._set_cover(wx0, wy0, wx1, wy1)
            self.down = None

        # ----- update -------------------------------------------------
        def _mosquito_attractors(self):
            """Build the attraction-field sources for this tick.

            Human  = CO2 (long) + pheromone (short).
            Robot  = pheromone (short) + UV (medium, gated by ambient light:
                     UV pulls strongly at night/dusk, weakly in daylight).
            Bush   = short-range harborage.
            """
            # cue tuples are (lambda_px, strength, cue_type); cue_type lets each
            # species weight CO2/odour vs UV vs harborage differently.
            attractors = []
            for h in self.fm.humans:
                attractors.append((h.x, h.y,
                                   [(LAM_CO2, S_CO2, "human"),
                                    (LAM_PHERO, S_PHERO, "human")]))
            for f in self.fm.features.values():     # bushes + drains = harborage
                if f.kind in (FeatureKind.BUSH, FeatureKind.DRAIN):
                    attractors.append((f.x, f.y, [(LAM_BUSH, S_BUSH, "harborage")]))
            uv = S_UV * (1.0 - self.light)      # bright -> weak UV; dark -> strong
            for rob in self.robots.values():
                attractors.append((rob.x, rob.y,
                                   [(LAM_PHERO, S_PHERO, "human"),
                                    (LAM_UV, uv, "uv")]))
            return attractors

        def update(self, dt, keys) -> None:
            # time-of-day + weather drive ambient light (UV gate), temperature,
            # and mosquito activity.
            self.light, self.temp, _ = env_state(self.time, self.weather)
            bnds = (4, 4, CANVAS_W - 4, CANVAS_H - 4)
            self.fm.tick_humans((8, 8, CANVAS_W - 8, CANVAS_H - 8))
            # (risk mitigation now lives inside the HFA-CCPP planner, which
            #  services risk along the robot's coverage footprint)
            # per-species activity for the current time-of-day / weather / temp
            activity = {sp: species_activity(sp, self.time, self.weather, self.temp)
                        for sp in Species}
            self.fm.tick_mosquitoes(bnds, self._mosquito_attractors(),
                                    activity, lure_cap=MOZ_LURE_CAP, step=MOZ_STEP)
            # features breed mosquitoes (rain boosts breeding); robots trap them
            breed = 1.6 if self.weather == "Rain" else 1.0
            self.fm.spawn_from_features(PX_PER_M, FPS, breed, MOZ_CAP)
            # 5% capture per SECOND of contact (hard to catch — lets the swarm
            # visibly gather on the UV lure before one is drawn in)
            self.captured += self.fm.trap(
                [(r.x, r.y) for r in self.robots.values()],
                MOZ_CAPTURE_R, catch_prob=MOZ_CATCH / FPS)
            fwd = (1 if keys[pygame.K_w] else 0) - (1 if keys[pygame.K_s] else 0)
            # A = counter-clockwise, D = clockwise (screen is y-down, so CW is
            # +theta and CCW is -theta).
            rot = (1 if keys[pygame.K_d] else 0) - (1 if keys[pygame.K_a] else 0)
            for rid, rob in self.robots.items():
                # Driving over open grass is slow going — halve the speed there
                # (paved surfaces and shelters keep full speed).
                sc = self.scale * (0.5 if self._on_grass(rob.x, rob.y) else 1.0)
                if rid == self.selected and (fwd or rot):
                    self.missions.pop(rid, None)
                    self.cover.pop(rid, None)
                    rob.teleop(fwd, rot, sc, dt, self.passable)
                    continue
                m = self.missions.get(rid)
                if m is None:
                    rob.follow(sc, dt, self.passable)
                elif m["type"] == "p2p":
                    self._tick_p2p(rid, m)
                    rob.follow(sc, dt, self.passable)
                elif m["type"] == "cover":
                    self._tick_cover(rid, m)
                    rob.follow(sc, dt, self.passable)

        # ----- render -------------------------------------------------
        def draw(self) -> None:
            self.screen.fill((0, 0, 0))
            canvas = self.grass.copy()
            # Shelter (canopy) and pavilion (rectangle) are open roofs drawn
            # LAST so robots and humans show *under* them. Buildings are solid
            # and draw at ground level.
            roof_kinds = (FeatureKind.SHELTER, FeatureKind.PAVILION)
            ground = [f for f in self.fm.features.values()
                      if f.kind not in roof_kinds]
            roofs = [f for f in self.fm.features.values()
                     if f.kind in roof_kinds]
            for f in sorted(ground, key=lambda f: _ZORDER.get(f.kind, 9)):
                draw_feature(canvas, f, 0, self.font)
            self._draw_coverage_overlays(canvas)
            for h in self.fm.humans:
                draw_human(canvas, h, 0)
            self._draw_drag_preview(canvas)
            for rob in self.robots.values():
                self._draw_robot(canvas, rob)
            for f in roofs:                     # sheltered roofs over robots/humans
                draw_feature(canvas, f, 0, self.font)
            self._draw_planning_overlays(canvas)   # SHIFT: planner debug layers
            for m in self.fm.mosquitoes:        # tiny dots fly above everything
                draw_mosquito(canvas, m, 0)
            self.screen.blit(canvas, (0, HUD_H))
            self._draw_toolbar()
            self._draw_hud()
            pygame.display.flip()

        def _draw_planning_overlays(self, s) -> None:
            """Hold SHIFT to overlay the active planners' internal layers:
            HFA-CCPP renders the SUMMED two-layer neural field on the box grid,
            colour-mapped by node category (obstacle / covered / uncovered /
            risk-influenced); PP* draws predator circles + centres and the prey
            goal."""
            keys = pygame.key.get_pressed()
            if not (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]):
                return

            # --- HFA-CCPP layers (active coverage missions) ---------------
            # Colour map of the summed neural field (coverage GBNN + social
            # GBNN). Each free node is classed by the live field + coverage
            # state; obstacles are the -1 nodes excluded from selection.
            # Whole overlay is ~50% translucent (alphas capped near 128).
            A = 128                       # max cell opacity (~50%)
            OBSTACLE = (120, 124, 130)   # grey    — occupied (-1), incl. people
            COVERED = (64, 96, 140)      # blue    — already swept (~0 activity)
            UNCOVERED = (70, 200, 90)    # green   — coverage source (intensity ∝ Σ)
            RISKCOL = (232, 86, 60)      # red     — risk-influenced (social layer)
            covers = [m for m in self.missions.values()
                      if m.get("type") == "cover" and m.get("planner")]
            if covers:
                ov = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
                for m in covers:
                    planner = m["planner"]
                    grid, risk, region = (planner.world.grid,
                                          planner._residual_risk, m["region"])
                    covered = planner._covered
                    act = planner.gbnn_coverage.activity
                    soc = planner.gbnn_social.activity
                    rbias = planner.risk_bias
                    for r in range(ROWS_B):
                        for c in range(COLS_B):
                            rect = pygame.Rect(c * GP_BOX, r * GP_BOX,
                                               GP_BOX, GP_BOX)
                            if grid[r][c] == 1:                 # obstacle (-1 node)
                                # people and static obstacles render identically
                                ov.fill((*OBSTACLE, A), rect)
                                pygame.draw.rect(ov, (80, 84, 90, 90), rect, 1)
                                continue
                            if (r, c) not in region:            # outside coverage area
                                continue
                            rk = risk[r][c]
                            if (r, c) in covered:               # covered (blue)
                                ov.fill((*COVERED, 95), rect)
                            elif rk > 0.05:                     # risk-influenced (red)
                                a = int(45 + (A - 45) * min(1.0, rk))
                                ov.fill((*RISKCOL, a), rect)
                            else:                               # uncovered (green) by Σ
                                ssum = act[r][c] + rbias * soc[r][c]
                                a = int(35 + (A - 35) * max(0.0, min(1.0, ssum)))
                                ov.fill((*UNCOVERED, a), rect)
                s.blit(ov, (0, 0))
                self._draw_field_legend(s, OBSTACLE, COVERED, UNCOVERED, RISKCOL)
                # Path-plan trace drawn ON TOP of the field, so the route the
                # robot has planned/swept is visible over the colour map (part
                # of the SHIFT view, not hidden beneath it).
                for rid, m in self.missions.items():
                    if m.get("type") != "cover":
                        continue
                    tr = m.get("trace") or []
                    if len(tr) > 1:
                        col = (self.robots[rid].color if rid in self.robots
                               else (90, 200, 255))
                        pts = [(int(a), int(b)) for a, b in tr]
                        pygame.draw.lines(s, (15, 15, 15), False, pts, 4)
                        pygame.draw.lines(s, col, False, pts, 2)
                        for px, py in pts:
                            pygame.draw.circle(s, col, (px, py), 2)

            # --- PP* layers (active point-to-point missions) -------------
            p2ps = [m for m in self.missions.values() if m.get("type") == "p2p"]
            if p2ps:
                # Predators are PROXIMITY CLUSTERS of people (not one per
                # person): people within one dominance radius merge into a
                # single predator whose zone grows with the cluster's spread and
                # whose threat scales with the head-count.
                base_r = PP_RADIUS_CELLS * GP_FINE
                preds = cluster_predators([(h.x, h.y) for h in self.fm.humans],
                                          link_dist=PP_CLUSTER_LINK_PX,
                                          base_radius=base_r,
                                          cs_per_person=PP_CS_PER_PERSON,
                                          tolerance=PP_TOLERANCE)
                for p in preds:
                    cx, cy = int(p.center[0]), int(p.center[1])
                    pygame.draw.circle(s, (235, 90, 90), (cx, cy), int(p.radius), 2)
                    pygame.draw.circle(s, (235, 90, 90), (cx, cy), 4)
                    if p.count > 1:                             # cluster size
                        img = self.font.render(f"x{p.count}", True, (255, 215, 215))
                        s.blit(img, (cx + 6, cy - 7))
                for m in p2ps:                                  # prey goal
                    gx, gy = int(m["goal"][0]), int(m["goal"][1])
                    pygame.draw.circle(s, (90, 230, 140), (gx, gy), 8, 2)
                    pygame.draw.circle(s, (90, 230, 140), (gx, gy), 3)

        def _draw_field_legend(self, s, obstacle, covered, uncovered,
                               riskcol) -> None:
            """Compact key for the summed-neural-field colour map (top-left)."""
            rows = [("Obstacle (-1)", obstacle), ("Covered", covered),
                    ("Uncovered", uncovered), ("Risk-influenced", riskcol)]
            pad, sw, lh = 8, 12, 18
            w, h = 150, pad * 2 + lh * len(rows)
            panel = pygame.Surface((w, h), pygame.SRCALPHA)
            panel.fill((18, 22, 24, 200))
            s.blit(panel, (8, 8))
            pygame.draw.rect(s, (90, 96, 100), pygame.Rect(8, 8, w, h), 1)
            for i, (label, col) in enumerate(rows):
                y = 8 + pad + i * lh
                pygame.draw.rect(s, col, pygame.Rect(16, y, sw, sw))
                pygame.draw.rect(s, (200, 206, 200), pygame.Rect(16, y, sw, sw), 1)
                s.blit(self.font.render(label, True, (224, 230, 224)),
                       (16 + sw + 8, y - 1))

        def _draw_coverage_overlays(self, s) -> None:
            """Selected coverage area outline (the swept trace is drawn only with
            the SHIFT planner overlay, see _draw_planning_overlays)."""
            for rid, m in self.missions.items():
                if m.get("type") != "cover":
                    continue
                lo_x, lo_y, hi_x, hi_y = m["rect"]
                pygame.draw.rect(s, (255, 230, 120),
                                 pygame.Rect(lo_x, lo_y, hi_x - lo_x, hi_y - lo_y), 2)

        def _draw_drag_preview(self, s) -> None:
            if not (self.dragging and self.down and self.drag_now):
                return
            if self.tool in LASSO_SPAWN:
                if len(self.lasso) >= 2:
                    pygame.draw.lines(s, (150, 196, 226), True,
                                      [(int(x), int(y)) for x, y in self.lasso], 2)
                return
            wx0, wy0 = self.down
            wx1, wy1 = self.drag_now
            rect = pygame.Rect(min(wx0, wx1), min(wy0, wy1),
                               abs(wx1 - wx0), abs(wy1 - wy0))
            col = (230, 230, 230) if self.tool in DRAG_SPAWN else SEL_DRAG
            pygame.draw.rect(s, col, rect, 2)

        def _draw_robot(self, s, rob: Robot) -> None:
            # Area-of-effect rendered as a PULSING wave of UV light expanding
            # radially out to the AoE radius (the further of UV / pheromone
            # reach). Brighter in the dark (the lamp dominates at night), but
            # always visible so the serviced radius reads in daylight too.
            uv = 1.0 - self.light
            aoe = int(AOE_PX)
            cx, cy = int(rob.x), int(rob.y)
            # Always clearly visible (it represents the live AoE), a bit stronger
            # at night when the UV lamp dominates.
            bright = 0.65 + 0.35 * uv
            t = pygame.time.get_ticks() / 1000.0
            period, n_waves = 1.4, 3
            wave = pygame.Surface((aoe * 2 + 4, aoe * 2 + 4), pygame.SRCALPHA)
            ctr = aoe + 2
            # BLUE translucent glow area filling the AoE — the area between the
            # wave bands (radial falloff via a few stacked filled discs).
            for rr, aa in ((aoe, 22), (int(aoe * 0.66), 18), (int(aoe * 0.33), 22)):
                pygame.draw.circle(wave, (60, 120, 255, int(aa * bright)),
                                   (ctr, ctr), rr)
            # PURPLE pulsing waves: thick translucent bands expanding outward and
            # fading (not thin lines).
            band = max(5, int(aoe * 0.16))
            for k in range(n_waves):
                frac = ((t / period) + k / n_waves) % 1.0      # 0→1 expanding
                rr = int(frac * aoe)
                if rr < band:
                    continue
                a = int(110 * (1.0 - frac) * bright)           # fade as it grows
                pygame.draw.circle(wave, (185, 155, 255, a), (ctr, ctr), rr, band)
            # faint steady purple boundary so the AoE extent is always readable
            pygame.draw.circle(wave, (170, 140, 255, int(50 + 50 * bright)),
                               (ctr, ctr), aoe, 1)
            s.blit(wave, (cx - ctr, cy - ctr))
            # The planned route is part of the SHIFT debug view only — hold
            # SHIFT to see the path plan (and the predator circles); otherwise
            # just the robots are shown.
            _keys = pygame.key.get_pressed()
            if rob.has_path and (_keys[pygame.K_LSHIFT] or _keys[pygame.K_RSHIFT]):
                # draw from the robot through every remaining waypoint to the
                # goal. Dark underlay + bright line for contrast.
                pts = ([(int(rob.x), int(rob.y))]
                       + [(int(x), int(y)) for (x, y) in rob.waypoints[rob.wp_i:]])
                if len(pts) > 1:
                    pygame.draw.lines(s, (30, 30, 30), False, pts, 4)
                    pygame.draw.lines(s, PATH_COL, False, pts, 2)
                    for (px, py) in pts[1:]:
                        pygame.draw.circle(s, PATH_COL, (px, py), 3)
                    gx, gy = pts[-1]
                    pygame.draw.circle(s, (90, 220, 140), (gx, gy), 5)   # goal
            poly = [(int(x), int(y)) for (x, y) in rob.polygon_world()]
            if rob.rid == self.selected:
                pygame.draw.circle(s, (255, 255, 255), (int(rob.x), int(rob.y)),
                                   int(rob.radius) + 5, 2)
            pygame.draw.polygon(s, rob.color, poly)
            pygame.draw.polygon(s, (255, 255, 255), poly, 2)
            hx = rob.x + (rob.radius + 6) * math.cos(rob.theta)
            hy = rob.y + (rob.radius + 6) * math.sin(rob.theta)
            pygame.draw.line(s, (255, 255, 255), (int(rob.x), int(rob.y)),
                             (int(hx), int(hy)), 2)
            img = self.font.render(str(rob.rid), True, (245, 245, 245))
            s.blit(img, (int(rob.x) - img.get_width() // 2, int(rob.y) - 24))

        def _draw_toolbar(self) -> None:
            pygame.draw.rect(self.screen, TOOLBAR_BG,
                             pygame.Rect(CANVAS_W, HUD_H, TOOLBAR_W, CANVAS_H))
            for i, kind in enumerate(PALETTE):
                rect = pygame.Rect(CANVAS_W + 8, HUD_H + 10 + i * 40,
                                   TOOLBAR_W - 16, 32)
                col = TOOLBAR_SEL if self.tool == kind else TOOLBAR_BTN
                pygame.draw.rect(self.screen, col, rect, border_radius=4)
                self.screen.blit(self.font.render(FEATURE_LABELS[kind], True,
                                                  HUD_TEXT), (rect.x + 8, rect.y + 9))
            # --- environment tab (time-of-day + weather, clickable dropdowns)
            y = self._panel_y()
            self.screen.blit(self.font.render("Environment", True, HUD_ACCENT),
                             (CANVAS_W + 8, y))
            rows = self._env_rows()
            for key, label in (("time", f"Time: {self.time}"),
                               ("weather", f"Weather: {self.weather}")):
                rect = rows[key]
                pygame.draw.rect(self.screen, TOOLBAR_BTN, rect, border_radius=4)
                self.screen.blit(self.font.render(label, True, HUD_TEXT),
                                 (rect.x + 6, rect.y + 8))
                self.screen.blit(self.font.render("v", True, (170, 180, 180)),
                                 (rect.right - 13, rect.y + 8))
            ty = rows["weather"].bottom + 8
            self.screen.blit(self.font.render(
                f"{self.temp:.0f}°C   light {self.light:.2f}", True, HUD_TEXT),
                (CANVAS_W + 8, ty))
            # dropdown overlay drawn last so it sits on top
            if self.dropdown is not None:
                current = self.time if self.dropdown == "time" else self.weather
                for val, rect in self._dropdown_rects():
                    sel = (val == current)
                    pygame.draw.rect(self.screen,
                                     TOOLBAR_SEL if sel else (60, 66, 68),
                                     rect, border_radius=3)
                    self.screen.blit(self.font.render(val, True, HUD_TEXT),
                                     (rect.x + 6, rect.y + 4))

        def _draw_hud(self) -> None:
            pygame.draw.rect(self.screen, HUD_BG, pygame.Rect(0, 0, WINDOW_W, HUD_H))
            self.screen.blit(self.font_b.render(
                "Park — 1-5 spawn  WASD drive  Q/E speed  "
                "click: PP*  drag: coverage", True, HUD_TEXT), (10, 8))
            self.screen.blit(self.font.render(
                f"robot: {self.selected}   speed: {self.scale:.1f}   "
                f"planner: {PLABEL[self.planner_name]}   "
                f"human-first: {self.human_first}   "
                f"tool: {FEATURE_LABELS[self.tool] if self.tool else 'none'}   "
                f"trapped: {self.captured}", True, HUD_ACCENT), (10, 38))
            counts = {sp: 0 for sp in Species}
            for m in self.fm.mosquitoes:
                counts[m.species] += 1
            self.screen.blit(self.font.render(
                "mosquitoes  " + "  ".join(
                    f"{SPECIES_LABEL[sp]}:{counts[sp]}" for sp in Species)
                + f"   total:{len(self.fm.mosquitoes)}", True, (200, 220, 210)),
                (520, 64))
            self.screen.blit(self.font.render(self.msg, True, HUD_TEXT), (10, 64))

        # ----- main loop ----------------------------------------------
        def run(self) -> int:
            dt = 1.0 / FPS
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        self.on_key(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.on_mousedown(event.pos)
                    elif event.type == pygame.MOUSEMOTION:
                        self.on_mousemotion(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.on_mouseup(event.pos)
                self.update(dt, pygame.key.get_pressed())
                self.draw()
                self.clock.tick(FPS)
            pygame.quit()
            return 0

    return Sandbox().run()


# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Ecological Computation park sandbox")
    ap.add_argument("--test", action="store_true",
                    help="run the headless self-test and exit (no display)")
    args = ap.parse_args(argv)
    if args.test:
        return _run_headless_test()
    return _run_pygame_sandbox()


if __name__ == "__main__":
    raise SystemExit(main())
