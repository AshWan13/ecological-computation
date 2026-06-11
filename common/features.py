"""
common/features.py — Park feature system for the sandbox.

Structured after ``obstacles.py`` from the sibling repositories: every feature
carries a world-frame pose + size, knows how to test containment / circular
collision, rasterises itself into an occupancy grid, and draws itself with
recognisable shapes.

Feature categories
------------------
  No-entry (rasterise to obstacles):
      BUSH, TREE (stump only), POND, EATERY, TOILET, BUILDING
  Walkable (render only, never block):
      PAVEMENT, SHELTER (an open canopy — robots/humans pass freely under it)
  Agent:
      HUMAN — random-walks each tick; exported as "human cells" for the
              planners (predator avoidance in PP*, human-first ordering in
              HFA-CCPP).  A person's cell is also treated as occupied by the
              path planners (a robot cannot drive through someone) — see
              ``build_occupancy_grid(block_humans=True)``.

A tree is built from an *accessible canopy* (large green disc the robot may
pass under) and an *inaccessible stump* (small trunk disc that blocks). Only
the stump enters the occupancy grid.

World units are pixels (1 unit = 1 px); the demo maps world->screen by adding
the HUD offset, so no separate metres scaling is needed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]


# ============================================================================
#  Feature registry
# ============================================================================

class FeatureKind(Enum):
    HUMAN    = auto()
    BUSH     = auto()
    TREE     = auto()
    SHELTER  = auto()      # rectangular drag-drawn shelter (open roof)
    PAVILION = auto()      # standalone canopy (fixed-size)
    BUILDING = auto()
    POND     = auto()
    PAVEMENT = auto()
    DRAIN    = auto()
    EATERY   = auto()
    TOILET   = auto()


# Default half-extents (half_w, half_h) in world px for fixed-size kinds.
# For TREE, half_w is the canopy radius (visual); the stump radius is separate.
FEATURE_DIMS: Dict[FeatureKind, Tuple[float, float]] = {
    FeatureKind.HUMAN:    (8.0, 8.0),
    FeatureKind.BUSH:     (24.0, 18.0),
    FeatureKind.TREE:     (34.0, 34.0),   # canopy radius (looks bigger than bush)
    FeatureKind.SHELTER:  (70.0, 50.0),   # rectangular drag-drawn roof
    FeatureKind.PAVILION: (50.0, 42.0),   # standalone fixed-size canopy
    FeatureKind.BUILDING: (78.0, 60.0),
    FeatureKind.POND:     (46.0, 32.0),
    FeatureKind.PAVEMENT: (60.0, 40.0),
    FeatureKind.DRAIN:    (16.0, 16.0),
    FeatureKind.EATERY:   (44.0, 30.0),
    FeatureKind.TOILET:   (26.0, 26.0),
}

TREE_STUMP_RADIUS: float = 8.0   # the only part of a tree that blocks

FEATURE_LABELS: Dict[FeatureKind, str] = {
    FeatureKind.HUMAN: "Human", FeatureKind.BUSH: "Bush",
    FeatureKind.TREE: "Tree", FeatureKind.SHELTER: "Shelter",
    FeatureKind.PAVILION: "Pavilion",
    FeatureKind.BUILDING: "Building", FeatureKind.POND: "Pond",
    FeatureKind.PAVEMENT: "Pavement", FeatureKind.DRAIN: "Drain",
    FeatureKind.EATERY: "Eatery", FeatureKind.TOILET: "Toilet",
}

# No-entry kinds become obstacles; everything else is traversable.
# (DRAIN is a flush ground cover — walkable, planner-neutral.)
# BUILDING is a solid, inaccessible structure. SHELTER is an OPEN canopy: it is
# NOT occupied, so robots and people move freely under it (a walkable planning
# surface, like pavement, just roofed).
NO_ENTRY = {FeatureKind.BUSH, FeatureKind.TREE, FeatureKind.POND,
            FeatureKind.EATERY, FeatureKind.TOILET, FeatureKind.BUILDING}
WALKABLE = {FeatureKind.PAVEMENT, FeatureKind.SHELTER, FeatureKind.PAVILION}

# Where people congregate: walk on these (incl. under the shelter/pavilion), or
# hang near the solid blocks (building gathers people just outside it).
HUMAN_WALK_ZONES = {FeatureKind.PAVEMENT, FeatureKind.SHELTER,
                    FeatureKind.PAVILION, FeatureKind.DRAIN}
HUMAN_NEAR_ZONES = {FeatureKind.EATERY, FeatureKind.TOILET,
                    FeatureKind.BUILDING}


# ============================================================================
#  Mosquito species (Aedes / Culex / Anopheles)
# ============================================================================
# Parameter tables grounded in published facts (see README "Sources"):
#   * Aedes — aggressive DAY biter, peaks morning & late afternoon; strongly
#     human/CO2-seeking, weakly drawn to light; breeds in clean stagnant water,
#     Ae. albopictus rests in vegetation.
#   * Culex — DUSK-to-DAWN; breeds in polluted water / blocked drains; the
#     species most readily caught by UV light traps.
#   * Anopheles — night (post-midnight peak); clean / swampy water; rare in
#     urban areas.

class Species(Enum):
    AEDES = auto()
    CULEX = auto()
    ANOPHELES = auto()


SPECIES_LABEL = {Species.AEDES: "Aedes", Species.CULEX: "Culex",
                 Species.ANOPHELES: "Anopheles"}

# Host-seeking / flight activity by time of day (0..1).
SPECIES_ACTIVITY = {
    Species.AEDES:     {"Dawn": 1.00, "Day": 0.70, "Dusk": 0.90, "Night": 0.20},
    Species.CULEX:     {"Dawn": 0.50, "Day": 0.20, "Dusk": 1.00, "Night": 1.00},
    Species.ANOPHELES: {"Dawn": 0.60, "Day": 0.10, "Dusk": 0.70, "Night": 1.00},
}
# Flight-activity multiplier by weather (rain/wind suppress flight).
SPECIES_WEATHER = {"Clear": 1.00, "Rain": 0.60, "Windy": 0.70}

# Responsiveness to each attractor cue type (multiplies cue strength).
SPECIES_CUE = {
    Species.AEDES:     {"human": 1.00, "uv": 0.30, "harborage": 1.00},
    Species.CULEX:     {"human": 0.70, "uv": 1.00, "harborage": 0.70},
    Species.ANOPHELES: {"human": 0.80, "uv": 0.70, "harborage": 0.60},
}
# Per-feature resting/dwell (linger) probability by species.
SPECIES_DWELL = {
    Species.AEDES:     {FeatureKind.BUSH: 0.90, FeatureKind.SHELTER: 0.60,
                        FeatureKind.BUILDING: 0.40, FeatureKind.DRAIN: 0.50,
                        FeatureKind.POND: 0.20},
    Species.CULEX:     {FeatureKind.DRAIN: 0.90, FeatureKind.BUSH: 0.60,
                        FeatureKind.SHELTER: 0.50, FeatureKind.BUILDING: 0.40,
                        FeatureKind.POND: 0.30},
    Species.ANOPHELES: {FeatureKind.BUSH: 0.85, FeatureKind.POND: 0.50,
                        FeatureKind.SHELTER: 0.40, FeatureKind.BUILDING: 0.30,
                        FeatureKind.DRAIN: 0.30},
}
# Default urban composition for ambient / initial spawns. In urban tropical
# settings Culex (drains) and Aedes (containers) dominate; Anopheles is rare.
SPECIES_BASE_MIX = {Species.CULEX: 0.50, Species.AEDES: 0.45,
                    Species.ANOPHELES: 0.05}
# Breeding-source -> species: bush/containers -> Aedes; drains -> Culex;
# open water bodies (clean-ish) -> Anopheles + organic-tolerant Culex
# (Aedes breeds in containers, not open ponds, so it is absent here).
POND_SPECIES_MIX = {Species.ANOPHELES: 0.55, Species.CULEX: 0.45}


def _temp_activity(temp_c: float) -> float:
    """Activity factor vs temperature — bell peaking ~26 °C (Culex/Aedes
    abundance peaks 23–26 °C, falling at extremes)."""
    return math.exp(-((temp_c - 26.0) / 7.0) ** 2)


def species_activity(species, tod: str, weather: str, temp_c: float) -> float:
    """Combined 0..1 activity for a species under the environment."""
    return (SPECIES_ACTIVITY[species][tod] * SPECIES_WEATHER[weather]
            * _temp_activity(temp_c))


def _weighted_choice(weights: dict):
    r = random.random() * sum(weights.values())
    acc = 0.0
    last = None
    for k, w in weights.items():
        last = k
        acc += w
        if r <= acc:
            return k
    return last


def _pick_weighted(items, weights):
    """Pick one item from a list, proportional to ``weights`` (no hashing)."""
    total = sum(weights)
    if total <= 0:
        return items[-1]
    r = random.random() * total
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]

# Kinds sized by rectangular click-drag (others, incl. the standalone PAVILION
# canopy, spawn at a fixed size on a single click).
DRAG_SPAWN = {FeatureKind.BUSH, FeatureKind.SHELTER,
              FeatureKind.BUILDING, FeatureKind.PAVEMENT}
# Kinds drawn as a freeform lasso (collect the cursor path, close on release).
LASSO_SPAWN = {FeatureKind.POND}

# Toolbar order.
PALETTE: List[FeatureKind] = [
    FeatureKind.HUMAN, FeatureKind.BUSH, FeatureKind.TREE,
    FeatureKind.SHELTER, FeatureKind.PAVILION, FeatureKind.BUILDING,
    FeatureKind.PAVEMENT, FeatureKind.DRAIN, FeatureKind.POND,
    FeatureKind.EATERY, FeatureKind.TOILET,
]


def _rect_circle_hit(dx: float, dy: float, hw: float, hh: float,
                     r: float) -> bool:
    """True if a circle of radius r centred (dx, dy) from a rect centre
    overlaps the axis-aligned rect of half-extents (hw, hh)."""
    nx = max(-hw, min(hw, dx))
    ny = max(-hh, min(hh, dy))
    return (dx - nx) ** 2 + (dy - ny) ** 2 <= r * r


def _point_in_poly(px: float, py: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _seg_point_dist(px: float, py: float, ax: float, ay: float,
                    bx: float, by: float) -> float:
    """Distance from point (px,py) to segment a-b."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _poly_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(poly)
    return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)


def _poly_area(poly: List[Tuple[float, float]]) -> float:
    """Absolute polygon area (shoelace)."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


# ============================================================================
#  Feature
# ============================================================================

_next_uid = 0


def _gen_uid() -> int:
    global _next_uid
    _next_uid += 1
    return _next_uid


@dataclass
class Feature:
    kind: FeatureKind
    x: float
    y: float
    half_w: float
    half_h: float
    uid: int = field(default_factory=_gen_uid)
    # Ponds are freeform polygons drawn lasso-style; absolute world points.
    poly: Optional[List[Tuple[float, float]]] = None

    @property
    def no_entry(self) -> bool:
        return self.kind in NO_ENTRY

    @property
    def label(self) -> str:
        return FEATURE_LABELS[self.kind]

    # ---- geometry ----------------------------------------------------
    def contains(self, px: float, py: float) -> bool:
        """Point-in-footprint test (used for despawn hit-testing)."""
        if self.poly is not None:
            return _point_in_poly(px, py, self.poly)
        dx, dy = px - self.x, py - self.y
        if self.kind == FeatureKind.TREE:          # whole canopy is clickable
            return dx * dx + dy * dy <= self.half_w ** 2
        return abs(dx) <= self.half_w and abs(dy) <= self.half_h

    def blocks_circle(self, cx: float, cy: float, r: float) -> bool:
        """True if a robot circle (cx, cy, r) collides with this feature.

        Only no-entry features block. A tree blocks via its stump disc only;
        a pond blocks anywhere inside its polygon (or within r of an edge).
        """
        if not self.no_entry:
            return False
        if self.poly is not None:
            if _point_in_poly(cx, cy, self.poly):
                return True
            n = len(self.poly)
            for i in range(n):
                ax, ay = self.poly[i]
                bx, by = self.poly[(i + 1) % n]
                if _seg_point_dist(cx, cy, ax, ay, bx, by) <= r:
                    return True
            return False
        dx, dy = cx - self.x, cy - self.y
        if self.kind == FeatureKind.TREE:
            rad = TREE_STUMP_RADIUS + r
            return dx * dx + dy * dy <= rad * rad
        return _rect_circle_hit(dx, dy, self.half_w, self.half_h, r)

    def occupies_point(self, px: float, py: float) -> bool:
        """True if a grid-cell centre (px, py) falls in the blocking footprint."""
        if not self.no_entry:
            return False
        if self.poly is not None:
            return _point_in_poly(px, py, self.poly)
        dx, dy = px - self.x, py - self.y
        if self.kind == FeatureKind.TREE:
            return dx * dx + dy * dy <= TREE_STUMP_RADIUS ** 2
        return abs(dx) <= self.half_w and abs(dy) <= self.half_h


@dataclass
class Human:
    hid: int
    x: float
    y: float
    _tx: float = 0.0
    _ty: float = 0.0
    _timer: int = 0


@dataclass
class Mosquito:
    """A single mosquito. ``species`` drives its time-of-day activity, cue
    responsiveness, and habitat preference (see the SPECIES_* tables)."""
    x: float
    y: float
    white: bool = False   # rendered black or white for contrast on any backdrop
    species: "Species" = Species.AEDES


# ============================================================================
#  Feature Manager
# ============================================================================

class FeatureManager:
    """Owns placed features + human agents; rasterises to the occupancy grid."""

    def __init__(self) -> None:
        self.features: Dict[int, Feature] = {}
        self.humans: List[Human] = []
        self.mosquitoes: List[Mosquito] = []
        self._next_hid = 1
        # fractional mosquito-spawn budgets per breeding source
        self._acc = {"bush": 0.0, "pond": 0.0, "drain": 0.0}

    # ---- spawn / despawn --------------------------------------------
    def spawn(self, kind: FeatureKind, x: float, y: float,
              half_w: Optional[float] = None,
              half_h: Optional[float] = None):
        if kind == FeatureKind.HUMAN:
            h = Human(self._next_hid, x, y, _tx=x, _ty=y,
                      _timer=random.randint(20, 60))
            self._next_hid += 1
            self.humans.append(h)
            return h
        dw, dh = FEATURE_DIMS[kind]
        f = Feature(kind, x, y,
                    dw if half_w is None else max(half_w, 8.0),
                    dh if half_h is None else max(half_h, 8.0))
        self.features[f.uid] = f
        return f

    def spawn_poly(self, kind: FeatureKind, points: List[Tuple[float, float]]):
        """Spawn a freeform polygon feature (e.g. a lasso-drawn pond)."""
        if len(points) < 3:
            return None
        cx, cy = _poly_centroid(points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        f = Feature(kind, cx, cy,
                    max((max(xs) - min(xs)) / 2, 8.0),
                    max((max(ys) - min(ys)) / 2, 8.0),
                    poly=[(float(px), float(py)) for px, py in points])
        self.features[f.uid] = f
        return f

    def despawn_at(self, px: float, py: float) -> bool:
        for h in list(self.humans):
            if math.hypot(px - h.x, py - h.y) <= FEATURE_DIMS[FeatureKind.HUMAN][0] + 3:
                self.humans.remove(h)
                return True
        for uid in sorted(self.features, reverse=True):
            if self.features[uid].contains(px, py):
                del self.features[uid]
                return True
        return False

    def feature_at(self, px: float, py: float) -> Optional[Feature]:
        """Topmost placed feature whose footprint contains (px, py), else None.
        Used to pick a feature up for moving (does not remove it)."""
        for uid in sorted(self.features, reverse=True):
            if self.features[uid].contains(px, py):
                return self.features[uid]
        return None

    def human_at(self, px: float, py: float) -> Optional["Human"]:
        """The person under (px, py), if any."""
        for h in self.humans:
            if math.hypot(px - h.x, py - h.y) <= FEATURE_DIMS[FeatureKind.HUMAN][0] + 3:
                return h
        return None

    def move_feature(self, f: Feature, nx: float, ny: float) -> None:
        """Reposition feature ``f`` so its centre is at (nx, ny), translating its
        polygon points too (for lasso-drawn ponds)."""
        dx, dy = nx - f.x, ny - f.y
        f.x, f.y = nx, ny
        if f.poly is not None:
            f.poly = [(px + dx, py + dy) for px, py in f.poly]

    def clear(self) -> None:
        self.features.clear()
        self.humans.clear()

    # ---- collision ---------------------------------------------------
    def blocks_circle(self, cx: float, cy: float, r: float,
                      exclude_human: Optional["Human"] = None) -> bool:
        """True if a circle (cx, cy, r) collides with any no-entry feature OR
        any live person. A person physically occupies their position, so the
        robot (teleop or planned) and other people cannot move into it.
        ``exclude_human`` skips one person (used so a walker never blocks
        itself)."""
        if any(f.blocks_circle(cx, cy, r) for f in self.features.values()):
            return True
        hr = FEATURE_DIMS[FeatureKind.HUMAN][0]
        for h in self.humans:
            if h is exclude_human:
                continue
            if (cx - h.x) ** 2 + (cy - h.y) ** 2 <= (hr + r) ** 2:
                return True
        return False

    # ---- mosquito swarm ----------------------------------------------
    def spawn_mosquitoes(self, n: int,
                         bounds: Tuple[float, float, float, float]) -> None:
        """Seed ``n`` mosquitoes at random, species drawn from the urban base
        composition (Culex/Aedes common, Anopheles rare)."""
        x0, y0, x1, y1 = bounds
        self.mosquitoes = [
            Mosquito(random.uniform(x0, x1), random.uniform(y0, y1),
                     white=(i % 2 == 0), species=_weighted_choice(SPECIES_BASE_MIX))
            for i in range(n)
        ]

    # 8-neighbour steps (no "stay") for directed lure motion.
    _DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1),
             (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def _field_at(self, px: float, py: float, attractors, species) -> float:
        """Species-weighted exponential attraction field at (px, py).

        ``attractors`` entries are ``(x, y, cues)`` with cues
        ``(lambda_px, strength, cue_type)``; cue_type in {human, uv, harborage}
        scales by the species' responsiveness (SPECIES_CUE)."""
        gain = SPECIES_CUE[species]
        phi = 0.0
        for (ax, ay, cues) in attractors:
            d = math.hypot(px - ax, py - ay)
            for (lam, s, ctype) in cues:
                g = s * gain.get(ctype, 1.0)
                if g > 0.0:
                    phi += g * math.exp(-d / lam)
        return phi

    def _habitat_dwell(self, px: float, py: float, species) -> float:
        """Per-tick linger probability for the resting habitat under (px, py),
        for this species."""
        pref = SPECIES_DWELL[species]
        best = 0.0
        for f in self.features.values():
            if f.kind in pref and f.contains(px, py):
                best = max(best, pref[f.kind])
        return best

    def tick_mosquitoes(self, bounds: Tuple[float, float, float, float],
                        attractors, activity: dict, lure_cap: float = 0.10,
                        K: float = 2.0, step: float = 2.5,
                        jitter: float = 1.2) -> None:
        """Probabilistic, species-aware mosquito motion (not a magnet).

        Per mosquito each tick:
          0. *Activity* — with probability (1 - species_activity) it is at rest
             for this time-of-day/weather and barely moves (e.g. Aedes at night,
             Culex by day).
          1. *Dwell* — if over a preferred resting habitat it lingers.
          2. *Lure* — otherwise drawn by the (species-weighted) field with
             probability  p = lure_cap * Phi/(Phi+K)  (0% with no cue nearby,
             capped at lure_cap); on success steps one cell up-gradient.
          3. *Wander* — else a random hover step.

        ``activity`` maps Species -> 0..1 for the current environment.
        """
        x0, y0, x1, y1 = bounds
        for m in self.mosquitoes:
            # 0) inactive now -> rest in place (tiny jitter)
            if random.random() > activity.get(m.species, 1.0):
                m.x = max(x0, min(x1, m.x + random.uniform(-0.5, 0.5)))
                m.y = max(y0, min(y1, m.y + random.uniform(-0.5, 0.5)))
                continue
            # 1) habitat dwell
            d = self._habitat_dwell(m.x, m.y, m.species)
            if d > 0.0 and random.random() < d:
                m.x = max(x0, min(x1, m.x + random.uniform(-jitter, jitter)))
                m.y = max(y0, min(y1, m.y + random.uniform(-jitter, jitter)))
                continue
            # 2) probabilistic lure (capped)
            phi = self._field_at(m.x, m.y, attractors, m.species)
            p_lure = lure_cap * (phi / (phi + K)) if phi > 0.0 else 0.0
            if random.random() < p_lure:
                best, best_phi = None, -1.0
                for dx, dy in self._DIRS:
                    px, py = m.x + dx * step, m.y + dy * step
                    if not (x0 <= px <= x1 and y0 <= py <= y1):
                        continue
                    f = self._field_at(px, py, attractors, m.species)
                    if f > best_phi:
                        best_phi, best = f, (dx, dy)
                if best is not None:
                    m.x = max(x0, min(x1, m.x + best[0] * step))
                    m.y = max(y0, min(y1, m.y + best[1] * step))
                continue
            # 3) wander
            dx, dy = random.choice(self._DIRS)
            m.x = max(x0, min(x1, m.x + dx * step + random.uniform(-jitter, jitter)))
            m.y = max(y0, min(y1, m.y + dy * step + random.uniform(-jitter, jitter)))

    def trap(self, robots_xy, radius: float, catch_prob: float = 0.05) -> int:
        """Lure-and-catch sink: a mosquito on the robot is captured with only
        ``catch_prob`` (5%) probability per tick — luring is unaffected, so
        many mosquitoes hover on the robot before one is actually caught.
        Returns the number captured this tick."""
        if not robots_xy:
            return 0
        r2 = radius * radius
        survivors, trapped = [], 0
        for m in self.mosquitoes:
            on = any((m.x - rx) ** 2 + (m.y - ry) ** 2 <= r2 for rx, ry in robots_xy)
            if on and random.random() < catch_prob:
                trapped += 1
            else:
                survivors.append(m)
        self.mosquitoes = survivors
        return trapped

    def _rand_in_poly(self, poly, tries: int = 20):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        for _ in range(tries):
            px = random.uniform(min(xs), max(xs))
            py = random.uniform(min(ys), max(ys))
            if _point_in_poly(px, py, poly):
                return (px, py)
        return _poly_centroid(poly)

    def _add_mosquito(self, x, y, species) -> None:
        self.mosquitoes.append(
            Mosquito(x, y, white=(len(self.mosquitoes) % 2 == 0), species=species))

    def spawn_from_features(self, px_per_m: float, fps: float,
                            breeding_mult: float, cap: int) -> int:
        """Breed mosquitoes from features at grounded, low rates:

          * Bush  — 1 mosquito per 10 m² of bush per 2 min  -> Aedes (vegetation)
          * Pond  — 3 mosquitoes per water body per 5 min    -> Aedes / Anopheles
          * Drain — 1 mosquito per drain per 5 min           -> Culex (polluted)

        ``breeding_mult`` boosts rates in wet weather. Budgets accumulate
        fractionally; spawns land at the source, capped at ``cap`` total.
        """
        if len(self.mosquitoes) >= cap:
            return 0
        feats = self.features.values()
        bushes = [f for f in feats if f.kind == FeatureKind.BUSH]
        ponds = [f for f in feats if f.kind == FeatureKind.POND and f.poly]
        drains = [f for f in feats if f.kind == FeatureKind.DRAIN]

        m2 = px_per_m * px_per_m
        bush_units = sum((4 * f.half_w * f.half_h) / m2 for f in bushes) / 10.0
        self._acc["bush"] += breeding_mult * bush_units / (120.0 * fps)
        self._acc["pond"] += breeding_mult * len(ponds) * 3.0 / (300.0 * fps)
        self._acc["drain"] += breeding_mult * len(drains) * 1.0 / (300.0 * fps)

        eps = 1e-6   # guard against float accumulation landing just under 1
        spawned = 0
        while self._acc["bush"] >= 1.0 - eps and bushes and len(self.mosquitoes) < cap:
            self._acc["bush"] -= 1.0
            b = _pick_weighted(bushes, [4 * f.half_w * f.half_h for f in bushes])
            self._add_mosquito(b.x + random.uniform(-b.half_w, b.half_w),
                               b.y + random.uniform(-b.half_h, b.half_h),
                               Species.AEDES)
            spawned += 1
        while self._acc["pond"] >= 1.0 - eps and ponds and len(self.mosquitoes) < cap:
            self._acc["pond"] -= 1.0
            p = _pick_weighted(ponds, [_poly_area(f.poly) for f in ponds])
            pt = self._rand_in_poly(p.poly)
            self._add_mosquito(pt[0], pt[1], _weighted_choice(POND_SPECIES_MIX))
            spawned += 1
        while self._acc["drain"] >= 1.0 - eps and drains and len(self.mosquitoes) < cap:
            self._acc["drain"] -= 1.0
            d = random.choice(drains)
            self._add_mosquito(d.x, d.y, Species.CULEX)
            spawned += 1
        return spawned

    # ---- dynamics ----------------------------------------------------
    def _pick_human_target(self, h, bounds, grass_prob: float = 0.05,
                           stay_prob: float = 0.90):
        """Pick a walk target. If the person is on a walk-zone they usually
        wander locally within it (so they *stay* on pavements/shelters); when
        not, they head to a zone (95%) or, rarely (5%), out onto grass."""
        x0, y0, x1, y1 = bounds
        walk = [f for f in self.features.values() if f.kind in HUMAN_WALK_ZONES]
        near = [f for f in self.features.values() if f.kind in HUMAN_NEAR_ZONES]
        cur = next((f for f in walk if f.contains(h.x, h.y)), None)
        r = random.random()
        if cur is not None and r < stay_prob:                # linger on zone
            return (max(x0, min(x1, cur.x + random.uniform(-0.8, 0.8) * cur.half_w)),
                    max(y0, min(y1, cur.y + random.uniform(-0.8, 0.8) * cur.half_h)))
        if (walk or near) and r < (1.0 - grass_prob):
            f = random.choice(walk + near)
            if f.kind in HUMAN_NEAR_ZONES:                   # stand just outside
                ang = random.uniform(0, 2 * math.pi)
                rad = max(f.half_w, f.half_h) + random.uniform(8, 24)
                return (max(x0, min(x1, f.x + math.cos(ang) * rad)),
                        max(y0, min(y1, f.y + math.sin(ang) * rad)))
            return (max(x0, min(x1, f.x + random.uniform(-f.half_w, f.half_w))),
                    max(y0, min(y1, f.y + random.uniform(-f.half_h, f.half_h))))
        return (random.uniform(x0, x1), random.uniform(y0, y1))   # grass

    def tick_humans(self, bounds: Tuple[float, float, float, float],
                    speed: float = 0.65) -> None:
        """Walk toward a preferred zone target (re-picked on arrival/timeout)."""
        x0, y0, x1, y1 = bounds
        for h in self.humans:
            h._timer -= 1
            if h._timer <= 0:
                h._tx, h._ty = self._pick_human_target(h, bounds)
                h._timer = random.randint(120, 320)
            dx, dy = h._tx - h.x, h._ty - h.y
            dist = math.hypot(dx, dy)
            if dist < 1.5:
                h._timer = 0
                continue
            step = min(speed, dist)
            nx, ny = h.x + dx / dist * step, h.y + dy / dist * step
            if not self.blocks_circle(nx, ny, 4.0, exclude_human=h):
                h.x, h.y = nx, ny
            else:
                h._timer = 0

    # ---- rasterisation ----------------------------------------------
    def build_occupancy_grid(self, rows: int, cols: int, gp: float,
                             inflate: int = 0,
                             plannable_only: bool = False,
                             block_humans: bool = False) -> List[List[int]]:
        """Occupancy grid (0 free, 1 obstacle) at pitch ``gp`` px.

        Two traversability models:

        * ``plannable_only=False`` (teleop model) — a cell is an obstacle only
          if its centre lies inside a no-entry footprint; open grass is free.
        * ``plannable_only=True`` (planning model) — autonomous planners may
          only travel on *walkable* surfaces (pavement, shelter), so a cell is
          free **only** when it lies on a walkable feature and is not
          obstructed. Open grass is non-plannable. ``inflate`` adds clearance
          from EVERY non-plannable cell — obstacles AND the grass boundary — so
          autonomous paths keep a margin from grass edges too.

        ``block_humans=True`` additionally stamps each person's current cell as
        occupied, so the planners route around people (a robot cannot drive
        through someone). Applied last, so it also blocks a walkable cell that
        a person happens to be standing on.
        """
        obst = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            cy = (r + 0.5) * gp
            for c in range(cols):
                cx = (c + 0.5) * gp
                for f in self.features.values():
                    if f.occupies_point(cx, cy):
                        obst[r][c] = 1
                        break

        if plannable_only:
            grid = [[1] * cols for _ in range(rows)]
            for r in range(rows):
                cy = (r + 0.5) * gp
                for c in range(cols):
                    if obst[r][c]:
                        continue
                    cx = (c + 0.5) * gp
                    for f in self.features.values():
                        if f.kind in WALKABLE and f.contains(cx, cy):
                            grid[r][c] = 0          # plannable paved surface
                            break
            if inflate > 0:
                # Inflate ALL non-plannable cells (obstacles + grass) into the
                # plannable region, so paths keep ``inflate`` cells of clearance
                # from grass edges as well as from no-entry features.
                grid = self._dilate(grid, rows, cols, inflate)
            result = grid
        else:
            if inflate > 0:
                obst = self._dilate(obst, rows, cols, inflate)
            result = obst

        if block_humans:
            for h in self.humans:
                r, c = int(h.y // gp), int(h.x // gp)
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = 1               # a person's cell is occupied
        return result

    @staticmethod
    def _dilate(grid, rows, cols, k):
        out = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    for dr in range(-k, k + 1):
                        for dc in range(-k, k + 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < rows and 0 <= cc < cols:
                                out[rr][cc] = 1
        return out

    def human_cells(self, rows: int, cols: int, gp: float) -> List[Cell]:
        cells = []
        for h in self.humans:
            rc = (int(h.y // gp), int(h.x // gp))
            if 0 <= rc[0] < rows and 0 <= rc[1] < cols:
                cells.append(rc)
        return cells

    def human_cells_id(self, rows: int, cols: int, gp: float):
        """Live people as ``(id, row, col)`` — identity lets the coverage
        planner carry each person's residual risk as they move."""
        out = []
        for h in self.humans:
            r, c = int(h.y // gp), int(h.x // gp)
            if 0 <= r < rows and 0 <= c < cols:
                out.append((h.hid, r, c))
        return out


# ============================================================================
#  Rendering
# ============================================================================

def draw_feature(screen, f: Feature, oy: int, font) -> None:
    """Draw one feature. ``oy`` is the world->screen y offset (HUD height)."""
    import pygame
    x, y = int(f.x), int(f.y + oy)
    hw, hh = int(f.half_w), int(f.half_h)
    k = f.kind

    if k == FeatureKind.POND and f.poly is not None:
        _draw_water(screen, f, oy)
        return

    if k == FeatureKind.PAVEMENT:
        rect = pygame.Rect(x - hw, y - hh, 2 * hw, 2 * hh)
        pygame.draw.rect(screen, (172, 170, 162), rect, border_radius=3)
        for sx in range(x - hw + 12, x + hw, 24):       # paving seams
            pygame.draw.line(screen, (150, 148, 140), (sx, y - hh), (sx, y + hh), 1)
        for sy in range(y - hh + 12, y + hh, 24):
            pygame.draw.line(screen, (150, 148, 140), (x - hw, sy), (x + hw, sy), 1)

    elif k == FeatureKind.POND:
        rect = pygame.Rect(x - hw, y - hh, 2 * hw, 2 * hh)
        pygame.draw.ellipse(screen, (62, 118, 188), rect)            # no outline
        hi = pygame.Rect(x - int(hw * 0.6), y - int(hh * 0.6),
                         int(hw * 0.7), int(hh * 0.5))
        pygame.draw.ellipse(screen, (104, 162, 214), hi)

    elif k == FeatureKind.BUSH:
        for bx in range(x - hw, x + hw, 13):
            for by in range(y - hh, y + hh, 13):
                pygame.draw.circle(screen, (48, 96, 46), (bx + 7, by + 7), 9)
        for bx in range(x - hw + 6, x + hw, 16):
            pygame.draw.circle(screen, (60, 112, 56), (bx, y), 7)

    elif k == FeatureKind.TREE:
        # Top-down tree with an ORGANIC, lumpy crown (not a clean circle): the
        # canopy is the union of leaf masses placed round a jittered ring, so its
        # silhouette is irregular. The trunk is buried under the foliage — only a
        # small sliver peeks out at the bottom. Deterministic per tree (no flicker).
        R = max(8, hw)
        rng = random.Random(f.uid)
        greens_dark = [(34, 70, 36), (40, 82, 42)]
        greens_mid = [(50, 100, 52), (62, 118, 62)]
        green_hi = (84, 144, 86)
        # leaf masses: a central blob + a jittered ring of off-centre blobs.
        masses = [(x, y, int(rng.uniform(0.46, 0.58) * R))]
        nb = rng.randint(7, 9)
        for i in range(nb):
            ang = 2 * math.pi * i / nb + rng.uniform(-0.35, 0.35)
            dist = rng.uniform(0.34, 0.56) * R
            mr = int(rng.uniform(0.42, 0.60) * R)
            masses.append((x + int(math.cos(ang) * dist),
                           y + int(math.sin(ang) * dist), mr))
        # soft ground shadow (same lumpy shape, offset down-right)
        pad = R + 14
        sh = pygame.Surface((2 * pad, 2 * pad), pygame.SRCALPHA)
        for (mx, my, mr) in masses:
            pygame.draw.circle(sh, (0, 0, 0, 26), (mx - x + pad, my - y + pad), mr)
        screen.blit(sh, (x - pad + 6, y - pad + 7))
        # trunk FIRST so the canopy buries it; only the bottom sliver shows.
        tr = int(TREE_STUMP_RADIUS)
        pygame.draw.ellipse(screen, (96, 68, 42),
                            pygame.Rect(x - tr + 1, y + R - int(tr * 1.2),
                                        2 * tr - 2, 2 * tr))
        pygame.draw.ellipse(screen, (74, 52, 32),
                            pygame.Rect(x - tr + 1, y + R - int(tr * 1.2),
                                        2 * tr - 2, 2 * tr), 1)
        # crown: dark base masses, dappled mid foliage, then upper highlights.
        for (mx, my, mr) in masses:
            pygame.draw.circle(screen, rng.choice(greens_dark), (mx, my), mr)
        for _ in range(16):
            ang = rng.uniform(0, 2 * math.pi)
            rad = rng.uniform(0.10, 0.70) * R
            bx, by = x + int(math.cos(ang) * rad), y + int(math.sin(ang) * rad)
            pygame.draw.circle(screen, rng.choice(greens_mid), (bx, by),
                               rng.randint(int(R * 0.22), int(R * 0.40)))
        for _ in range(9):
            ang = rng.uniform(math.pi, 2 * math.pi)                  # sin<0 → upper crown
            rad = rng.uniform(0.10, 0.58) * R
            bx, by = x + int(math.cos(ang) * rad), y + int(math.sin(ang) * rad)
            pygame.draw.circle(screen, green_hi, (bx, by),
                               rng.randint(int(R * 0.12), int(R * 0.22)))

    elif k == FeatureKind.EATERY:
        # hawker food stall: ground shadow, two-tone wall, glowing serving
        # window, wooden counter with stools, pitched roof + ridge, scalloped
        # awning, and a lit "FOOD" signboard.
        sh = pygame.Surface((2 * hw + 12, 14), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 55), sh.get_rect())
        screen.blit(sh, (x - hw - 6, y + hh - 6))
        roof_h = max(14, int(hh * 0.5))
        wall = pygame.Rect(x - hw, y - hh + roof_h, 2 * hw, 2 * hh - roof_h)
        pygame.draw.rect(screen, (234, 212, 178), wall)
        pygame.draw.rect(screen, (210, 186, 150),
                         pygame.Rect(wall.x, wall.centery, wall.w, wall.h // 2))
        win = pygame.Rect(x - int(hw * 0.74), y - int(hh * 0.02),
                          int(hw * 1.48), int(hh * 0.5))
        pygame.draw.rect(screen, (66, 56, 48), win)
        pygame.draw.rect(screen, (250, 224, 168), win.inflate(-6, -6))   # warm glow
        pygame.draw.rect(screen, (118, 102, 84), win, 2)
        counter = pygame.Rect(x - hw, y + hh - 10, 2 * hw, 10)           # counter
        pygame.draw.rect(screen, (128, 88, 52), counter)
        pygame.draw.rect(screen, (96, 64, 38), counter, 1)
        for sxp in (x - int(hw * 0.55), x, x + int(hw * 0.55)):          # stools
            pygame.draw.circle(screen, (84, 84, 90), (sxp, y + hh + 3), 3)
        pygame.draw.rect(screen, (150, 130, 100), wall, 2)
        roof = [(x - hw - 4, y - hh + roof_h), (x + hw + 4, y - hh + roof_h),
                (x + int(hw * 0.5), y - hh), (x - int(hw * 0.5), y - hh)]
        pygame.draw.polygon(screen, (170, 68, 54), roof)                 # roof
        pygame.draw.polygon(screen, (120, 44, 36), roof, 2)
        pygame.draw.line(screen, (214, 124, 104),
                         (x - int(hw * 0.5), y - hh), (x + int(hw * 0.5), y - hh), 2)
        for i, sxp in enumerate(range(x - hw, x + hw, 11)):             # awning
            w = min(11, x + hw - sxp)
            col = (236, 230, 216) if i % 2 == 0 else (208, 76, 60)
            ey = y - hh + roof_h
            pygame.draw.rect(screen, col, pygame.Rect(sxp, ey, w, 6))
            pygame.draw.polygon(screen, col, [(sxp, ey + 6), (sxp + w, ey + 6),
                                              (sxp + w / 2, ey + 11)])
        sign = pygame.Rect(x - 17, y - hh - 13, 34, 12)                 # signboard
        pygame.draw.rect(screen, (54, 44, 38), sign, border_radius=2)
        pygame.draw.rect(screen, (250, 224, 168), sign, 1, border_radius=2)
        _label(screen, font, "FOOD", x, y - hh - 7)

    elif k == FeatureKind.TOILET:
        # restroom block: ground shadow, two-tone wall, overhanging roof slab,
        # framed door with panel + knob, a vent, and a blue restroom sign.
        sh = pygame.Surface((2 * hw + 10, 12), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 55), sh.get_rect())
        screen.blit(sh, (x - hw - 5, y + hh - 5))
        roof_h = max(8, int(hh * 0.32))
        wall = pygame.Rect(x - hw, y - hh + roof_h, 2 * hw, 2 * hh - roof_h)
        pygame.draw.rect(screen, (208, 210, 214), wall)
        pygame.draw.rect(screen, (190, 193, 198),
                         pygame.Rect(wall.x, wall.centery, wall.w, wall.h // 2))
        pygame.draw.rect(screen, (150, 154, 160), wall, 2)
        pygame.draw.rect(screen, (122, 128, 136),                        # roof slab
                         pygame.Rect(x - hw - 3, y - hh, 2 * hw + 6, roof_h))
        pygame.draw.rect(screen, (92, 98, 106),
                         pygame.Rect(x - hw - 3, y - hh, 2 * hw + 6, roof_h), 1)
        dw = max(8, int(hw * 0.5))
        door = pygame.Rect(x - dw, y - hh + roof_h + 4, 2 * dw, 2 * hh - roof_h - 7)
        pygame.draw.rect(screen, (70, 90, 120), door.inflate(4, 4))      # frame
        pygame.draw.rect(screen, (98, 106, 118), door)
        pygame.draw.rect(screen, (122, 130, 142), door.inflate(-6, -10)) # panel
        pygame.draw.circle(screen, (232, 224, 158), (door.right - 4, door.centery), 2)
        pygame.draw.rect(screen, (150, 180, 200),                        # vent
                         pygame.Rect(x + int(hw * 0.55), y + int(hh * 0.1), 6, 6))
        # inclusive restroom sign above the block: white male figure on a blue
        # half, white female figure on a red half.
        plq = pygame.Rect(x - 14, y - hh - 14, 28, 13)
        half = plq.w // 2
        blue_h = pygame.Rect(plq.x, plq.y, half, plq.h)
        red_h = pygame.Rect(plq.x + half, plq.y, plq.w - half, plq.h)
        pygame.draw.rect(screen, (48, 104, 176), blue_h, border_radius=2)
        pygame.draw.rect(screen, (196, 64, 64), red_h, border_radius=2)
        white = (238, 240, 246)
        mx = blue_h.centerx                                              # male (blue bg)
        pygame.draw.circle(screen, white, (mx, plq.y + 3), 2)
        pygame.draw.rect(screen, white, pygame.Rect(mx - 2, plq.y + 5, 4, 5))
        fx = red_h.centerx                                              # female (red bg)
        pygame.draw.circle(screen, white, (fx, plq.y + 3), 2)
        pygame.draw.polygon(screen, white, [(fx, plq.y + 5),
                                            (fx - 3, plq.y + 10),
                                            (fx + 3, plq.y + 10)])

    elif k == FeatureKind.DRAIN:
        rect = pygame.Rect(x - hw, y - hh, 2 * hw, 2 * hh)
        pygame.draw.rect(screen, (96, 98, 100), rect, border_radius=2)
        pygame.draw.rect(screen, (66, 68, 70), rect, 2, border_radius=2)
        for sx in range(x - hw + 3, x + hw - 2, 4):     # grate slots
            pygame.draw.line(screen, (52, 54, 56), (sx, y - hh + 3),
                             (sx, y + hh - 3), 1)

    elif k == FeatureKind.BUILDING:
        # A solid, inaccessible building (opaque) — nothing can enter.
        # Ground shadow, two-tone wall, lit windows, a door, and an overhanging
        # roof slab.
        wall_hi, wall_lo = (176, 156, 130), (150, 132, 110)
        roof_c, edge = (98, 80, 66), (70, 56, 46)
        sh = pygame.Surface((2 * hw + 14, 16), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 70), sh.get_rect())
        screen.blit(sh, (x - hw - 7, y + hh - 7))
        roof_h = max(12, int(hh * 0.30))
        wall = pygame.Rect(x - hw, y - hh + roof_h, 2 * hw, 2 * hh - roof_h)
        pygame.draw.rect(screen, wall_hi, wall)                       # upper wall
        pygame.draw.rect(screen, wall_lo,                            # lower wall
                         pygame.Rect(wall.x, wall.centery, wall.w, wall.h - wall.h // 2))
        # window grid (lit panes), leaving room for a central door
        win_c, frame_c = (250, 226, 170), (78, 64, 52)
        ww, wgap = 10, 8
        wy0 = wall.y + 7
        rows_w = max(1, (wall.h - 14) // (ww + wgap))
        for ri in range(rows_w):
            ry = wy0 + ri * (ww + wgap)
            if ry + ww > wall.bottom - 4:
                break
            for wx0 in range(wall.x + 7, wall.right - ww - 2, ww + wgap):
                # skip the centre-bottom cell to leave space for the door
                if ri == rows_w - 1 and abs((wx0 + ww // 2) - x) < max(9, int(hw * 0.5)):
                    continue
                pygame.draw.rect(screen, win_c, pygame.Rect(wx0, ry, ww, ww))
                pygame.draw.rect(screen, frame_c, pygame.Rect(wx0, ry, ww, ww), 1)
        dw = max(7, int(hw * 0.32))
        door = pygame.Rect(x - dw, wall.bottom - max(12, int(hh * 0.4)),
                           2 * dw, max(12, int(hh * 0.4)))
        pygame.draw.rect(screen, (92, 70, 52), door)
        pygame.draw.rect(screen, frame_c, door, 1)
        pygame.draw.circle(screen, (232, 224, 158), (door.right - 3, door.centery), 1)
        pygame.draw.rect(screen, edge, wall, 2)                       # wall outline
        roof = pygame.Rect(x - hw - 3, y - hh, 2 * hw + 6, roof_h)     # roof slab
        pygame.draw.rect(screen, roof_c, roof)
        pygame.draw.rect(screen, edge, roof, 2)
        _label(screen, font, f.label, x, y - hh + roof_h // 2)

    elif k == FeatureKind.PAVILION:
        # An OPEN canopy/pavilion — drawn over robots/humans (they pass under
        # it). Translucent roof so you see who is sheltering, four support posts
        # at the corners, a ridge line, and a label. Never blocks movement.
        post = (96, 80, 60)
        for px, py in [(x - hw + 4, y + hh - 4), (x + hw - 4, y + hh - 4),
                       (x - hw + 4, y - hh + 4), (x + hw - 4, y - hh + 4)]:
            pygame.draw.line(screen, post, (px, py), (px, py - 6), 3)   # post legs
        roof = pygame.Surface((2 * hw + 10, 2 * hh + 10), pygame.SRCALPHA)
        pygame.draw.polygon(roof, (210, 188, 150, 120),               # translucent
                            [(0, hh + 5), (hw + 5, 0), (2 * hw + 10, hh + 5),
                             (hw + 5, 2 * hh + 10)])
        screen.blit(roof, (x - hw - 5, y - hh - 5))
        edge = (150, 128, 96)
        pygame.draw.polygon(screen, edge,                             # roof outline
                            [(x - hw - 5, y), (x, y - hh - 5),
                             (x + hw + 5, y), (x, y + hh + 5)], 2)
        pygame.draw.line(screen, edge, (x, y - hh - 5), (x, y + hh + 5), 1)  # ridge
        _label(screen, font, f.label, x, y)

    elif k == FeatureKind.SHELTER:
        # A RECTANGULAR open shelter — drag-drawn to any size, drawn over
        # robots/humans (they pass under it). Translucent rectangular roof with
        # corner posts, a beam border, and a couple of ridge lines. Never blocks.
        post = (96, 80, 60)
        for px, py in [(x - hw + 4, y - hh + 4), (x + hw - 4, y - hh + 4),
                       (x - hw + 4, y + hh - 4), (x + hw - 4, y + hh - 4)]:
            pygame.draw.line(screen, post, (px, py), (px, py - 6), 3)   # post legs
        roof = pygame.Surface((2 * hw, 2 * hh), pygame.SRCALPHA)
        roof.fill((210, 188, 150, 120))                                # translucent
        screen.blit(roof, (x - hw, y - hh))
        edge = (150, 128, 96)
        pygame.draw.rect(screen, edge,                                 # beam border
                         pygame.Rect(x - hw, y - hh, 2 * hw, 2 * hh), 2)
        for ry in range(y - hh + 10, y + hh - 4, 14):                  # ridge battens
            pygame.draw.line(screen, (176, 154, 120), (x - hw + 3, ry),
                             (x + hw - 3, ry), 1)
        _label(screen, font, f.label, x, y - hh + 9)


def draw_human(screen, h: Human, oy: int) -> None:
    import pygame
    x, y = int(h.x), int(h.y + oy)
    pygame.draw.circle(screen, (236, 168, 70), (x, y + 4), 7)        # body
    pygame.draw.circle(screen, (180, 120, 60), (x, y + 4), 7, 1)
    pygame.draw.circle(screen, (248, 212, 170), (x, y - 4), 4)       # head


def draw_mosquito(screen, m: Mosquito, oy: int) -> None:
    """A very small black/white dot."""
    x, y = int(m.x), int(m.y + oy)
    base = (245, 245, 245) if m.white else (12, 12, 12)
    ring = (12, 12, 12) if m.white else (235, 235, 235)
    if 0 <= x < screen.get_width() and 0 <= y < screen.get_height():
        screen.set_at((x, y), base)
        if x + 1 < screen.get_width():
            screen.set_at((x + 1, y), ring)


def _label(screen, font, text, cx, cy) -> None:
    img = font.render(text, True, (245, 245, 245))
    screen.blit(img, (cx - img.get_width() // 2, cy - img.get_height() // 2))


def _draw_water(screen, f: "Feature", oy: int) -> None:
    """Render a lasso-drawn pond as a textured water body (no outline):
    base water + horizontal elliptical shades + short ripple/wave reflection
    strokes, all clipped to the polygon. Deterministic per pond (no flicker)."""
    import pygame
    poly = [(int(px), int(py + oy)) for px, py in f.poly]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    bw, bh = max(1, maxx - minx), max(1, maxy - miny)
    local = [(px - minx, py - miny) for px, py in poly]

    tex = pygame.Surface((bw, bh), pygame.SRCALPHA)
    pygame.draw.polygon(tex, (58, 112, 178), local)        # base water

    rng = random.Random(f.uid)                              # stable per pond
    shades = [(80, 136, 198), (94, 152, 212), (46, 98, 162)]
    for _ in range(max(3, bw * bh // 2600)):                # a few elliptical shades
        ex, ey = rng.randint(0, bw), rng.randint(0, bh)
        ew, eh = rng.randint(16, 34), rng.randint(4, 8)
        pygame.draw.ellipse(tex, rng.choice(shades),
                            pygame.Rect(ex - ew // 2, ey - eh // 2, ew, eh))
    for _ in range(max(3, bw * bh // 2200)):                # sparse ripple strokes
        x, y = rng.randint(0, max(1, bw - 18)), rng.randint(0, bh)
        seg, stp, amp = rng.randint(2, 3), rng.randint(4, 5), 1
        pts = [(x + i * stp, y + (amp if i % 2 else -amp)) for i in range(seg + 1)]
        pygame.draw.lines(tex, (152, 196, 226), False, pts, 1)

    mask = pygame.Surface((bw, bh), pygame.SRCALPHA)        # clip to polygon
    pygame.draw.polygon(mask, (255, 255, 255, 255), local)
    tex.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    screen.blit(tex, (minx, miny))
