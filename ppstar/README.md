# `ppstar` - Predator-Dominance and Prey-Approach Planner (PP*)

Human-aware **point-to-point** path planning for a mosquito vector-control
robot. A grid A\* whose cost-to-go is reshaped by a *prey-approach* pull toward
the mosquito hotspot and a *predator-dominance* penalty around a human crowd, so
the robot reaches its target while routing clear of people. Plain A\* and
Dijkstra are included as baselines (the demo's planner toggle cycles all three).

## Paper

> Ash Yaw Sang, Veerajagadheswar Prabakaran, Mohan Rajesh Elara, Anh Vu Le.
> **Efficient Path Planner via Predator Dominance and Prey Approach for a
> Vector Surveillance Robot.** In *Social Robotics + AI* (ICSR+AI 2025),
> Lecture Notes in Computer Science (LNAI) vol. 16131, pp. 281-294. Springer,
> Singapore, 2026.
> DOI: [10.1007/978-981-95-2379-5_19](https://doi.org/10.1007/978-981-95-2379-5_19)

```bibtex
@inproceedings{sang2026ppstar,
  author    = {Sang, Ash Yaw and Prabakaran, Veerajagadheswar and
               Elara, Mohan Rajesh and Le, Anh Vu},
  title     = {Efficient Path Planner via Predator Dominance and Prey Approach
               for a Vector Surveillance Robot},
  booktitle = {Social Robotics + AI (ICSR+AI 2025)},
  series    = {Lecture Notes in Computer Science},
  volume    = {16131},
  pages     = {281--294},
  publisher = {Springer, Singapore},
  year      = {2026},
  doi       = {10.1007/978-981-95-2379-5_19}
}
```

## The algorithm

The planner is an 8-connected A\* over the occupancy grid, returning a
**node-to-node** path (one waypoint per grid cell). The priority of a candidate
cell `n` is (paper Eqn 6)

```
q(n) = g(n) + h1(n) + h0(n)
```

| Term | Meaning | Formula |
|---|---|---|
| `g(n)`  | accumulated Euclidean path cost | diagonal step = √2 |
| `h1(n)` | **prey approach** - pull toward the goal (mosquito hotspot) | `dist(n, goal)` (Eqn 4) |
| `h0(n)` | **predator dominance** - penalty inside each crowd's avoidance zone | `(Cs - Ct) · max(Cr² - dist(n, P0)², 0)` (Eqn 3) |

`h0` is zero outside the dominance radius and rises quadratically toward the
crowd centre `P0` inside it. It is deliberately **non-admissible** - a repulsion
field, not a lower bound - which is what bends the path away from people.

**The predator is a cluster of people, not one person.** People standing close
together (single-linkage within a proximity threshold) are grouped into one
predator described by the paper's crowd variables:

| Variable | Meaning | In code |
|---|---|---|
| `Cs` | **crowd size** - population weight, `cs_per_person · N`, scales with the head-count `N` | `Predator.Cs` |
| `Ct` | **crowd tolerance** - the robot's allowance toward the crowd | `Predator.Ct` |
| `Cr` | **crowd radius** - dispersion radius (`base_radius` + the cluster's spread) | `Predator.Cr` |

So the robot threads through a lone person but takes a wider berth around a
larger, denser crowd. Clusters are rebuilt from the live human positions every
re-plan.

Reductions to the baselines:

- drop `h0` (no crowd) → plain **A\*** (`PPStar.astar`).
- drop `h1` too → **Dijkstra** (`PPStar.dijkstra`).

`ppstar_maze(maze, start, end, crowd, setnow)` is the module's main entry,
mirroring the reference code's function name; `crowd` is the list of people
cells, clustered internally.

## Theory pointer

| Concept | Symbol | Paper § / Eqn | Code reference |
|---|---|---|---|
| Prey-approach pull | `h1` | Eqn 4 | `open_ppstar.py::PPStar._search` (`use_h1`) |
| Predator-dominance penalty | `h0` | Eqn 3 | `open_ppstar.py::PPStar._h0` |
| Crowd clustering (Cs, Ct, Cr) | - | §III-A/B | `open_ppstar.py::cluster_predators` |
| Combined cost | `q` | Eqn 6 | `PPStar._search` |
| Baseline A\* / Dijkstra | - | §IV | `PPStar.astar`, `PPStar.dijkstra` |

## Run standalone

```bash
python -m ppstar.open_ppstar --scenario 1   # 1, 2, or 3
```

Prints, for each of Dijkstra / A\* / PP\*: path length, steps, nodes expanded,
and accumulated crowd exposure, followed by PP\*'s exposure reduction versus
A\*, plus the clustered predators (centre, `N`, `Cs`, `Ct`, `Cr`). On the three
shipped mazes PP\* reaches the prey with **zero** crowd exposure where the
baselines cut through the crowd.

## Known limitations

- Crowd clustering is single-linkage by Euclidean proximity; the deployed
  system would estimate a crowd-density field and tolerances from perception.
- `h0` being non-admissible means PP\* optimises the *reshaped* cost, not raw
  distance - that is the intent, but it expands more nodes than plain A\*.
- The interactive demo re-plans periodically (every 2 ticks) against the live
  crowd; hardware dynamics and behavioural recovery are not modelled.
