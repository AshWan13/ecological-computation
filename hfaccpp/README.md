# `hfaccpp` - Human-First Complete Coverage (HFA-CCPP)

Human-activity-aware **complete area coverage** for a mosquito-control robot,
a human-first coverage planner built on the
Glasius Bio-inspired Neural Network (GBNN): it keeps GBNN's deadlock-free
guarantee of full coverage, but services human-dense regions - where mosquitoes
congregate - first.

## Paper

> Ash Yaw Sang Wan, Prabakaran Veerajagadheswar, Mohan Rajesh Elara, Anh Vu Le.
> **Human activity-aware coverage path planning for robot-based mosquito
> control.** *Scientific Reports* **15**, 31009 (2025).
> DOI: [10.1038/s41598-025-16114-1](https://doi.org/10.1038/s41598-025-16114-1)

```bibtex
@article{wansang2025hfaccpp,
  author  = {Wan, Ash Yaw Sang and Veerajagadheswar, Prabakaran and
             Elara, Mohan Rajesh and Le, Anh Vu},
  title   = {Human activity-aware coverage path planning for robot-based
             mosquito control},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {31009},
  year    = {2025},
  doi     = {10.1038/s41598-025-16114-1}
}
```

## The algorithm

The base GBNN (see [`../common/replicated_gbnn.py`](../common/replicated_gbnn.py))
relaxes a shunting neural field over the grid:

```
x_i = G( Σ_j w_ij · [x_j]^+  +  I_i )
G(x) = -1 if x<0 ;  1 if x≥1 ;  b·x otherwise
w_ij = exp(-α · d_ij²) if d_ij < r else 0
```

with `I_i = +E` for uncovered free cells, `-E` for obstacles, `0` for covered
cells. Because uncovered cells are the only positive sources, the field has no
interior local maxima and coverage is **deadlock-free and complete**.

**The human-first contribution - two stacked neural layers.** HFA-CCPP runs a
second GBNN, so each step has two layers over the same grid:

- the **coverage layer ν** - sources are all uncovered cells (above), and
- the **social layer μ** - sources are the *uncovered cells in a person's
  vicinity*, so its activity gradient points to the nearest un-serviced crowd.

The next waypoint is the free neighbour with the highest **sum** of the two
layers (paper Eqn 10):

```
wp_{i+1} = argmax_k ( N_kν  +  risk_bias · N_kμ )
```

Because the social layer is *summed* (not a separate beeline), the robot keeps
sweeping while leaning toward human-dense cells; as those cells are covered the
social sources vanish and μ fades, so the robot finishes on plain coverage.
Occupied cells settle to `-1` in both layers and are excluded from selection.
Completeness is unchanged - only the *order* is human-first.

The robot's area of effect mitigates each person's mosquito **risk** as it
works, modelled with the paper's residual risk `r*_r` and carrying risk `r*_c`.

## Theory pointer

| Concept | Symbol | Paper § / Eqn | Code reference |
|---|---|---|---|
| Neuron transfer | `f(x)` | Eqn 1 / 4 | `common/replicated_gbnn.py::GBNN.G_x` |
| Lateral weights | `w_ij` | Eqn 3 | `common/replicated_gbnn.py::GBNN.w_ij` |
| External input | `α_i` | Eqn 2 / 5 | `common/replicated_gbnn.py::GBNN.external_input` |
| Coverage / social layers | `ν`, `μ` | §4.3-4.4 | `HFACoveragePlanner.gbnn_coverage`, `…gbnn_social` |
| Summed-layer waypoint | `wp` | Eqn 10 | `open_hfaccpp.py::HFACoveragePlanner.step` |
| Residual / carrying risk | `r*_r`, `r*_c` | Eqn 7 / 8 | `…_residual_risk`, `…_carrying_risk` |
| Coverage footprint | - | §4 | `open_hfaccpp.py::HFACoveragePlanner._cover_footprint` |

## Run standalone

```bash
python -m hfaccpp.open_hfaccpp --rows 12 --cols 12 --seed 0
```

Both base GBNN and HFA-CCPP reach 100% coverage; HFA-CCPP reports a lower
human-weighted coverage delay (high-human cells visited earlier) - typically a
40-55% reduction on the shipped scenes.

## Known limitations

- The map and human cells come from the toy `common.environment` scene; the
  deployed system consumes a SLAM map and a perception-derived human-density
  estimate.
- `footprint`, `risk_threshold`, and the turn penalty are constructor
  arguments, not auto-tuned.
- Single-process simulation: no multi-robot coordination, behavioural recovery,
  or hardware dynamics.
