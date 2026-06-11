"""Shared utilities for the Ecological Computation sandbox.

Nothing in this package is a research contribution on its own — it provides
the grid world, the point-to-point / coverage scenario library, and the base
Glasius Bio-inspired Neural Network reference implementation that the two
algorithm modules (`ppstar`, `hfaccpp`) build on.
"""

from .environment import (  # noqa: F401
    GridWorld,
    Cell,
    pp_scenario,
    predator_setting,
    coverage_scenario,
    start_cell,
)
from .replicated_gbnn import GBNN  # noqa: F401
