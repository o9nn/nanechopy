"""
====================================
Autognosis Engine (:mod:`reservoirpy.pytorch.autognosis`)
====================================

Self-aware learning engine integrated from echo-jnn.

The Autognosis Engine provides:
- **A000081 Parameter Derivation**: Mathematically grounded parameters from OEIS sequence
- **Deep Tree Echo Integration**: Rooted tree-based reservoir computing
- **B-Series Computational Ridges**: Numerical integration methods
- **Ontogenetic Evolution**: Self-organizing parameter adaptation
- **J-Surface Reactor**: Unified gradient-evolution dynamics

Key Components
--------------
.. autosummary::
   :toctree: generated/

    AutognosisEngine
    AutognosisNode
    A000081Parameters
    BSeriesKernel
    OntogeneticState

Architecture
------------
The autognosis system operates through five integrated layers:

1. **Rooted Tree Foundation** (A000081): Fundamental structures
2. **B-Series Computational Ridges**: Numerical integration methods
3. **Reservoir Echo States**: Temporal pattern learning
4. **Membrane Computing Gardens**: Evolutionary containers
5. **J-Surface Reactor Core**: Unified gradient-evolution dynamics
"""

from .engine import AutognosisEngine
from .node import AutognosisNode
from .a000081 import A000081Parameters, get_a000081_value, derive_parameters
from .bseries import BSeriesKernel, RootedTree
from .ontogenetic import OntogeneticState, OntogeneticEngine

__all__ = [
    "AutognosisEngine",
    "AutognosisNode",
    "A000081Parameters",
    "get_a000081_value",
    "derive_parameters",
    "BSeriesKernel",
    "RootedTree",
    "OntogeneticState",
    "OntogeneticEngine",
]
