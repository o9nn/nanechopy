"""
====================================
DTESNN - Deep Tree Echo State Neural Network
(:mod:`reservoirpy.pytorch.dtesnn`)
====================================

Deep Tree Echo State Neural Network with A000081-synchronized components.

The DTESNN architecture unifies three isomorphic systems through the A000081
sequence (rooted tree enumeration):

1. **B-Series Ridge Trees** (Readout Layer)
   - Rooted trees define ridge regression structure
   - Tree roots are "planted" in membrane compartments
   - Elementary weights derived from tree density/symmetry

2. **P-System Membrane Nests** (Reservoir Layer)
   - Hierarchical membrane compartments
   - Each membrane contains reservoir units
   - Communication rules based on tree adjacency

3. **J-Surface Elementary Differentials** (ESN Layer)
   - Elementary differentials indexed by trees
   - Symplectic dynamics on J-surface manifold
   - Gradient-evolution coupling

All three systems share synchronized parameters through A000081 isomorphism:
- Tree order n → A000081[n] structures in each system
- Cumulative counts define layer sizes
- Tree properties (density, symmetry) determine weights

Key Classes
-----------
.. autosummary::
   :toctree: generated/

    DTESNN
    BSeriesRidgeTree
    MembraneReservoir
    JSurfaceESN
    A000081Synchronizer

Architecture Diagram
--------------------
```
Input → [J-Surface ESN] → [Membrane Reservoir] → [Ridge Tree Readout] → Output
              ↑                    ↑                      ↑
              └────────────────────┴──────────────────────┘
                        A000081 Synchronization
```

Examples
--------
>>> from reservoirpy.pytorch.dtesnn import DTESNN
>>> 
>>> # Create DTESNN with base order 6
>>> model = DTESNN(base_order=6, seed=42)
>>> 
>>> # Train
>>> model.fit(X_train, y_train, warmup=100)
>>> 
>>> # Predict
>>> predictions = model.run(X_test)
>>> 
>>> # Inspect synchronized structure
>>> print(model.get_tree_structure())
"""

from .synchronizer import (
    A000081Synchronizer,
    TreeIsomorphism,
    SynchronizedParameters,
    IsomorphismType,
)
from .ridge_tree import (
    BSeriesRidgeTree,
    HierarchicalRidgeTree,
    RidgeTreeNode,
)
from .membrane_reservoir import (
    MembraneReservoir,
    MembraneNest,
    MembraneCompartment,
    CommunicationRule,
)
from .jsurface_esn import (
    JSurfaceESN,
    ElementaryDifferential,
    ElementaryDifferentialNode,
    JMatrix,
)
from .model import DTESNN, DTESNNConfig, DTESNNEnsemble

__all__ = [
    # Main Model
    "DTESNN",
    "DTESNNConfig",
    "DTESNNEnsemble",
    # Synchronization
    "A000081Synchronizer",
    "TreeIsomorphism",
    "SynchronizedParameters",
    "IsomorphismType",
    # Ridge Trees (Readout)
    "BSeriesRidgeTree",
    "HierarchicalRidgeTree",
    "RidgeTreeNode",
    # Membrane Reservoir
    "MembraneReservoir",
    "MembraneNest",
    "MembraneCompartment",
    "CommunicationRule",
    # J-Surface ESN
    "JSurfaceESN",
    "ElementaryDifferential",
    "ElementaryDifferentialNode",
    "JMatrix",
]
