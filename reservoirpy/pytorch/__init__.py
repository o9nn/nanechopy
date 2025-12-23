"""
====================================
PyTorch Integration (:mod:`reservoirpy.pytorch`)
====================================

PyTorch-based reservoir computing with ATen, nn, and RNN nodes.

This module provides:

- **TorchNode**: Base class for PyTorch nodes
- **ATen Nodes**: Low-level tensor operations
- **NN Nodes**: Neural network layers (Linear, Conv, Attention)
- **RNN Nodes**: Recurrent networks (RNN, LSTM, GRU, ESN)
- **Autognosis**: Self-aware learning engine from echo-jnn
- **Models**: High-level model classes
- **Ops**: Node connection operations

Key Classes
-----------
.. autosummary::
   :toctree: generated/

    TorchNode
    TorchTrainableNode
    ESNTorchNode
    AutognosisNode
    AutognosisModel
    TorchReservoirModel
    HybridESNModel

Examples
--------
>>> from reservoirpy.pytorch import ESNTorchNode, AutognosisModel
>>> 
>>> # Create ESN with PyTorch backend
>>> esn = ESNTorchNode(units=500, sr=0.9, lr=0.3)
>>> esn.fit(X_train, y_train)
>>> predictions = esn.run(X_test)
>>> 
>>> # Create self-aware model
>>> model = AutognosisModel(base_order=5)
>>> model.fit(X_train, y_train)
>>> predictions = model.run(X_test)
>>> print(model.explain_parameters())
"""

from .node import (
    TorchNode,
    TorchTrainableNode,
    TorchOnlineNode,
    TorchParallelNode,
    TorchState,
    TorchTimestep,
    TorchTimeseries,
    to_tensor,
)

from .nodes import (
    # ATen nodes
    ATenNode,
    ATenOp,
    ATenChain,
    # NN nodes
    NNLinearNode,
    NNConvNode,
    NNLayerNormNode,
    NNAttentionNode,
    # RNN nodes
    RNNNode,
    LSTMNode,
    GRUNode,
    ESNTorchNode,
)

from .autognosis import (
    AutognosisEngine,
    AutognosisNode,
    A000081Parameters,
    get_a000081_value,
    derive_parameters,
    BSeriesKernel,
    RootedTree,
    OntogeneticState,
    OntogeneticEngine,
)

from .model import (
    TorchReservoirModel,
    HybridESNModel,
    AutognosisModel,
)

from .ops import (
    link,
    merge,
    link_feedback,
    TorchModel,
    MergedNode,
)

__all__ = [
    # Base classes
    "TorchNode",
    "TorchTrainableNode",
    "TorchOnlineNode",
    "TorchParallelNode",
    "TorchState",
    "TorchTimestep",
    "TorchTimeseries",
    "to_tensor",
    # ATen nodes
    "ATenNode",
    "ATenOp",
    "ATenChain",
    # NN nodes
    "NNLinearNode",
    "NNConvNode",
    "NNLayerNormNode",
    "NNAttentionNode",
    # RNN nodes
    "RNNNode",
    "LSTMNode",
    "GRUNode",
    "ESNTorchNode",
    # Autognosis
    "AutognosisEngine",
    "AutognosisNode",
    "A000081Parameters",
    "get_a000081_value",
    "derive_parameters",
    "BSeriesKernel",
    "RootedTree",
    "OntogeneticState",
    "OntogeneticEngine",
    # Models
    "TorchReservoirModel",
    "HybridESNModel",
    "AutognosisModel",
    # Ops
    "link",
    "merge",
    "link_feedback",
    "TorchModel",
    "MergedNode",
]
