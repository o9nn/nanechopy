"""
====================================
PyTorch Nodes (:mod:`reservoirpy.pytorch.nodes`)
====================================

Collection of PyTorch-based neural network nodes.

ATen Nodes
----------
Low-level tensor operations using PyTorch's ATen backend:
- ATenNode: Base class for ATen operations
- ATenMatmulNode: Matrix multiplication
- ATenAddNode: Element-wise addition
- ATenMulNode: Element-wise multiplication
- ATenTanhNode: Hyperbolic tangent activation
- ATenSigmoidNode: Sigmoid activation
- ATenSoftmaxNode: Softmax activation
- ATenLayerNormNode: Layer normalization
- ATenDropoutNode: Dropout regularization
- ATenConcatNode: Tensor concatenation
- ATenSplitNode: Tensor splitting
- ATenReshapeNode: Tensor reshaping

NN Nodes
--------
Neural network layers:
- NNLinearNode: Fully connected layer
- NNConv1dNode: 1D convolution
- NNConv2dNode: 2D convolution
- NNBatchNormNode: Batch normalization
- NNLayerNormNode: Layer normalization
- NNDropoutNode: Dropout
- NNAttentionNode: Multi-head attention
- NNEmbeddingNode: Embedding layer

RNN Nodes
---------
Recurrent neural network variants:
- RNNNode: Vanilla RNN
- LSTMNode: Long Short-Term Memory
- GRUNode: Gated Recurrent Unit
- ESNTorchNode: Echo State Network with PyTorch backend
"""

from .aten import (
    ATenNode,
    ATenOp,
    ATenChain,
)

from .nn_layers import (
    NNLinearNode,
    NNConvNode,
    NNLayerNormNode,
    NNAttentionNode,
)

from .rnn import (
    RNNNode,
    LSTMNode,
    GRUNode,
    ESNTorchNode,
)

__all__ = [
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
]
