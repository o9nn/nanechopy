"""
====================================
NN Layer Nodes (:mod:`reservoirpy.pytorch.nodes.nn_layers`)
====================================

Standard neural network layers wrapped as reservoir computing nodes.

These nodes wrap PyTorch's nn module layers to integrate them into
the reservoir computing framework, enabling hybrid architectures
that combine traditional deep learning with reservoir computing.

Supported layers:
- Linear (fully connected)
- Conv1d, Conv2d (convolutional)
- MultiheadAttention (transformer attention)
- LayerNorm, BatchNorm (normalization)
"""

# License: MIT License
# Copyright: nanechopy contributors

from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()

from ..node import (
    TorchNode, TorchTrainableNode, TorchParallelNode,
    TorchState, TorchTimestep, TorchTimeseries, to_tensor
)


class NNLinearNode(TorchTrainableNode):
    """Linear (fully connected) layer node.
    
    Wraps torch.nn.Linear as a reservoir computing node.
    Supports training via ridge regression or gradient descent.
    
    Parameters
    ----------
    units : int
        Number of output units (output_dim)
    input_dim : int, optional
        Input dimension (inferred if not provided)
    bias : bool, default True
        Whether to include bias term
    activation : str or callable, optional
        Activation function ('relu', 'tanh', 'sigmoid', 'gelu', or callable)
    ridge : float, optional
        Ridge regression parameter for training
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> # Create a linear layer with ReLU
    >>> linear = NNLinearNode(units=100, activation='relu')
    >>> output = linear.run(input_data)
    
    >>> # Train with ridge regression
    >>> linear.fit(X_train, y_train, ridge=1e-6)
    """
    
    def __init__(
        self,
        units: int,
        input_dim: Optional[int] = None,
        bias: bool = True,
        activation: Optional[Union[str, Callable]] = None,
        ridge: Optional[float] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, requires_grad=requires_grad, name=name)
        
        self.units = units
        self.output_dim = units
        self.input_dim = input_dim
        self.use_bias = bias
        self.ridge = ridge
        
        # Parse activation
        if isinstance(activation, str):
            self.activation = self._get_activation(activation)
        else:
            self.activation = activation
        
        self._linear: Optional[nn.Linear] = None
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'gelu': F.gelu,
            'silu': F.silu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'softmax': lambda x: F.softmax(x, dim=-1),
            'softplus': F.softplus,
        }
        return activations.get(name.lower(), lambda x: x)
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the linear layer."""
        self._set_input_dim(x)
        
        # Create nn.Linear
        self._linear = nn.Linear(
            self.input_dim, 
            self.units, 
            bias=self.use_bias,
            device=self.device,
            dtype=self.dtype
        )
        
        # Register parameters
        self.register_parameter("weight", self._linear.weight)
        if self.use_bias:
            self.register_parameter("bias", self._linear.bias)
        
        # Initialize state
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply linear transformation."""
        result = F.linear(x, self.weight, self.bias if self.use_bias else None)
        
        if self.activation is not None:
            result = self.activation(result)
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Vectorized run."""
        result = F.linear(x, self.weight, self.bias if self.use_bias else None)
        
        if self.activation is not None:
            result = self.activation(result)
        
        return {"out": result[-1]}, result
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "NNLinearNode":
        """Train the linear layer.
        
        Uses ridge regression if ridge parameter is provided,
        otherwise uses the stored ridge value.
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Apply warmup
        x_train = x[warmup:]
        y_train = y[warmup:] if y is not None else None
        
        if y_train is None:
            return self
        
        # Ridge regression
        ridge_param = ridge or self.ridge or 1e-6
        
        # X: (n_samples, input_dim), Y: (n_samples, output_dim)
        X = x_train
        Y = y_train
        
        # Add bias column if using bias
        if self.use_bias:
            ones = torch.ones(X.shape[0], 1, device=self.device, dtype=self.dtype)
            X_aug = torch.cat([X, ones], dim=1)
        else:
            X_aug = X
        
        # Ridge regression: W = (X^T X + λI)^(-1) X^T Y
        XtX = X_aug.T @ X_aug
        XtY = X_aug.T @ Y
        
        reg = ridge_param * torch.eye(XtX.shape[0], device=self.device, dtype=self.dtype)
        W = torch.linalg.solve(XtX + reg, XtY)
        
        # Update parameters
        if self.use_bias:
            self.weight.data = W[:-1].T
            self.bias.data = W[-1]
        else:
            self.weight.data = W.T
        
        return self


class NNConvNode(TorchNode):
    """1D Convolutional layer node.
    
    Wraps torch.nn.Conv1d for sequence processing in reservoir computing.
    
    Parameters
    ----------
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolving kernel
    stride : int, default 1
        Stride of the convolution
    padding : int or str, default 0
        Padding added to input
    dilation : int, default 1
        Spacing between kernel elements
    groups : int, default 1
        Number of blocked connections
    bias : bool, default True
        Whether to include bias
    activation : str or callable, optional
        Activation function
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
    """
    
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, str] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        activation: Optional[Union[str, Callable]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        # Parse activation
        if isinstance(activation, str):
            self.activation = self._get_activation(activation)
        else:
            self.activation = activation
        
        self._conv: Optional[nn.Conv1d] = None
        self._buffer: Optional[Tensor] = None
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'gelu': F.gelu,
        }
        return activations.get(name.lower(), lambda x: x)
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the convolutional layer."""
        self._set_input_dim(x)
        
        # Create nn.Conv1d
        self._conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.use_bias,
            device=self.device,
            dtype=self.dtype
        )
        
        # Register parameters
        self.register_parameter("weight", self._conv.weight)
        if self.use_bias:
            self.register_parameter("bias", self._conv.bias)
        
        # Initialize buffer for step-by-step processing
        self._buffer = torch.zeros(
            self.kernel_size, self.input_dim,
            device=self.device, dtype=self.dtype
        )
        
        self.output_dim = self.out_channels
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply convolution on buffered input."""
        # Update buffer
        self._buffer = torch.roll(self._buffer, -1, dims=0)
        self._buffer[-1] = x
        
        # Apply convolution: (1, in_channels, kernel_size) -> (1, out_channels, 1)
        x_conv = self._buffer.T.unsqueeze(0)  # (1, in_channels, kernel_size)
        result = self._conv(x_conv).squeeze()  # (out_channels,)
        
        if self.activation is not None:
            result = self.activation(result)
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Vectorized convolution on full sequence."""
        # x: (timesteps, input_dim) -> (1, input_dim, timesteps)
        x_conv = x.T.unsqueeze(0)
        
        # Apply convolution
        result = self._conv(x_conv)  # (1, out_channels, out_timesteps)
        result = result.squeeze(0).T  # (out_timesteps, out_channels)
        
        if self.activation is not None:
            result = self.activation(result)
        
        return {"out": result[-1]}, result


class NNAttentionNode(TorchNode):
    """Multi-head attention node.
    
    Wraps torch.nn.MultiheadAttention for transformer-style
    attention in reservoir computing architectures.
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    dropout : float, default 0.0
        Dropout probability
    bias : bool, default True
        Whether to include bias in projections
    add_bias_kv : bool, default False
        Whether to add bias to key and value
    kdim : int, optional
        Dimension of key (defaults to embed_dim)
    vdim : int, optional
        Dimension of value (defaults to embed_dim)
    batch_first : bool, default True
        Whether input is (batch, seq, feature)
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> attn = NNAttentionNode(embed_dim=64, num_heads=4)
    >>> output = attn.run(sequence_data)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_bias = bias
        self.add_bias_kv = add_bias_kv
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.batch_first = batch_first
        
        self._attn: Optional[nn.MultiheadAttention] = None
        self._context_buffer: Optional[Tensor] = None
        self._context_size: int = 16  # Default context window
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the attention layer."""
        self._set_input_dim(x)
        
        # Ensure input_dim matches embed_dim
        if self.input_dim != self.embed_dim:
            # Add projection layer
            self._input_proj = nn.Linear(
                self.input_dim, self.embed_dim,
                device=self.device, dtype=self.dtype
            )
        else:
            self._input_proj = None
        
        # Create attention layer
        self._attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.use_bias,
            add_bias_kv=self.add_bias_kv,
            kdim=self.kdim,
            vdim=self.vdim,
            batch_first=self.batch_first,
            device=self.device,
            dtype=self.dtype
        )
        
        # Initialize context buffer
        self._context_buffer = torch.zeros(
            self._context_size, self.embed_dim,
            device=self.device, dtype=self.dtype
        )
        
        self.output_dim = self.embed_dim
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply attention on context window."""
        # Project input if needed
        if self._input_proj is not None:
            x = self._input_proj(x)
        
        # Update context buffer
        self._context_buffer = torch.roll(self._context_buffer, -1, dims=0)
        self._context_buffer[-1] = x
        
        # Apply attention: query is current, key/value is context
        query = x.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
        key_value = self._context_buffer.unsqueeze(0)  # (1, context_size, embed_dim)
        
        attn_output, _ = self._attn(query, key_value, key_value)
        result = attn_output.squeeze()  # (embed_dim,)
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Apply self-attention on full sequence."""
        # Project input if needed
        if self._input_proj is not None:
            x = self._input_proj(x)
        
        # Add batch dimension: (seq, embed) -> (1, seq, embed)
        x_batch = x.unsqueeze(0)
        
        # Self-attention
        attn_output, _ = self._attn(x_batch, x_batch, x_batch)
        result = attn_output.squeeze(0)  # (seq, embed_dim)
        
        return {"out": result[-1]}, result


class NNLayerNormNode(TorchNode):
    """Layer normalization node.
    
    Wraps torch.nn.LayerNorm for normalization in reservoir computing.
    
    Parameters
    ----------
    normalized_shape : int or tuple
        Shape to normalize over
    eps : float, default 1e-5
        Epsilon for numerical stability
    elementwise_affine : bool, default True
        Whether to learn affine parameters
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
    """
    
    def __init__(
        self,
        normalized_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self._norm: Optional[nn.LayerNorm] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the layer norm."""
        self._set_input_dim(x)
        
        norm_shape = self.normalized_shape or self.input_dim
        
        self._norm = nn.LayerNorm(
            normalized_shape=norm_shape,
            eps=self.eps,
            elementwise_affine=self.elementwise_affine,
            device=self.device,
            dtype=self.dtype
        )
        
        self.output_dim = self.input_dim
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply layer normalization."""
        result = self._norm(x)
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Vectorized layer normalization."""
        result = self._norm(x)
        return {"out": result[-1]}, result
