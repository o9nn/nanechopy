"""
====================================
RNN Nodes (:mod:`reservoirpy.pytorch.nodes.rnn`)
====================================

Recurrent Neural Network nodes for reservoir computing.

This module provides RNN variants as reservoir computing nodes:
- RNNNode: Vanilla RNN
- LSTMNode: Long Short-Term Memory
- GRUNode: Gated Recurrent Unit
- ESNTorchNode: Echo State Network with PyTorch backend

These nodes combine the temporal dynamics of RNNs with the
reservoir computing paradigm, enabling hybrid architectures.
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
        Generator = Any
        float32 = Any
        float64 = Any
        tanh = None
        relu = None
        sigmoid = None
    torch = _TorchPlaceholder()

from ..node import (
    TorchNode, TorchTrainableNode, TorchOnlineNode, TorchParallelNode,
    TorchState, TorchTimestep, TorchTimeseries, to_tensor
)


class RNNNode(TorchNode):
    """Vanilla RNN node.
    
    Implements a simple recurrent neural network:
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    
    Parameters
    ----------
    hidden_size : int
        Number of hidden units
    input_dim : int, optional
        Input dimension (inferred if not provided)
    nonlinearity : str, default 'tanh'
        Activation function ('tanh' or 'relu')
    bias : bool, default True
        Whether to include bias
    num_layers : int, default 1
        Number of stacked RNN layers
    dropout : float, default 0.0
        Dropout probability between layers
    bidirectional : bool, default False
        Whether to use bidirectional RNN
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> rnn = RNNNode(hidden_size=100)
    >>> output = rnn.run(sequence_data)
    """
    
    def __init__(
        self,
        hidden_size: int,
        input_dim: Optional[int] = None,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.nonlinearity = nonlinearity
        self.use_bias = bias
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self._rnn: Optional[nn.RNN] = None
        self._hidden: Optional[Tensor] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the RNN."""
        self._set_input_dim(x)
        
        self._rnn = nn.RNN(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity=self.nonlinearity,
            bias=self.use_bias,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            device=self.device,
            dtype=self.dtype
        )
        
        # Output dimension
        num_directions = 2 if self.bidirectional else 1
        self.output_dim = self.hidden_size * num_directions
        
        # Initialize hidden state
        self._hidden = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply one RNN step."""
        # x: (input_dim,) -> (1, 1, input_dim)
        x_batch = x.unsqueeze(0).unsqueeze(0)
        
        output, self._hidden = self._rnn(x_batch, self._hidden)
        result = output.squeeze()  # (output_dim,)
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run RNN on full sequence."""
        # x: (seq, input_dim) -> (1, seq, input_dim)
        x_batch = x.unsqueeze(0)
        
        output, self._hidden = self._rnn(x_batch, self._hidden)
        result = output.squeeze(0)  # (seq, output_dim)
        
        return {"out": result[-1]}, result
    
    def reset(self) -> TorchState:
        """Reset hidden state."""
        previous_state = self.state
        num_directions = 2 if self.bidirectional else 1
        self._hidden = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        return previous_state


class LSTMNode(TorchNode):
    """Long Short-Term Memory (LSTM) node.
    
    Implements LSTM with cell state and hidden state:
    - Forget gate: f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)
    - Input gate: i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)
    - Cell candidate: g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)
    - Cell state: c_t = f_t * c_{t-1} + i_t * g_t
    - Output gate: o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)
    - Hidden state: h_t = o_t * tanh(c_t)
    
    Parameters
    ----------
    hidden_size : int
        Number of hidden units
    input_dim : int, optional
        Input dimension (inferred if not provided)
    bias : bool, default True
        Whether to include bias
    num_layers : int, default 1
        Number of stacked LSTM layers
    dropout : float, default 0.0
        Dropout probability between layers
    bidirectional : bool, default False
        Whether to use bidirectional LSTM
    proj_size : int, default 0
        Size of projection layer (0 for no projection)
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> lstm = LSTMNode(hidden_size=128, num_layers=2)
    >>> output = lstm.run(sequence_data)
    """
    
    def __init__(
        self,
        hidden_size: int,
        input_dim: Optional[int] = None,
        bias: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.use_bias = bias
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        
        self._lstm: Optional[nn.LSTM] = None
        self._hidden: Optional[Tuple[Tensor, Tensor]] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the LSTM."""
        self._set_input_dim(x)
        
        self._lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.use_bias,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            proj_size=self.proj_size,
            device=self.device,
            dtype=self.dtype
        )
        
        # Output dimension
        num_directions = 2 if self.bidirectional else 1
        h_out_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        self.output_dim = h_out_size * num_directions
        
        # Initialize hidden and cell states
        h0 = torch.zeros(
            self.num_layers * num_directions, 1, h_out_size,
            device=self.device, dtype=self.dtype
        )
        c0 = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        self._hidden = (h0, c0)
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "cell": torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype)
        }
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply one LSTM step."""
        x_batch = x.unsqueeze(0).unsqueeze(0)
        
        output, self._hidden = self._lstm(x_batch, self._hidden)
        h_out = output.squeeze()
        c_out = self._hidden[1][-1].squeeze()
        
        return {"out": h_out, "cell": c_out}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run LSTM on full sequence."""
        x_batch = x.unsqueeze(0)
        
        output, self._hidden = self._lstm(x_batch, self._hidden)
        result = output.squeeze(0)
        
        return {
            "out": result[-1],
            "cell": self._hidden[1][-1].squeeze()
        }, result
    
    def reset(self) -> TorchState:
        """Reset hidden and cell states."""
        previous_state = self.state
        num_directions = 2 if self.bidirectional else 1
        h_out_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        
        h0 = torch.zeros(
            self.num_layers * num_directions, 1, h_out_size,
            device=self.device, dtype=self.dtype
        )
        c0 = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        self._hidden = (h0, c0)
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "cell": torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype)
        }
        return previous_state


class GRUNode(TorchNode):
    """Gated Recurrent Unit (GRU) node.
    
    Implements GRU:
    - Reset gate: r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)
    - Update gate: z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)
    - Candidate: n_t = tanh(W_in @ x_t + r_t * (W_hn @ h_{t-1} + b_hn))
    - Hidden: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    
    Parameters
    ----------
    hidden_size : int
        Number of hidden units
    input_dim : int, optional
        Input dimension (inferred if not provided)
    bias : bool, default True
        Whether to include bias
    num_layers : int, default 1
        Number of stacked GRU layers
    dropout : float, default 0.0
        Dropout probability between layers
    bidirectional : bool, default False
        Whether to use bidirectional GRU
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
    """
    
    def __init__(
        self,
        hidden_size: int,
        input_dim: Optional[int] = None,
        bias: bool = True,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.use_bias = bias
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self._gru: Optional[nn.GRU] = None
        self._hidden: Optional[Tensor] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the GRU."""
        self._set_input_dim(x)
        
        self._gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.use_bias,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            device=self.device,
            dtype=self.dtype
        )
        
        num_directions = 2 if self.bidirectional else 1
        self.output_dim = self.hidden_size * num_directions
        
        self._hidden = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply one GRU step."""
        x_batch = x.unsqueeze(0).unsqueeze(0)
        
        output, self._hidden = self._gru(x_batch, self._hidden)
        result = output.squeeze()
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run GRU on full sequence."""
        x_batch = x.unsqueeze(0)
        
        output, self._hidden = self._gru(x_batch, self._hidden)
        result = output.squeeze(0)
        
        return {"out": result[-1]}, result
    
    def reset(self) -> TorchState:
        """Reset hidden state."""
        previous_state = self.state
        num_directions = 2 if self.bidirectional else 1
        self._hidden = torch.zeros(
            self.num_layers * num_directions, 1, self.hidden_size,
            device=self.device, dtype=self.dtype
        )
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        return previous_state


class ESNTorchNode(TorchTrainableNode):
    """Echo State Network with PyTorch backend.
    
    Implements the classic Echo State Network with PyTorch tensors,
    enabling GPU acceleration and integration with deep learning.
    
    The reservoir dynamics follow:
    x[t+1] = (1 - lr) * x[t] + lr * f(W_in @ u[t+1] + W @ x[t] + b)
    
    Parameters
    ----------
    units : int
        Number of reservoir units
    lr : float, default 1.0
        Leaking rate (1.0 = no leaking)
    sr : float, default 0.9
        Spectral radius of recurrent weights
    input_scaling : float, default 1.0
        Scaling factor for input weights
    input_connectivity : float, default 0.1
        Connectivity of input weights
    rc_connectivity : float, default 0.1
        Connectivity of recurrent weights
    activation : str or callable, default 'tanh'
        Activation function
    bias : float, default 0.0
        Bias value (0.0 for no bias)
    ridge : float, default 1e-6
        Ridge regression parameter for readout training
    input_dim : int, optional
        Input dimension (inferred if not provided)
    output_dim : int, optional
        Output dimension (inferred from training data)
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    seed : int, optional
        Random seed for reproducibility
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> # Create ESN
    >>> esn = ESNTorchNode(units=500, sr=0.9, lr=0.3)
    >>> 
    >>> # Run reservoir
    >>> states = esn.run(X_train)
    >>> 
    >>> # Train readout
    >>> esn.fit(X_train, y_train)
    >>> 
    >>> # Predict
    >>> predictions = esn.run(X_test)
    """
    
    def __init__(
        self,
        units: int,
        lr: float = 1.0,
        sr: float = 0.9,
        input_scaling: float = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        activation: Union[str, Callable] = 'tanh',
        bias: float = 0.0,
        ridge: float = 1e-6,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.units = units
        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.bias_value = bias
        self.ridge = ridge
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.seed = seed
        
        # Parse activation
        if isinstance(activation, str):
            self.activation = self._get_activation(activation)
        else:
            self.activation = activation
        
        # Weights (initialized later)
        self.W_in: Optional[Tensor] = None
        self.W: Optional[Tensor] = None
        self.W_out: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            'tanh': torch.tanh,
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'identity': lambda x: x,
        }
        return activations.get(name.lower(), torch.tanh)
    
    def _create_sparse_matrix(
        self, 
        rows: int, 
        cols: int, 
        connectivity: float,
        generator: torch.Generator
    ) -> Tensor:
        """Create sparse random matrix."""
        mask = torch.rand(rows, cols, generator=generator, device=self.device, dtype=self.dtype) < connectivity
        values = torch.randn(rows, cols, generator=generator, device=self.device, dtype=self.dtype)
        return values * mask
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize ESN weights."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Set up random generator
        generator = torch.Generator(device=self.device)
        if self.seed is not None:
            generator.manual_seed(self.seed)
        
        # Initialize input weights
        self.W_in = self._create_sparse_matrix(
            self.units, self.input_dim, 
            self.input_connectivity, generator
        ) * self.input_scaling
        self.register_buffer("_W_in", self.W_in)
        
        # Initialize recurrent weights
        W = self._create_sparse_matrix(
            self.units, self.units,
            self.rc_connectivity, generator
        )
        
        # Scale to spectral radius
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W)
            current_sr = torch.max(torch.abs(eigenvalues)).real
            if current_sr > 0:
                W = W * (self.sr / current_sr)
        
        self.W = W
        self.register_buffer("_W", self.W)
        
        # Initialize bias
        if self.bias_value != 0:
            self.bias = torch.full(
                (self.units,), self.bias_value,
                device=self.device, dtype=self.dtype
            )
        else:
            self.bias = torch.zeros(self.units, device=self.device, dtype=self.dtype)
        self.register_buffer("_bias", self.bias)
        
        # Output weights (initialized during training)
        if self._output_dim is not None:
            self.W_out = torch.zeros(
                self._output_dim, self.units,
                device=self.device, dtype=self.dtype
            )
            self.register_parameter("_W_out", self.W_out)
            self.output_dim = self._output_dim
        else:
            self.output_dim = self.units
        
        # Initialize state
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.units, device=self.device, dtype=self.dtype)
        }
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Compute one ESN step."""
        s = state["reservoir"]
        
        # Reservoir update
        pre_activation = self.W_in @ x + self.W @ s + self.bias
        new_s = self.activation(pre_activation)
        new_s = (1 - self.lr) * s + self.lr * new_s
        
        # Output
        if self.W_out is not None:
            out = self.W_out @ new_s
        else:
            out = new_s
        
        return {"out": out, "reservoir": new_s}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run ESN on full sequence."""
        n_timesteps = x.shape[0]
        states = torch.zeros(n_timesteps, self.units, device=self.device, dtype=self.dtype)
        
        s = state["reservoir"]
        for t in range(n_timesteps):
            pre_activation = self.W_in @ x[t] + self.W @ s + self.bias
            new_s = self.activation(pre_activation)
            s = (1 - self.lr) * s + self.lr * new_s
            states[t] = s
        
        # Output
        if self.W_out is not None:
            outputs = states @ self.W_out.T
        else:
            outputs = states
        
        return {"out": outputs[-1], "reservoir": s}, outputs
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "ESNTorchNode":
        """Train ESN readout with ridge regression.
        
        Parameters
        ----------
        x : array-like or Tensor
            Input data
        y : array-like or Tensor
            Target data
        warmup : int
            Number of timesteps to discard
        ridge : float, optional
            Ridge parameter (overrides default)
            
        Returns
        -------
        ESNTorchNode
            Self (for chaining)
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        if y is None:
            return self
        
        # Set output dimension
        self._output_dim = y.shape[-1]
        self.output_dim = self._output_dim
        
        # Collect reservoir states
        _, states = self._run(self.state, x)
        
        # Apply warmup
        states_train = states[warmup:]
        y_train = y[warmup:]
        
        # Ridge regression
        ridge_param = ridge or self.ridge
        
        # S: (n_samples, units), Y: (n_samples, output_dim)
        StS = states_train.T @ states_train
        StY = states_train.T @ y_train
        
        reg = ridge_param * torch.eye(self.units, device=self.device, dtype=self.dtype)
        self.W_out = torch.linalg.solve(StS + reg, StY).T
        self.register_parameter("_W_out", self.W_out)
        
        return self
    
    def reset(self) -> TorchState:
        """Reset reservoir state."""
        previous_state = self.state
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.units, device=self.device, dtype=self.dtype)
        }
        return previous_state
