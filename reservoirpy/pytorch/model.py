"""
====================================
PyTorch Models (:mod:`reservoirpy.pytorch.model`)
====================================

Model definitions for PyTorch-based reservoir computing.

This module provides high-level model classes that combine
multiple nodes into complete architectures:

- TorchReservoirModel: Standard reservoir computing model
- HybridESNModel: ESN with deep learning components
- AutognosisModel: Self-aware model with echo-jnn integration
"""

# License: MIT License
# Copyright: nanechopy contributors

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
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

from .node import TorchNode, TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
from .nodes import ESNTorchNode, NNLinearNode, LSTMNode, GRUNode, NNAttentionNode
from .autognosis import AutognosisNode, AutognosisEngine, A000081Parameters, derive_parameters


class TorchReservoirModel(TorchTrainableNode):
    """Standard reservoir computing model with PyTorch backend.
    
    Combines an input layer, reservoir, and readout into a complete
    model for temporal sequence processing.
    
    Parameters
    ----------
    reservoir_size : int
        Number of reservoir units
    input_dim : int, optional
        Input dimension (inferred if not provided)
    output_dim : int, optional
        Output dimension (inferred from training)
    lr : float, default 0.3
        Leaking rate
    sr : float, default 0.9
        Spectral radius
    input_scaling : float, default 1.0
        Input weight scaling
    ridge : float, default 1e-6
        Ridge regression parameter
    activation : str, default 'tanh'
        Reservoir activation function
    input_layer : TorchNode, optional
        Custom input preprocessing layer
    readout_layer : TorchNode, optional
        Custom readout layer
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    seed : int, optional
        Random seed
    name : str, optional
        Model name
        
    Examples
    --------
    >>> # Create model
    >>> model = TorchReservoirModel(reservoir_size=500, lr=0.3, sr=0.9)
    >>> 
    >>> # Train
    >>> model.fit(X_train, y_train, warmup=100)
    >>> 
    >>> # Predict
    >>> predictions = model.run(X_test)
    """
    
    def __init__(
        self,
        reservoir_size: int,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        lr: float = 0.3,
        sr: float = 0.9,
        input_scaling: float = 1.0,
        ridge: float = 1e-6,
        activation: str = 'tanh',
        input_layer: Optional[TorchNode] = None,
        readout_layer: Optional[TorchNode] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.reservoir_size = reservoir_size
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.ridge = ridge
        self.activation = activation
        self.seed = seed
        
        # Layers
        self.input_layer = input_layer
        self.readout_layer = readout_layer
        
        # Reservoir (created on initialize)
        self._reservoir: Optional[ESNTorchNode] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the model."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Initialize input layer if provided
        reservoir_input_dim = self.input_dim
        if self.input_layer is not None:
            self.input_layer.initialize(x)
            reservoir_input_dim = self.input_layer.output_dim
        
        # Create reservoir
        self._reservoir = ESNTorchNode(
            units=self.reservoir_size,
            lr=self.lr,
            sr=self.sr,
            input_scaling=self.input_scaling,
            ridge=self.ridge,
            activation=self.activation,
            input_dim=reservoir_input_dim,
            output_dim=self._output_dim,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
        )
        
        # Initialize reservoir with dummy data
        dummy = torch.zeros(reservoir_input_dim, device=self.device, dtype=self.dtype)
        self._reservoir.initialize(dummy, y)
        
        # Set output dimension
        self.output_dim = self._output_dim or self.reservoir_size
        
        # Initialize state
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.reservoir_size, device=self.device, dtype=self.dtype)
        }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep."""
        # Input layer
        if self.input_layer is not None:
            x = self.input_layer.step(x)
        
        # Reservoir
        reservoir_state = self._reservoir._step(self._reservoir.state, x)
        self._reservoir.state = reservoir_state
        
        # Output
        out = reservoir_state["out"]
        
        # Readout layer
        if self.readout_layer is not None:
            out = self.readout_layer.step(out)
        
        return {"out": out, "reservoir": reservoir_state["reservoir"]}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence."""
        # Input layer
        if self.input_layer is not None:
            _, x = self.input_layer._run(self.input_layer.state, x)
        
        # Reservoir
        final_state, outputs = self._reservoir._run(self._reservoir.state, x)
        self._reservoir.state = final_state
        
        # Readout layer
        if self.readout_layer is not None:
            _, outputs = self.readout_layer._run(self.readout_layer.state, outputs)
        
        return {"out": outputs[-1], "reservoir": final_state["reservoir"]}, outputs
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "TorchReservoirModel":
        """Train the model."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Process through input layer
        if self.input_layer is not None:
            _, x_processed = self.input_layer._run(self.input_layer.state, x)
        else:
            x_processed = x
        
        # Train reservoir
        self._reservoir.fit(x_processed, y, warmup=warmup, ridge=ridge or self.ridge)
        
        # Update output dimension
        if y is not None:
            self._output_dim = y.shape[-1]
            self.output_dim = self._output_dim
        
        return self
    
    def reset(self) -> TorchState:
        """Reset model state."""
        previous_state = self.state
        
        if self._reservoir is not None:
            self._reservoir.reset()
        if self.input_layer is not None:
            self.input_layer.reset()
        if self.readout_layer is not None:
            self.readout_layer.reset()
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.reservoir_size, device=self.device, dtype=self.dtype)
        }
        
        return previous_state


class HybridESNModel(TorchTrainableNode):
    """Hybrid model combining ESN with deep learning components.
    
    This model integrates Echo State Networks with LSTM/GRU layers
    and attention mechanisms for enhanced sequence modeling.
    
    Parameters
    ----------
    reservoir_size : int
        Number of ESN reservoir units
    hidden_size : int
        Size of RNN hidden state
    rnn_type : str, default 'lstm'
        Type of RNN ('lstm', 'gru', 'rnn')
    num_rnn_layers : int, default 1
        Number of RNN layers
    use_attention : bool, default False
        Whether to use attention mechanism
    attention_heads : int, default 4
        Number of attention heads
    lr : float, default 0.3
        ESN leaking rate
    sr : float, default 0.9
        ESN spectral radius
    ridge : float, default 1e-6
        Ridge regression parameter
    dropout : float, default 0.0
        Dropout probability
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    seed : int, optional
        Random seed
    name : str, optional
        Model name
        
    Examples
    --------
    >>> # Create hybrid model
    >>> model = HybridESNModel(
    ...     reservoir_size=500,
    ...     hidden_size=128,
    ...     rnn_type='lstm',
    ...     use_attention=True
    ... )
    >>> 
    >>> # Train
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Predict
    >>> predictions = model.run(X_test)
    """
    
    def __init__(
        self,
        reservoir_size: int,
        hidden_size: int,
        rnn_type: str = 'lstm',
        num_rnn_layers: int = 1,
        use_attention: bool = False,
        attention_heads: int = 4,
        lr: float = 0.3,
        sr: float = 0.9,
        ridge: float = 1e-6,
        dropout: float = 0.0,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.reservoir_size = reservoir_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.num_rnn_layers = num_rnn_layers
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.lr = lr
        self.sr = sr
        self.ridge = ridge
        self.dropout = dropout
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.seed = seed
        
        # Components (created on initialize)
        self._esn: Optional[ESNTorchNode] = None
        self._rnn: Optional[Union[LSTMNode, GRUNode]] = None
        self._attention: Optional[NNAttentionNode] = None
        self._readout: Optional[NNLinearNode] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the model."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Create ESN
        self._esn = ESNTorchNode(
            units=self.reservoir_size,
            lr=self.lr,
            sr=self.sr,
            ridge=self.ridge,
            input_dim=self.input_dim,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
        )
        dummy = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)
        self._esn.initialize(dummy)
        
        # Create RNN
        if self.rnn_type == 'lstm':
            self._rnn = LSTMNode(
                hidden_size=self.hidden_size,
                input_dim=self.reservoir_size,
                num_layers=self.num_rnn_layers,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype,
            )
        elif self.rnn_type == 'gru':
            self._rnn = GRUNode(
                hidden_size=self.hidden_size,
                input_dim=self.reservoir_size,
                num_layers=self.num_rnn_layers,
                dropout=self.dropout,
                device=self.device,
                dtype=self.dtype,
            )
        
        dummy_esn = torch.zeros(self.reservoir_size, device=self.device, dtype=self.dtype)
        self._rnn.initialize(dummy_esn)
        
        # Create attention if enabled
        combined_dim = self.hidden_size
        if self.use_attention:
            self._attention = NNAttentionNode(
                embed_dim=self.hidden_size,
                num_heads=self.attention_heads,
                device=self.device,
                dtype=self.dtype,
            )
            dummy_rnn = torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype)
            self._attention.initialize(dummy_rnn)
        
        # Create readout
        self._readout = NNLinearNode(
            units=self._output_dim or self.hidden_size,
            input_dim=combined_dim,
            ridge=self.ridge,
            device=self.device,
            dtype=self.dtype,
        )
        dummy_combined = torch.zeros(combined_dim, device=self.device, dtype=self.dtype)
        self._readout.initialize(dummy_combined)
        
        self.output_dim = self._output_dim or self.hidden_size
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.reservoir_size, device=self.device, dtype=self.dtype),
            "rnn": torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype),
        }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep."""
        # ESN
        esn_state = self._esn._step(self._esn.state, x)
        self._esn.state = esn_state
        esn_out = esn_state["reservoir"]
        
        # RNN
        rnn_state = self._rnn._step(self._rnn.state, esn_out)
        self._rnn.state = rnn_state
        rnn_out = rnn_state["out"]
        
        # Attention (if enabled)
        if self._attention is not None:
            attn_state = self._attention._step(self._attention.state, rnn_out)
            self._attention.state = attn_state
            combined = attn_state["out"]
        else:
            combined = rnn_out
        
        # Readout
        out = self._readout.step(combined)
        
        return {
            "out": out,
            "reservoir": esn_out,
            "rnn": rnn_out,
        }
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "HybridESNModel":
        """Train the model."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Collect features through ESN and RNN
        n_steps = x.shape[0]
        features = []
        
        for t in range(n_steps):
            state = self._step(self.state, x[t])
            self.state = state
            
            if self._attention is not None:
                features.append(self._attention.state["out"])
            else:
                features.append(state["rnn"])
        
        features = torch.stack(features, dim=0)
        
        # Train readout
        self._readout.fit(features, y, warmup=warmup, ridge=ridge or self.ridge)
        
        if y is not None:
            self._output_dim = y.shape[-1]
            self.output_dim = self._output_dim
        
        return self
    
    def reset(self) -> TorchState:
        """Reset model state."""
        previous_state = self.state
        
        if self._esn is not None:
            self._esn.reset()
        if self._rnn is not None:
            self._rnn.reset()
        if self._attention is not None:
            self._attention.reset()
        if self._readout is not None:
            self._readout.reset()
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.reservoir_size, device=self.device, dtype=self.dtype),
            "rnn": torch.zeros(self.hidden_size, device=self.device, dtype=self.dtype),
        }
        
        return previous_state


class AutognosisModel(TorchTrainableNode):
    """Self-aware model with echo-jnn autognosis integration.
    
    Combines AutognosisNode with additional processing layers
    for complete self-aware reservoir computing.
    
    Parameters
    ----------
    base_order : int, default 5
        Base order for A000081 parameter derivation
    enable_evolution : bool, default True
        Whether to enable ontogenetic evolution
    use_bseries : bool, default True
        Whether to use B-series integration
    add_preprocessing : bool, default False
        Whether to add input preprocessing
    add_postprocessing : bool, default False
        Whether to add output postprocessing
    preprocessing_units : int, default 64
        Units in preprocessing layer
    postprocessing_units : int, default 64
        Units in postprocessing layer
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    seed : int, optional
        Random seed
    name : str, optional
        Model name
        
    Examples
    --------
    >>> # Create autognosis model
    >>> model = AutognosisModel(base_order=5, enable_evolution=True)
    >>> 
    >>> # Train
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Predict
    >>> predictions = model.run(X_test)
    >>> 
    >>> # Get status
    >>> status = model.get_status()
    """
    
    def __init__(
        self,
        base_order: int = 5,
        enable_evolution: bool = True,
        use_bseries: bool = True,
        add_preprocessing: bool = False,
        add_postprocessing: bool = False,
        preprocessing_units: int = 64,
        postprocessing_units: int = 64,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.base_order = base_order
        self.enable_evolution = enable_evolution
        self.use_bseries = use_bseries
        self.add_preprocessing = add_preprocessing
        self.add_postprocessing = add_postprocessing
        self.preprocessing_units = preprocessing_units
        self.postprocessing_units = postprocessing_units
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.seed = seed
        
        # Get A000081 parameters
        self.params = derive_parameters(base_order)
        
        # Components (created on initialize)
        self._preprocessor: Optional[NNLinearNode] = None
        self._autognosis: Optional[AutognosisNode] = None
        self._postprocessor: Optional[NNLinearNode] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the model."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Create preprocessor if enabled
        autognosis_input_dim = self.input_dim
        if self.add_preprocessing:
            self._preprocessor = NNLinearNode(
                units=self.preprocessing_units,
                input_dim=self.input_dim,
                activation='tanh',
                device=self.device,
                dtype=self.dtype,
            )
            dummy = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)
            self._preprocessor.initialize(dummy)
            autognosis_input_dim = self.preprocessing_units
        
        # Create autognosis node
        self._autognosis = AutognosisNode(
            base_order=self.base_order,
            enable_evolution=self.enable_evolution,
            bseries_order=3 if self.use_bseries else 1,
            input_dim=autognosis_input_dim,
            output_dim=self._output_dim,
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
        )
        dummy = torch.zeros(autognosis_input_dim, device=self.device, dtype=self.dtype)
        self._autognosis.initialize(dummy, y)
        
        # Create postprocessor if enabled
        if self.add_postprocessing and self._output_dim is not None:
            self._postprocessor = NNLinearNode(
                units=self._output_dim,
                input_dim=self._autognosis.output_dim,
                activation=None,
                device=self.device,
                dtype=self.dtype,
            )
            dummy = torch.zeros(self._autognosis.output_dim, device=self.device, dtype=self.dtype)
            self._postprocessor.initialize(dummy)
        
        # Set output dimension
        if self._postprocessor is not None:
            self.output_dim = self._output_dim
        else:
            self.output_dim = self._autognosis.output_dim
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.params.reservoir_size, device=self.device, dtype=self.dtype),
        }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep."""
        # Preprocessing
        if self._preprocessor is not None:
            x = self._preprocessor.step(x)
        
        # Autognosis
        autognosis_state = self._autognosis._step(self._autognosis.state, x)
        self._autognosis.state = autognosis_state
        out = autognosis_state["out"]
        
        # Postprocessing
        if self._postprocessor is not None:
            out = self._postprocessor.step(out)
        
        return {
            "out": out,
            "reservoir": autognosis_state["reservoir"],
        }
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence."""
        # Preprocessing
        if self._preprocessor is not None:
            _, x = self._preprocessor._run(self._preprocessor.state, x)
        
        # Autognosis
        final_state, outputs = self._autognosis._run(self._autognosis.state, x)
        self._autognosis.state = final_state
        
        # Postprocessing
        if self._postprocessor is not None:
            _, outputs = self._postprocessor._run(self._postprocessor.state, outputs)
        
        return {
            "out": outputs[-1],
            "reservoir": final_state["reservoir"],
        }, outputs
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "AutognosisModel":
        """Train the model."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Preprocess if needed
        if self._preprocessor is not None:
            _, x_processed = self._preprocessor._run(self._preprocessor.state, x)
        else:
            x_processed = x
        
        # Train autognosis
        self._autognosis.fit(x_processed, y, warmup=warmup, ridge=ridge)
        
        if y is not None:
            self._output_dim = y.shape[-1]
            self.output_dim = self._output_dim
        
        return self
    
    def reset(self) -> TorchState:
        """Reset model state."""
        previous_state = self.state
        
        if self._preprocessor is not None:
            self._preprocessor.reset()
        if self._autognosis is not None:
            self._autognosis.reset()
        if self._postprocessor is not None:
            self._postprocessor.reset()
        
        self.state = {
            "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            "reservoir": torch.zeros(self.params.reservoir_size, device=self.device, dtype=self.dtype),
        }
        
        return previous_state
    
    def get_status(self) -> Dict:
        """Get model status including autognosis state."""
        status = {
            "initialized": self.initialized,
            "base_order": self.base_order,
            "parameters": self.params.to_dict(),
        }
        
        if self._autognosis is not None:
            status["autognosis"] = self._autognosis.get_status()
        
        return status
    
    def explain_parameters(self) -> str:
        """Get explanation of A000081 parameter derivation."""
        return self.params.explain()
