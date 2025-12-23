"""
====================================
Autognosis Node (:mod:`reservoirpy.pytorch.autognosis.node`)
====================================

Autognosis node wrapping the engine as a ReservoirPy-compatible node.

The AutognosisNode integrates the AutognosisEngine into the ReservoirPy
node framework, enabling seamless use with other nodes and models.
"""

# License: MIT License
# Copyright: nanechopy contributors

from typing import Callable, Dict, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()

from ..node import TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
from .engine import AutognosisEngine, AutognosisConfig
from .a000081 import A000081Parameters, derive_parameters


class AutognosisNode(TorchTrainableNode):
    """Autognosis node for self-aware reservoir computing.
    
    Wraps the AutognosisEngine as a ReservoirPy-compatible node,
    enabling integration with other nodes and models.
    
    Parameters
    ----------
    base_order : int, default 5
        Base order for A000081 parameter derivation
    enable_evolution : bool, default True
        Whether to enable ontogenetic evolution
    evolution_interval : int, default 100
        Steps between evolution updates
    bseries_order : int, default 3
        Maximum order for B-series integration
    feedback_strength : float, default 0.0
        Strength of feedback connections
    ridge : float, optional
        Ridge parameter (defaults to A000081-derived)
    input_dim : int, optional
        Input dimension (inferred if not provided)
    output_dim : int, optional
        Output dimension (inferred from training)
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    seed : int, optional
        Random seed
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> # Create autognosis node
    >>> node = AutognosisNode(base_order=5)
    >>> 
    >>> # Run on data
    >>> states = node.run(X_train)
    >>> 
    >>> # Train readout
    >>> node.fit(X_train, y_train)
    >>> 
    >>> # Predict
    >>> predictions = node.run(X_test)
    """
    
    def __init__(
        self,
        base_order: int = 5,
        enable_evolution: bool = True,
        evolution_interval: int = 100,
        bseries_order: int = 3,
        feedback_strength: float = 0.0,
        ridge: Optional[float] = None,
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
        self.evolution_interval = evolution_interval
        self.bseries_order = bseries_order
        self.feedback_strength = feedback_strength
        self.ridge = ridge
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.seed = seed
        
        # Derive parameters
        self.params = derive_parameters(base_order)
        
        # Engine (initialized later)
        self._engine: Optional[AutognosisEngine] = None
        
        # Set output_dim from reservoir size
        self.output_dim = output_dim or self.params.reservoir_size
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the autognosis node."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Create config
        config = AutognosisConfig(
            base_order=self.base_order,
            enable_evolution=self.enable_evolution,
            evolution_interval=self.evolution_interval,
            bseries_order=self.bseries_order,
            feedback_strength=self.feedback_strength,
        )
        
        # Create engine
        self._engine = AutognosisEngine(
            config=config,
            device=self.device,
            seed=self.seed,
        )
        self._engine.initialize(self.input_dim, self._output_dim)
        
        # Update output_dim
        if self._output_dim is not None:
            self.output_dim = self._output_dim
        else:
            self.output_dim = self.params.reservoir_size
        
        # Initialize state
        if TORCH_AVAILABLE:
            self.state = {
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "reservoir": torch.zeros(self.params.reservoir_size, device=self.device, dtype=self.dtype)
            }
        else:
            self.state = {
                "out": np.zeros(self.output_dim),
                "reservoir": np.zeros(self.params.reservoir_size)
            }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Compute one autognosis step."""
        # Convert to numpy for engine
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Step through engine
        reservoir_state = self._engine.step(x_np)
        
        # Compute output
        if self._engine.W_out is not None:
            out = reservoir_state @ self._engine.W_out.T
        else:
            out = reservoir_state
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            out = torch.from_numpy(out).to(device=self.device, dtype=self.dtype)
            reservoir = torch.from_numpy(reservoir_state).to(device=self.device, dtype=self.dtype)
        else:
            reservoir = reservoir_state
        
        return {"out": out, "reservoir": reservoir}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run autognosis on full sequence."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Process through engine
        states = self._engine.process(x_np)
        
        # Compute outputs
        if self._engine.W_out is not None:
            outputs = states @ self._engine.W_out.T
        else:
            outputs = states
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            outputs = torch.from_numpy(outputs).to(device=self.device, dtype=self.dtype)
            final_reservoir = torch.from_numpy(states[-1]).to(device=self.device, dtype=self.dtype)
        else:
            final_reservoir = states[-1]
        
        return {
            "out": outputs[-1] if len(outputs.shape) > 1 else outputs,
            "reservoir": final_reservoir
        }, outputs
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "AutognosisNode":
        """Train the autognosis readout.
        
        Parameters
        ----------
        x : array-like or Tensor
            Input data
        y : array-like or Tensor
            Target data
        warmup : int
            Warmup steps to discard
        ridge : float, optional
            Ridge parameter
            
        Returns
        -------
        AutognosisNode
            Self (for chaining)
        """
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        if y is not None:
            if TORCH_AVAILABLE and isinstance(y, Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = np.asarray(y)
        else:
            y_np = None
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Train engine
        ridge_param = ridge or self.ridge or self.params.ridge_param
        self._engine.fit(x_np, y_np, warmup=warmup, ridge=ridge_param)
        
        # Update output_dim
        if y_np is not None:
            self._output_dim = y_np.shape[-1]
            self.output_dim = self._output_dim
        
        return self
    
    def reset(self) -> TorchState:
        """Reset node state."""
        previous_state = self.state
        
        if self._engine is not None:
            self._engine.reset()
        
        if TORCH_AVAILABLE:
            self.state = {
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "reservoir": torch.zeros(self.params.reservoir_size, device=self.device, dtype=self.dtype)
            }
        else:
            self.state = {
                "out": np.zeros(self.output_dim),
                "reservoir": np.zeros(self.params.reservoir_size)
            }
        
        return previous_state
    
    def get_status(self) -> Dict:
        """Get engine status."""
        if self._engine is not None:
            return self._engine.get_status()
        return {"initialized": False}
    
    def get_parameters(self) -> A000081Parameters:
        """Get A000081-derived parameters."""
        return self.params
    
    def explain_parameters(self) -> str:
        """Get explanation of parameter derivation."""
        return self.params.explain()
