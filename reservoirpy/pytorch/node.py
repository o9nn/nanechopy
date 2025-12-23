"""
====================================
PyTorch Node API (:mod:`reservoirpy.pytorch.node`)
====================================

Base classes for PyTorch-integrated reservoir computing nodes.
Provides seamless integration between ReservoirPy's Node API and PyTorch tensors.

The TorchNode hierarchy mirrors the standard Node hierarchy:

```
TorchNode
╠══ TorchTrainableNode
║   ╠══ TorchParallelNode
║   ╚══ TorchOnlineNode
```

All TorchNodes operate on PyTorch tensors and support:
- Automatic differentiation via autograd
- GPU acceleration via CUDA
- Mixed precision training
- Integration with PyTorch optimizers
"""

# License: MIT License
# Copyright: nanechopy contributors

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, Optional, Sequence, Union, Any, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    # Create placeholder for type hints when torch not available
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()

from ..node import Node, TrainableNode, OnlineNode, ParallelNode
from ..type import NodeInput, State, Timeseries, Timestep


# Type aliases for PyTorch
TorchState = Dict[str, Tensor]
TorchTimestep = Tensor  # Shape: (input_dim,) or (batch, input_dim)
TorchTimeseries = Tensor  # Shape: (timesteps, input_dim) or (batch, timesteps, input_dim)


def _check_torch_available():
    """Raise ImportError if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TorchNode. "
            "Install it with: pip install torch"
        )


def to_tensor(x: Union[np.ndarray, Tensor, None], 
              device: Optional[str] = None,
              dtype: Optional[torch.dtype] = None) -> Optional[Tensor]:
    """Convert numpy array to PyTorch tensor."""
    if x is None:
        return None
    if isinstance(x, Tensor):
        tensor = x
    else:
        tensor = torch.from_numpy(np.asarray(x))
    
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def to_numpy(x: Union[np.ndarray, Tensor, None]) -> Optional[np.ndarray]:
    """Convert PyTorch tensor to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


class TorchNode(ABC):
    """Base class for PyTorch-integrated nodes.
    
    TorchNode provides the foundation for creating reservoir computing
    nodes that operate on PyTorch tensors. It supports:
    
    - Automatic device placement (CPU/GPU)
    - Mixed precision training (float16/float32/float64)
    - Gradient tracking for backpropagation
    - Seamless conversion between numpy and PyTorch
    
    Parameters
    ----------
    device : str, optional
        Device to place tensors on ('cpu', 'cuda', 'cuda:0', etc.)
    dtype : torch.dtype, optional
        Data type for tensors (torch.float32, torch.float64, etc.)
    requires_grad : bool, default False
        Whether to track gradients for parameters
    name : str, optional
        Name of the node
        
    Attributes
    ----------
    initialized : bool
        True if the node has been initialized
    input_dim : int
        Expected dimension of input
    output_dim : int
        Expected dimension of output
    state : TorchState
        Current state of the node (dict with 'out' key)
    device : str
        Device where tensors are placed
    dtype : torch.dtype
        Data type of tensors
    """
    
    initialized: bool = False
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    state: TorchState
    name: Optional[str] = None
    
    def __init__(
        self,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ):
        _check_torch_available()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or torch.float32
        self.requires_grad = requires_grad
        self.name = name
        self._parameters: Dict[str, Tensor] = {}
        self._buffers: Dict[str, Tensor] = {}
    
    def register_parameter(self, name: str, tensor: Tensor) -> None:
        """Register a trainable parameter."""
        if self.requires_grad:
            tensor = tensor.requires_grad_(True)
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        self._parameters[name] = tensor
        setattr(self, name, tensor)
    
    def register_buffer(self, name: str, tensor: Tensor) -> None:
        """Register a non-trainable buffer."""
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        self._buffers[name] = tensor
        setattr(self, name, tensor)
    
    def parameters(self) -> Iterable[Tensor]:
        """Return iterator over trainable parameters."""
        return self._parameters.values()
    
    def named_parameters(self) -> Iterable[Tuple[str, Tensor]]:
        """Return iterator over named parameters."""
        return self._parameters.items()
    
    def to(self, device: Optional[str] = None, dtype: Optional[torch.dtype] = None) -> "TorchNode":
        """Move node to specified device and/or dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        # Move parameters
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device=self.device, dtype=self.dtype)
            setattr(self, name, self._parameters[name])
        
        # Move buffers
        for name, buf in self._buffers.items():
            self._buffers[name] = buf.to(device=self.device, dtype=self.dtype)
            setattr(self, name, self._buffers[name])
        
        # Move state
        if hasattr(self, 'state') and self.state is not None:
            self.state = {
                k: v.to(device=self.device, dtype=self.dtype) if isinstance(v, Tensor) else v
                for k, v in self.state.items()
            }
        
        return self
    
    @abstractmethod
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize node dimensions and parameters.
        
        Parameters
        ----------
        x : Tensor
            Input data for dimension inference
        y : Tensor, optional
            Target data for output dimension inference
        """
        ...
    
    @abstractmethod
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Compute one step of the node.
        
        Parameters
        ----------
        state : TorchState
            Current state of the node
        x : TorchTimestep
            Input tensor for this timestep
            
        Returns
        -------
        TorchState
            New state of the node (must include 'out' key)
        """
        ...
    
    def step(self, x: Optional[Union[np.ndarray, TorchTimestep]] = None) -> TorchTimestep:
        """Call the node on a single timestep.
        
        Parameters
        ----------
        x : array-like or Tensor, optional
            Input for this timestep
            
        Returns
        -------
        Tensor
            Output tensor
        """
        if x is None:
            x = torch.empty((0,), device=self.device, dtype=self.dtype)
        else:
            x = to_tensor(x, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x)
        
        new_state = self._step(self.state, x)
        self.state = new_state
        return new_state["out"]
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run node on a timeseries (default implementation)."""
        n_timesteps = x.shape[0]
        outputs = []
        current_state = state
        
        for i in range(n_timesteps):
            current_state = self._step(current_state, x[i])
            outputs.append(current_state["out"])
        
        output = torch.stack(outputs, dim=0)
        return current_state, output
    
    def run(
        self, 
        x: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        iters: Optional[int] = None,
    ) -> TorchTimeseries:
        """Run the node on a sequence of data.
        
        Parameters
        ----------
        x : array-like or Tensor, optional
            Input timeseries of shape (timesteps, input_dim)
        iters : int, optional
            If x is None, run for this many iterations
            
        Returns
        -------
        Tensor
            Output timeseries of shape (timesteps, output_dim)
        """
        if x is None:
            x = torch.empty((iters, 0), device=self.device, dtype=self.dtype)
        else:
            x = to_tensor(x, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x)
        
        final_state, result = self._run(self.state, x)
        self.state = final_state
        return result
    
    def predict(
        self, 
        x: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        iters: Optional[int] = None,
    ) -> TorchTimeseries:
        """Alias for run()."""
        return self.run(x=x, iters=iters)
    
    def reset(self) -> TorchState:
        """Reset node state to zeros."""
        previous_state = self.state
        self.state = {
            key: torch.zeros_like(val) 
            for key, val in self.state.items()
        }
        return previous_state
    
    def _set_input_dim(self, x: Optional[Union[TorchTimestep, TorchTimeseries]]) -> None:
        """Set input dimension from data."""
        if x is None:
            return
        input_dim = x.shape[-1]
        if self.input_dim is not None and self.input_dim != input_dim:
            raise ValueError(
                f"Trying to set input_dim to {input_dim} but it's already {self.input_dim}"
            )
        self.input_dim = input_dim
    
    def _set_output_dim(self, y: Optional[Union[TorchTimestep, TorchTimeseries]]) -> None:
        """Set output dimension from data."""
        if y is None:
            return
        output_dim = y.shape[-1]
        if self.output_dim is not None and self.output_dim != output_dim:
            raise ValueError(
                f"Trying to set output_dim to {output_dim} but it's already {self.output_dim}"
            )
        self.output_dim = output_dim
    
    def __call__(self, x: Optional[TorchTimestep] = None) -> TorchTimestep:
        return self.step(x)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __rshift__(self, other):
        """Link nodes with >> operator."""
        from .ops import link
        return link(self, other)
    
    def __and__(self, other):
        """Merge nodes with & operator."""
        from .ops import merge
        return merge(self, other)


class TorchTrainableNode(TorchNode):
    """TorchNode that can be trained with fit().
    
    Extends TorchNode with training capabilities including:
    - Offline batch training via fit()
    - Integration with PyTorch optimizers
    - Loss function support
    """
    
    @abstractmethod
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        **kwargs,
    ) -> "TorchTrainableNode":
        """Train the node on data.
        
        Parameters
        ----------
        x : array-like or Tensor
            Input data
        y : array-like or Tensor, optional
            Target data
        warmup : int
            Number of timesteps to skip at start
            
        Returns
        -------
        TorchTrainableNode
            Self (for chaining)
        """
        ...


class TorchOnlineNode(TorchTrainableNode):
    """TorchNode that can be trained online.
    
    Extends TorchTrainableNode with online learning:
    - Incremental updates via partial_fit()
    - Per-timestep learning via _learning_step()
    """
    
    @abstractmethod
    def _learning_step(
        self, 
        x: TorchTimestep, 
        y: Optional[TorchTimestep]
    ) -> TorchTimestep:
        """Perform one learning step.
        
        Parameters
        ----------
        x : Tensor
            Input for this timestep
        y : Tensor, optional
            Target for this timestep
            
        Returns
        -------
        Tensor
            Prediction for this timestep
        """
        ...
    
    def partial_fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
    ) -> TorchTimeseries:
        """Fit incrementally on a timeseries.
        
        Parameters
        ----------
        x : array-like or Tensor
            Input timeseries
        y : array-like or Tensor, optional
            Target timeseries
            
        Returns
        -------
        Tensor
            Predictions during training
        """
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        n_timesteps = x.shape[0]
        predictions = []
        
        for i in range(n_timesteps):
            y_i = y[i] if y is not None else None
            pred = self._learning_step(x[i], y_i)
            predictions.append(pred)
        
        output = torch.stack(predictions, dim=0)
        self.state = {"out": predictions[-1]}
        return output
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        **kwargs,
    ) -> "TorchOnlineNode":
        """Fit by calling partial_fit on data."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        self.initialize(x, y)
        
        # Handle warmup
        x_train = x[warmup:]
        y_train = y[warmup:] if y is not None else None
        
        self.partial_fit(x_train, y_train)
        return self


class TorchParallelNode(TorchTrainableNode):
    """TorchNode that can be trained in parallel.
    
    Implements worker/master pattern for parallel training.
    """
    
    @abstractmethod
    def worker(
        self, 
        x: TorchTimeseries, 
        y: Optional[TorchTimeseries] = None
    ) -> Any:
        """Process one timeseries (called in parallel)."""
        ...
    
    @abstractmethod
    def master(self, results: Iterable) -> None:
        """Aggregate worker results."""
        ...
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        workers: int = 1,
        **kwargs,
    ) -> "TorchParallelNode":
        """Fit with optional parallel processing."""
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # For now, simple sequential processing
        # TODO: Add DataParallel support
        x_train = x[warmup:]
        y_train = y[warmup:] if y is not None else None
        
        result = self.worker(x_train, y_train)
        self.master([result])
        
        return self
