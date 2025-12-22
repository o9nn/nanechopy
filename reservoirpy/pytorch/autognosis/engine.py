"""
====================================
Autognosis Engine (:mod:`reservoirpy.pytorch.autognosis.engine`)
====================================

Main autognosis engine integrating all components.

The AutognosisEngine provides self-aware learning capabilities by:
1. Deriving parameters from A000081 sequence
2. Using B-series integration for stable dynamics
3. Evolving reservoir topology via ontogenetic engine
4. Adapting to task requirements through feedback

This is the primary interface for echo-jnn integration.
"""

# License: MIT License
# Copyright: nanechopy contributors (adapted from echo-jnn)

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
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
    torch = None

try:
    from .a000081 import A000081Parameters, derive_parameters, get_a000081_value
    from .bseries import BSeriesKernel, RootedTree, generate_trees
    from .ontogenetic import OntogeneticEngine, OntogeneticState, MembraneGarden
except ImportError:
    from a000081 import A000081Parameters, derive_parameters, get_a000081_value
    from bseries import BSeriesKernel, RootedTree, generate_trees
    from ontogenetic import OntogeneticEngine, OntogeneticState, MembraneGarden


@dataclass
class AutognosisConfig:
    """Configuration for AutognosisEngine.
    
    Attributes
    ----------
    base_order : int
        Base order for A000081 parameter derivation
    enable_evolution : bool
        Whether to enable ontogenetic evolution
    evolution_interval : int
        Steps between evolution updates
    bseries_order : int
        Maximum order for B-series integration
    symplectic : bool
        Whether to use symplectic integration
    adapt_topology : bool
        Whether to adapt reservoir topology
    feedback_strength : float
        Strength of feedback connections
    """
    base_order: int = 5
    enable_evolution: bool = True
    evolution_interval: int = 100
    bseries_order: int = 3
    symplectic: bool = True
    adapt_topology: bool = False
    feedback_strength: float = 0.1


class AutognosisEngine:
    """Self-aware learning engine integrating echo-jnn components.
    
    The AutognosisEngine combines:
    - A000081-derived parameters for mathematical grounding
    - B-series integration for numerical stability
    - Ontogenetic evolution for topology adaptation
    - Reservoir computing for temporal learning
    
    Parameters
    ----------
    config : AutognosisConfig, optional
        Configuration object
    base_order : int, optional
        Base order (overrides config)
    device : str, optional
        PyTorch device
    dtype : torch.dtype, optional
        PyTorch data type
    seed : int, optional
        Random seed
        
    Examples
    --------
    >>> # Create engine with default config
    >>> engine = AutognosisEngine(base_order=5)
    >>> 
    >>> # Initialize
    >>> engine.initialize(input_dim=10, output_dim=5)
    >>> 
    >>> # Process data
    >>> states = engine.process(X_train)
    >>> 
    >>> # Train readout
    >>> engine.fit(X_train, y_train)
    >>> 
    >>> # Predict
    >>> predictions = engine.predict(X_test)
    """
    
    def __init__(
        self,
        config: Optional[AutognosisConfig] = None,
        base_order: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,  # torch.dtype when available
        seed: Optional[int] = None,
    ):
        self.config = config or AutognosisConfig()
        if base_order is not None:
            self.config.base_order = base_order
        
        if TORCH_AVAILABLE:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self.dtype = dtype or torch.float32
        else:
            self.device = 'cpu'
            self.dtype = None
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Derive parameters
        self.params = derive_parameters(self.config.base_order)
        
        # Components (initialized later)
        self.ontogenetic: Optional[OntogeneticEngine] = None
        self.bseries: Optional[BSeriesKernel] = None
        
        # Reservoir state
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self.reservoir_state: Optional[np.ndarray] = None
        
        # Weights
        self.W_in: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.W_out: Optional[np.ndarray] = None
        self.W_fb: Optional[np.ndarray] = None
        
        # Training state
        self.step_count: int = 0
        self.is_initialized: bool = False
    
    def initialize(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
    ) -> None:
        """Initialize the engine.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int, optional
            Output dimension (defaults to reservoir size)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim or self.params.reservoir_size
        
        # Initialize components
        if self.config.enable_evolution:
            self.ontogenetic = OntogeneticEngine(
                base_order=self.config.base_order,
                seed=self.seed,
            )
            self.ontogenetic.initialize()
        
        self.bseries = BSeriesKernel(max_order=self.config.bseries_order)
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize state
        self.reservoir_state = np.zeros(self.params.reservoir_size)
        
        self.is_initialized = True
    
    def _initialize_weights(self) -> None:
        """Initialize reservoir weights."""
        units = self.params.reservoir_size
        
        # Input weights
        self.W_in = self.rng.standard_normal((units, self.input_dim))
        self.W_in *= self.params.input_scaling
        
        # Sparsify input weights
        mask = self.rng.random((units, self.input_dim)) < 0.1
        self.W_in *= mask
        
        # Recurrent weights
        self.W = self.rng.standard_normal((units, units))
        
        # Sparsify
        mask = self.rng.random((units, units)) < 0.1
        self.W *= mask
        
        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        current_sr = np.max(np.abs(eigenvalues))
        if current_sr > 0:
            self.W *= self.params.spectral_radius / current_sr
        
        # Feedback weights (if enabled)
        if self.config.feedback_strength > 0:
            self.W_fb = self.rng.standard_normal((units, self.output_dim))
            self.W_fb *= self.config.feedback_strength
        
        # Output weights (initialized during training)
        self.W_out = np.zeros((self.output_dim, units))
    
    def _reservoir_dynamics(self, state: np.ndarray, input_vec: np.ndarray) -> np.ndarray:
        """Compute reservoir dynamics for B-series integration."""
        pre_activation = self.W_in @ input_vec + self.W @ state
        return np.tanh(pre_activation) - state
    
    def step(
        self,
        x: np.ndarray,
        y_prev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance reservoir by one step.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector
        y_prev : np.ndarray, optional
            Previous output (for feedback)
            
        Returns
        -------
        np.ndarray
            New reservoir state
        """
        if not self.is_initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        # Add feedback if available
        if y_prev is not None and self.W_fb is not None:
            fb = self.W_fb @ y_prev
        else:
            fb = 0
        
        # Compute pre-activation
        pre_activation = self.W_in @ x + self.W @ self.reservoir_state + fb
        
        # Use B-series integration for update
        if self.bseries is not None:
            # Define dynamics
            def dynamics(s):
                return np.tanh(self.W_in @ x + self.W @ s + fb) - s
            
            new_state = self.bseries.evaluate(
                dynamics,
                self.reservoir_state,
                dt=self.params.leak_rate
            )
        else:
            new_state = np.tanh(pre_activation)
        
        # Leaky integration
        self.reservoir_state = (
            (1 - self.params.leak_rate) * self.reservoir_state +
            self.params.leak_rate * new_state
        )
        
        # Evolution step
        self.step_count += 1
        if (self.config.enable_evolution and 
            self.step_count % self.config.evolution_interval == 0):
            self._evolution_step()
        
        return self.reservoir_state
    
    def _evolution_step(self) -> None:
        """Perform ontogenetic evolution step."""
        if self.ontogenetic is not None:
            fitness = self.ontogenetic.evolve_step()
            
            # Optionally adapt topology
            if self.config.adapt_topology:
                config = self.ontogenetic.get_reservoir_config()
                # Could update weights based on evolution here
    
    def process(
        self,
        X: np.ndarray,
        y_feedback: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process input sequence through reservoir.
        
        Parameters
        ----------
        X : np.ndarray
            Input sequence of shape (timesteps, input_dim)
        y_feedback : np.ndarray, optional
            Feedback sequence for teacher forcing
            
        Returns
        -------
        np.ndarray
            Reservoir states of shape (timesteps, reservoir_size)
        """
        n_steps = X.shape[0]
        states = np.zeros((n_steps, self.params.reservoir_size))
        
        for t in range(n_steps):
            y_prev = y_feedback[t-1] if y_feedback is not None and t > 0 else None
            states[t] = self.step(X[t], y_prev)
        
        return states
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        warmup: int = 0,
        ridge: Optional[float] = None,
    ) -> "AutognosisEngine":
        """Train the readout layer.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (timesteps, input_dim)
        y : np.ndarray
            Target data of shape (timesteps, output_dim)
        warmup : int
            Warmup steps to discard
        ridge : float, optional
            Ridge parameter (defaults to A000081-derived)
            
        Returns
        -------
        AutognosisEngine
            Self (for chaining)
        """
        if not self.is_initialized:
            self.initialize(X.shape[-1], y.shape[-1])
        
        self.output_dim = y.shape[-1]
        
        # Collect states
        states = self.process(X)
        
        # Apply warmup
        states_train = states[warmup:]
        y_train = y[warmup:]
        
        # Ridge regression
        ridge_param = ridge or self.params.ridge_param
        
        StS = states_train.T @ states_train
        StY = states_train.T @ y_train
        
        reg = ridge_param * np.eye(self.params.reservoir_size)
        self.W_out = np.linalg.solve(StS + reg, StY).T
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (timesteps, input_dim)
            
        Returns
        -------
        np.ndarray
            Predictions of shape (timesteps, output_dim)
        """
        states = self.process(X)
        return states @ self.W_out.T
    
    def reset(self) -> None:
        """Reset reservoir state."""
        if self.is_initialized:
            self.reservoir_state = np.zeros(self.params.reservoir_size)
    
    def get_status(self) -> Dict:
        """Get current engine status.
        
        Returns
        -------
        Dict
            Status dictionary
        """
        status = {
            "initialized": self.is_initialized,
            "step_count": self.step_count,
            "parameters": self.params.to_dict(),
            "config": {
                "base_order": self.config.base_order,
                "enable_evolution": self.config.enable_evolution,
                "bseries_order": self.config.bseries_order,
            },
        }
        
        if self.ontogenetic is not None:
            status["evolution"] = self.ontogenetic.state.to_dict()
        
        return status
    
    def to_torch(self) -> "AutognosisTorchEngine":
        """Convert to PyTorch-based engine.
        
        Returns
        -------
        AutognosisTorchEngine
            PyTorch version of the engine
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for to_torch()")
        
        return AutognosisTorchEngine.from_numpy_engine(self)


if TORCH_AVAILABLE:
    _torch_base = nn.Module
else:
    _torch_base = object


class AutognosisTorchEngine(_torch_base):
    """PyTorch-based autognosis engine.
    
    Provides GPU-accelerated autognosis with autograd support.
    """
    
    def __init__(
        self,
        config: Optional[AutognosisConfig] = None,
        base_order: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,  # torch.dtype when available
    ):
        super().__init__()
        
        self.config = config or AutognosisConfig()
        if base_order is not None:
            self.config.base_order = base_order
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or torch.float32
        
        self.params = derive_parameters(self.config.base_order)
        
        # Weights as parameters/buffers
        self.W_in: Optional[Tensor] = None
        self.W: Optional[Tensor] = None
        self.W_out: Optional[nn.Parameter] = None
        
        self.reservoir_state: Optional[Tensor] = None
        self.is_initialized = False
    
    def initialize(self, input_dim: int, output_dim: Optional[int] = None) -> None:
        """Initialize the engine."""
        units = self.params.reservoir_size
        output_dim = output_dim or units
        
        # Input weights (buffer - not trained)
        W_in = torch.randn(units, input_dim, device=self.device, dtype=self.dtype)
        W_in *= self.params.input_scaling
        mask = torch.rand_like(W_in) < 0.1
        W_in *= mask
        self.register_buffer('W_in', W_in)
        
        # Recurrent weights (buffer - not trained)
        W = torch.randn(units, units, device=self.device, dtype=self.dtype)
        mask = torch.rand_like(W) < 0.1
        W *= mask
        
        # Scale to spectral radius
        with torch.no_grad():
            eigenvalues = torch.linalg.eigvals(W)
            current_sr = torch.max(torch.abs(eigenvalues)).real
            if current_sr > 0:
                W *= self.params.spectral_radius / current_sr
        self.register_buffer('W', W)
        
        # Output weights (parameter - trained)
        self.W_out = nn.Parameter(
            torch.zeros(output_dim, units, device=self.device, dtype=self.dtype)
        )
        
        # State
        self.reservoir_state = torch.zeros(units, device=self.device, dtype=self.dtype)
        
        self.is_initialized = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through reservoir.
        
        Parameters
        ----------
        x : Tensor
            Input of shape (batch, timesteps, input_dim) or (timesteps, input_dim)
            
        Returns
        -------
        Tensor
            Output of shape (batch, timesteps, output_dim) or (timesteps, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, n_steps, _ = x.shape
        units = self.params.reservoir_size
        
        states = torch.zeros(batch_size, n_steps, units, device=self.device, dtype=self.dtype)
        state = torch.zeros(batch_size, units, device=self.device, dtype=self.dtype)
        
        lr = self.params.leak_rate
        
        for t in range(n_steps):
            pre_act = x[:, t] @ self.W_in.T + state @ self.W.T
            new_state = torch.tanh(pre_act)
            state = (1 - lr) * state + lr * new_state
            states[:, t] = state
        
        # Output projection
        output = states @ self.W_out.T
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    @classmethod
    def from_numpy_engine(cls, engine: AutognosisEngine) -> "AutognosisTorchEngine":
        """Create from numpy-based engine."""
        torch_engine = cls(
            config=engine.config,
            device=engine.device,
        )
        
        if engine.is_initialized:
            torch_engine.initialize(engine.input_dim, engine.output_dim)
            
            # Copy weights
            torch_engine.W_in.copy_(torch.from_numpy(engine.W_in))
            torch_engine.W.copy_(torch.from_numpy(engine.W))
            if engine.W_out is not None:
                torch_engine.W_out.data.copy_(torch.from_numpy(engine.W_out))
        
        return torch_engine
