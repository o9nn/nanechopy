"""
====================================
DTESNN Model (:mod:`reservoirpy.pytorch.dtesnn.model`)
====================================

Deep Tree Echo State Neural Network - Main Model.

The DTESNN integrates three A000081-isomorphic components:
1. J-Surface ESN (Elementary Differentials) - Input processing
2. Membrane Reservoir (P-System Nests) - State dynamics
3. Ridge Tree (B-Series) - Output readout

All three systems share synchronized parameters through the A000081
sequence (rooted tree enumeration), creating a mathematically coherent
architecture where "Ridge Trees have their Roots Planted in Reservoir
Membranes."

Architecture
------------
```
Input → [J-Surface ESN] → [Membrane Reservoir] → [Ridge Tree Readout] → Output
              ↑                    ↑                      ↑
              └────────────────────┴──────────────────────┘
                        A000081 Synchronization
                        
Tree i ←→ Membrane i ←→ Differential i (isomorphic)
```

Key Features
------------
- Unified A000081 parameter derivation
- Tree-membrane-differential synchronization
- Symplectic dynamics on J-surface
- Hierarchical P-system communication
- B-series numerical integration
"""

# License: MIT License
# Copyright: nanechopy contributors

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()
    nn = None

try:
    from .synchronizer import A000081Synchronizer, TreeIsomorphism, SynchronizedParameters
    from .ridge_tree import BSeriesRidgeTree, RidgeTreeNode, HierarchicalRidgeTree
    from .membrane_reservoir import MembraneReservoir, MembraneNest, MembraneCompartment
    from .jsurface_esn import JSurfaceESN, ElementaryDifferentialNode, JMatrix
    from ..autognosis.bseries import RootedTree, generate_trees
    from ..autognosis.a000081 import get_a000081_value, cumulative_a000081, derive_parameters
    from ..node import TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
except ImportError:
    # For standalone testing
    import sys
    import os
    dtesnn_path = os.path.dirname(__file__)
    autognosis_path = os.path.join(dtesnn_path, '..', 'autognosis')
    sys.path.insert(0, dtesnn_path)
    sys.path.insert(0, autognosis_path)
    
    from synchronizer import A000081Synchronizer, TreeIsomorphism, SynchronizedParameters
    from ridge_tree import BSeriesRidgeTree, RidgeTreeNode, HierarchicalRidgeTree
    from membrane_reservoir import MembraneReservoir, MembraneNest, MembraneCompartment
    from jsurface_esn import JSurfaceESN, ElementaryDifferentialNode, JMatrix
    from bseries import RootedTree, generate_trees
    from a000081 import get_a000081_value, cumulative_a000081, derive_parameters
    
    # Minimal node stubs for standalone testing
    class TorchTrainableNode:
        def __init__(self, **kwargs):
            self.device = kwargs.get('device')
            self.dtype = kwargs.get('dtype')
            self.name = kwargs.get('name')
            self.seed = kwargs.get('seed')
            self.initialized = False
            self.state = {}
            self.input_dim = None
            self.output_dim = None
        def _set_input_dim(self, x):
            self.input_dim = x.shape[-1] if hasattr(x, 'shape') else len(x)
        def _set_output_dim(self, y):
            self.output_dim = y.shape[-1] if hasattr(y, 'shape') else len(y)
    TorchState = dict
    TorchTimestep = np.ndarray
    TorchTimeseries = np.ndarray
    def to_tensor(x): return x


@dataclass
class DTESNNConfig:
    """Configuration for DTESNN model.
    
    Attributes
    ----------
    base_order : int
        Base order for A000081 derivation (determines structure size)
    units_per_component : int
        Units per tree/membrane/differential
    total_units : int, optional
        Total units (overrides units_per_component)
    leak_rate : float
        Leaking rate for ESN dynamics
    spectral_radius : float
        Spectral radius of recurrent weights
    density : float
        Sparsity of recurrent weights
    ridge : float
        Ridge regularization parameter
    communication_strength : float
        Strength of inter-membrane communication
    use_symplectic : bool
        Whether to use symplectic integration in J-Surface
    use_hierarchical_ridge : bool
        Whether to use hierarchical ridge tree
    aggregation : str
        Ridge tree aggregation method
    synchronize_weights : bool
        Whether to synchronize weights across components
    """
    base_order: int = 6
    units_per_component: int = 30
    total_units: Optional[int] = None
    leak_rate: float = 0.3
    spectral_radius: float = 0.9
    density: float = 0.1
    ridge: float = 1e-6
    communication_strength: float = 0.1
    use_symplectic: bool = True
    use_hierarchical_ridge: bool = True
    aggregation: str = 'weighted_sum'
    synchronize_weights: bool = True


class DTESNN(TorchTrainableNode):
    """Deep Tree Echo State Neural Network.
    
    A unified architecture integrating three A000081-isomorphic systems:
    - J-Surface ESN: Elementary differentials for input processing
    - Membrane Reservoir: P-system nests for state dynamics
    - Ridge Tree: B-series trees for output readout
    
    The key insight is that rooted trees (A000081), membrane hierarchies,
    and elementary differentials are all enumerated by the same sequence,
    allowing synchronized parameter sharing.
    
    Parameters
    ----------
    config : DTESNNConfig, optional
        Configuration object
    base_order : int, optional
        Base order (overrides config)
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
        
    Attributes
    ----------
    synchronizer : A000081Synchronizer
        Synchronization mechanism
    jsurface : JSurfaceESN
        J-Surface ESN layer
    reservoir : MembraneReservoir
        Membrane reservoir layer
    ridge_tree : BSeriesRidgeTree
        Ridge tree readout layer
    plantings : Dict[int, str]
        Tree index -> membrane ID plantings
        
    Examples
    --------
    >>> # Create DTESNN with base order 6
    >>> model = DTESNN(base_order=6, seed=42)
    >>> 
    >>> # Train
    >>> model.fit(X_train, y_train, warmup=100)
    >>> 
    >>> # Predict
    >>> predictions = model.run(X_test)
    >>> 
    >>> # Inspect structure
    >>> print(model.get_structure_info())
    >>> 
    >>> # Explain synchronization
    >>> print(model.explain_synchronization())
    """
    
    def __init__(
        self,
        config: Optional[DTESNNConfig] = None,
        base_order: Optional[int] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.config = config or DTESNNConfig()
        if base_order is not None:
            self.config.base_order = base_order
        
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Create synchronizer
        self.synchronizer = A000081Synchronizer(
            base_order=self.config.base_order,
            seed=seed,
        )
        
        # Components (created on initialize)
        self.jsurface: Optional[JSurfaceESN] = None
        self.reservoir: Optional[MembraneReservoir] = None
        self.ridge_tree: Optional[BSeriesRidgeTree] = None
        
        # Tree plantings (tree_index -> membrane_id)
        self.plantings: Dict[int, str] = {}
        
        # Combined state dimensions
        self._jsurface_dim: int = 0
        self._reservoir_dim: int = 0
        self._combined_dim: int = 0
    
    def _create_components(self) -> None:
        """Create the three DTESNN components."""
        # Calculate units per component
        num_trees = len(self.synchronizer._all_trees)
        if self.config.total_units:
            units_per = self.config.total_units // (3 * num_trees)
        else:
            units_per = self.config.units_per_component
        
        # Create J-Surface ESN
        self.jsurface = JSurfaceESN(
            base_order=self.config.base_order,
            units_per_differential=units_per,
            input_dim=self.input_dim,
            leak_rate=self.config.leak_rate,
            spectral_radius=self.config.spectral_radius,
            density=self.config.density,
            use_symplectic=self.config.use_symplectic,
            device=self.device,
            dtype=self.dtype,
            seed=self.rng.integers(0, 2**31),
        )
        
        # Create Membrane Reservoir
        self.reservoir = MembraneReservoir(
            base_order=self.config.base_order,
            units_per_membrane=units_per,
            input_dim=self.input_dim,
            leak_rate=self.config.leak_rate,
            spectral_radius=self.config.spectral_radius,
            density=self.config.density,
            communication_strength=self.config.communication_strength,
            device=self.device,
            dtype=self.dtype,
            seed=self.rng.integers(0, 2**31),
        )
        
        # Create Ridge Tree (hierarchical or standard)
        if self.config.use_hierarchical_ridge:
            self.ridge_tree = HierarchicalRidgeTree(
                base_order=self.config.base_order,
                ridge=self.config.ridge,
                aggregation=self.config.aggregation,
                device=self.device,
                dtype=self.dtype,
                seed=self.rng.integers(0, 2**31),
            )
        else:
            self.ridge_tree = BSeriesRidgeTree(
                base_order=self.config.base_order,
                ridge=self.config.ridge,
                aggregation=self.config.aggregation,
                device=self.device,
                dtype=self.dtype,
                seed=self.rng.integers(0, 2**31),
            )
    
    def _plant_trees_in_membranes(self) -> None:
        """Plant ridge trees in membrane compartments.
        
        This creates the key connection between readout (ridge trees)
        and reservoir (membranes) based on A000081 isomorphism.
        """
        self.plantings = {}
        
        for iso in self.synchronizer.isomorphisms:
            tree_idx = iso.differential_index
            membrane_id = iso.membrane_id
            
            # Plant tree in membrane
            membrane_id_actual, connection_info = self.synchronizer.plant_tree_in_membrane(tree_idx)
            
            # Update reservoir
            if self.reservoir and self.reservoir.nest:
                # Find matching membrane
                for m_id in self.reservoir.nest.membranes:
                    if m_id.endswith(f"_{tree_idx}") or iso.order == self.reservoir.nest.membranes[m_id].order:
                        self.reservoir.plant_tree(tree_idx, m_id)
                        self.plantings[tree_idx] = m_id
                        break
            
            # Update ridge tree
            if self.ridge_tree and tree_idx < len(self.ridge_tree.forest):
                self.ridge_tree.plant_in_membrane(tree_idx, membrane_id_actual)
            
            # Update J-Surface ESN
            if self.jsurface and tree_idx < len(self.jsurface.differentials):
                self.jsurface.synchronize_with_membrane(membrane_id_actual, [tree_idx])
    
    def _synchronize_weights(self) -> None:
        """Synchronize weights across all three components."""
        if not self.config.synchronize_weights:
            return
        
        # Get weight matrices
        weight_tensor, adj_tensor = self.synchronizer.to_tensor_weights()
        if TORCH_AVAILABLE and hasattr(weight_tensor, 'numpy'):
            weight_np = weight_tensor.numpy()
            adj_np = adj_tensor.numpy()
        else:
            weight_np = np.asarray(weight_tensor)
            adj_np = np.asarray(adj_tensor)
        
        # Apply synchronized weights to ridge tree
        if self.ridge_tree:
            for i, node in enumerate(self.ridge_tree.forest):
                if i < len(weight_np):
                    # Scale node weights by synchronized weight
                    sync_weight = weight_np[i, i]
                    if node.weight is not None:
                        node.weight = node.weight * sync_weight
        
        # Apply to membrane reservoir
        if self.reservoir and self.reservoir.nest:
            for i, (m_id, membrane) in enumerate(self.reservoir.nest.membranes.items()):
                if i < len(weight_np):
                    # Scale recurrent weights
                    sync_weight = weight_np[i, i] if i < len(weight_np) else 1.0
                    if membrane.W is not None:
                        membrane.W = membrane.W * sync_weight
        
        # Apply to J-Surface ESN
        if self.jsurface:
            for i, node in enumerate(self.jsurface.differentials):
                if i < len(weight_np):
                    sync_weight = weight_np[i, i]
                    if node.W is not None:
                        node.W = node.W * sync_weight
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the DTESNN model."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Create components
        self._create_components()
        
        # Initialize J-Surface ESN
        self.jsurface.initialize(x)
        self._jsurface_dim = self.jsurface.output_dim
        
        # Initialize Membrane Reservoir (receives J-Surface output)
        # Create dummy input with J-Surface dimension
        if TORCH_AVAILABLE:
            jsurface_dummy = torch.zeros(self._jsurface_dim)
        else:
            jsurface_dummy = np.zeros(self._jsurface_dim)
        self.reservoir.initialize(jsurface_dummy)
        self._reservoir_dim = self.reservoir.output_dim
        
        # Combined dimension for ridge tree input
        self._combined_dim = self._jsurface_dim + self._reservoir_dim
        
        # Initialize Ridge Tree
        if TORCH_AVAILABLE:
            combined_dummy = torch.zeros(self._combined_dim)
        else:
            combined_dummy = np.zeros(self._combined_dim)
        self.ridge_tree.initialize(combined_dummy, y)
        
        # Plant trees in membranes
        self._plant_trees_in_membranes()
        
        # Synchronize weights
        self._synchronize_weights()
        
        # Set output dimension
        self.output_dim = self.ridge_tree.output_dim
        
        # Initialize state
        if TORCH_AVAILABLE:
            self.state = {
                "jsurface": torch.zeros(self._jsurface_dim, device=self.device, dtype=self.dtype),
                "reservoir": torch.zeros(self._reservoir_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "jsurface": np.zeros(self._jsurface_dim),
                "reservoir": np.zeros(self._reservoir_dim),
                "out": np.zeros(self.output_dim),
            }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep through DTESNN."""
        # 1. Process through J-Surface ESN
        jsurface_state = self.jsurface._step(self.jsurface.state, x)
        jsurface_out = jsurface_state["out"]
        self.jsurface.state = jsurface_state
        
        # 2. Process through Membrane Reservoir
        reservoir_state = self.reservoir._step(self.reservoir.state, jsurface_out)
        reservoir_out = reservoir_state["out"]
        self.reservoir.state = reservoir_state
        
        # 3. Combine J-Surface and Reservoir outputs
        if TORCH_AVAILABLE:
            if isinstance(jsurface_out, Tensor) and isinstance(reservoir_out, Tensor):
                combined = torch.cat([jsurface_out, reservoir_out])
            else:
                combined = np.concatenate([
                    jsurface_out.detach().cpu().numpy() if isinstance(jsurface_out, Tensor) else jsurface_out,
                    reservoir_out.detach().cpu().numpy() if isinstance(reservoir_out, Tensor) else reservoir_out,
                ])
        else:
            combined = np.concatenate([jsurface_out, reservoir_out])
        
        # 4. Process through Ridge Tree
        ridge_state = self.ridge_tree._step(self.ridge_tree.state, combined)
        ridge_out = ridge_state["out"]
        self.ridge_tree.state = ridge_state
        
        return {
            "jsurface": jsurface_out,
            "reservoir": reservoir_out,
            "out": ridge_out,
        }
    
    def _run(
        self,
        state: TorchState,
        x: TorchTimeseries,
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence through DTESNN."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        n_steps = x_np.shape[0]
        outputs = np.zeros((n_steps, self.output_dim))
        
        for t in range(n_steps):
            state = self._step(state, x_np[t])
            if TORCH_AVAILABLE and isinstance(state["out"], Tensor):
                outputs[t] = state["out"].detach().cpu().numpy()
            else:
                outputs[t] = state["out"]
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            outputs = torch.from_numpy(outputs).to(device=self.device, dtype=self.dtype)
            final_out = outputs[-1]
        else:
            final_out = outputs[-1]
        
        return {
            "jsurface": state["jsurface"],
            "reservoir": state["reservoir"],
            "out": final_out,
        }, outputs
    
    def _collect_states(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        warmup: int = 0,
    ) -> np.ndarray:
        """Collect combined states for training."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        n_steps = x_np.shape[0]
        combined_states = np.zeros((n_steps, self._combined_dim))
        
        for t in range(n_steps):
            # Process through J-Surface
            jsurface_state = self.jsurface._step(self.jsurface.state, x_np[t])
            jsurface_out = jsurface_state["out"]
            self.jsurface.state = jsurface_state
            
            # Process through Reservoir
            reservoir_state = self.reservoir._step(self.reservoir.state, jsurface_out)
            reservoir_out = reservoir_state["out"]
            self.reservoir.state = reservoir_state
            
            # Combine
            if TORCH_AVAILABLE:
                js_np = jsurface_out.detach().cpu().numpy() if isinstance(jsurface_out, Tensor) else jsurface_out
                res_np = reservoir_out.detach().cpu().numpy() if isinstance(reservoir_out, Tensor) else reservoir_out
            else:
                js_np = jsurface_out
                res_np = reservoir_out
            
            combined_states[t] = np.concatenate([js_np, res_np])
        
        return combined_states[warmup:]
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "DTESNN":
        """Train the DTESNN model.
        
        Parameters
        ----------
        x : array-like
            Input data
        y : array-like
            Target data
        warmup : int
            Warmup steps to discard
        ridge : float, optional
            Ridge parameter (overrides config)
            
        Returns
        -------
        DTESNN
            Self (for chaining)
        """
        if not self.initialized:
            self.initialize(x, y)
        
        # Convert targets
        if y is not None:
            if TORCH_AVAILABLE and isinstance(y, Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = np.asarray(y)
        else:
            return self
        
        # Reset states
        self.reset()
        
        # Collect combined states
        combined_states = self._collect_states(x, warmup=0)
        
        # Apply warmup
        states_train = combined_states[warmup:]
        y_train = y_np[warmup:]
        
        # Train ridge tree
        self.ridge_tree.fit(states_train, y_train, warmup=0, ridge=ridge)
        
        # Update output dimension
        self._output_dim = y_train.shape[-1]
        self.output_dim = self.ridge_tree.output_dim
        
        return self
    
    def get_structure_info(self) -> str:
        """Get comprehensive structure information."""
        lines = [
            "=" * 70,
            "DTESNN - Deep Tree Echo State Neural Network",
            "=" * 70,
            "",
            f"Base Order: {self.config.base_order}",
            f"Total Trees: {len(self.synchronizer._all_trees)}",
            f"Tree Counts by Order: {self.synchronizer.params.tree_counts}",
            "",
            "Component Dimensions:",
            f"  J-Surface ESN: {self._jsurface_dim}",
            f"  Membrane Reservoir: {self._reservoir_dim}",
            f"  Combined Input: {self._combined_dim}",
            f"  Output: {self.output_dim}",
            "",
        ]
        
        # J-Surface info
        if self.jsurface:
            lines.append("J-Surface ESN:")
            lines.append(f"  Differentials: {len(self.jsurface.differentials)}")
            lines.append(f"  Symplectic: {self.config.use_symplectic}")
        
        # Reservoir info
        if self.reservoir and self.reservoir.nest:
            lines.append("")
            lines.append("Membrane Reservoir:")
            lines.append(f"  Membranes: {len(self.reservoir.nest.membranes)}")
            lines.append(f"  Communication Strength: {self.config.communication_strength}")
        
        # Ridge Tree info
        if self.ridge_tree:
            lines.append("")
            lines.append("Ridge Tree Readout:")
            lines.append(f"  Trees: {len(self.ridge_tree.forest)}")
            lines.append(f"  Aggregation: {self.config.aggregation}")
            lines.append(f"  Hierarchical: {self.config.use_hierarchical_ridge}")
        
        # Plantings
        lines.append("")
        lines.append(f"Tree Plantings: {len(self.plantings)}")
        for tree_idx, membrane_id in list(self.plantings.items())[:5]:
            iso = self.synchronizer.isomorphisms[tree_idx]
            lines.append(f"  Tree {iso.tree} -> {membrane_id}")
        if len(self.plantings) > 5:
            lines.append(f"  ... and {len(self.plantings) - 5} more")
        
        return "\n".join(lines)
    
    def explain_synchronization(self) -> str:
        """Explain the A000081 synchronization."""
        return self.synchronizer.explain()
    
    def get_component_states(self) -> Dict[str, Any]:
        """Get states from all components."""
        states = {}
        
        if self.jsurface:
            states["jsurface"] = self.jsurface.get_differential_states()
        
        if self.reservoir:
            states["reservoir"] = self.reservoir.get_membrane_states()
        
        if self.ridge_tree:
            states["ridge_tree"] = {
                i: node.weight for i, node in enumerate(self.ridge_tree.forest)
            }
        
        return states
    
    def reset(self) -> TorchState:
        """Reset all component states."""
        previous_state = self.state
        
        if self.jsurface:
            self.jsurface.reset()
        
        if self.reservoir:
            self.reservoir.reset()
        
        if self.ridge_tree:
            self.ridge_tree.reset()
        
        if TORCH_AVAILABLE:
            self.state = {
                "jsurface": torch.zeros(self._jsurface_dim, device=self.device, dtype=self.dtype),
                "reservoir": torch.zeros(self._reservoir_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "jsurface": np.zeros(self._jsurface_dim),
                "reservoir": np.zeros(self._reservoir_dim),
                "out": np.zeros(self.output_dim),
            }
        
        return previous_state


class DTESNNEnsemble(TorchTrainableNode):
    """Ensemble of DTESNN models with different base orders.
    
    Combines multiple DTESNN models with varying A000081 base orders
    for improved performance and robustness.
    
    Parameters
    ----------
    base_orders : List[int]
        List of base orders for ensemble members
    aggregation : str, default 'mean'
        How to aggregate ensemble outputs ('mean', 'weighted', 'voting')
    """
    
    def __init__(
        self,
        base_orders: List[int] = [4, 5, 6],
        aggregation: str = 'mean',
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.base_orders = base_orders
        self.aggregation = aggregation
        
        self.models: List[DTESNN] = []
        self.ensemble_weights: Optional[np.ndarray] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize ensemble members."""
        self._set_input_dim(x)
        
        # Create ensemble members
        self.models = []
        for base_order in self.base_orders:
            model = DTESNN(
                base_order=base_order,
                input_dim=self.input_dim,
                device=self.device,
                dtype=self.dtype,
                seed=self.seed,
            )
            model.initialize(x, y)
            self.models.append(model)
        
        # Set output dimension
        self.output_dim = self.models[0].output_dim
        
        # Initialize ensemble weights
        self.ensemble_weights = np.ones(len(self.models)) / len(self.models)
        
        self.initialized = True
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        **kwargs,
    ) -> "DTESNNEnsemble":
        """Train all ensemble members."""
        if not self.initialized:
            self.initialize(x, y)
        
        for model in self.models:
            model.reset()
            model.fit(x, y, warmup=warmup, **kwargs)
        
        return self
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process through ensemble."""
        outputs = []
        for model in self.models:
            model_state = model._step(model.state, x)
            model.state = model_state
            outputs.append(model_state["out"])
        
        # Aggregate
        if self.aggregation == 'mean':
            if TORCH_AVAILABLE and isinstance(outputs[0], Tensor):
                result = torch.stack(outputs).mean(dim=0)
            else:
                result = np.mean(outputs, axis=0)
        elif self.aggregation == 'weighted':
            if TORCH_AVAILABLE and isinstance(outputs[0], Tensor):
                weights = torch.from_numpy(self.ensemble_weights).to(outputs[0].device)
                result = sum(o * w for o, w in zip(outputs, weights))
            else:
                result = sum(o * w for o, w in zip(outputs, self.ensemble_weights))
        else:
            result = outputs[0]  # Default to first
        
        return {"out": result}
    
    def reset(self) -> TorchState:
        """Reset all ensemble members."""
        for model in self.models:
            model.reset()
        return self.state
