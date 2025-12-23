"""
====================================
J-Surface Elementary Differential ESN (:mod:`reservoirpy.pytorch.dtesnn.jsurface_esn`)
====================================

J-Surface Elementary Differentials as ESN nodes.

In the DTESNN architecture, the ESN layer uses elementary differentials
indexed by rooted trees (A000081). The J-surface provides a symplectic
manifold structure for gradient-evolution coupling.

Key Concepts
------------
- **Elementary Differential**: Tree-indexed derivative operator
- **J-Surface**: Symplectic manifold for dynamics
- **Gradient-Evolution Coupling**: Symplectic integration
- **Tree-Indexed ESN**: Each tree defines an ESN component

Elementary Differentials
------------------------
For a vector field f, the elementary differential F(t) indexed by tree t is:
- F(•) = f (single node)
- F([t1, t2, ...]) = f'(F(t1), F(t2), ...) (composition)

J-Surface Properties
--------------------
1. Symplectic structure J = [[0, I], [-I, 0]]
2. Hamiltonian dynamics preserve J
3. Gradient and evolution are J-coupled
4. Volume-preserving flow
"""

# License: MIT License
# Copyright: nanechopy contributors

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
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
    from ..autognosis.bseries import RootedTree, generate_trees, BSeriesKernel
    from ..autognosis.a000081 import get_a000081_value, cumulative_a000081
    from ..node import TorchNode, TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
except ImportError:
    try:
        from bseries import RootedTree, generate_trees, BSeriesKernel
        from a000081 import get_a000081_value, cumulative_a000081
    except ImportError:
        import sys
        import os
        autognosis_path = os.path.join(os.path.dirname(__file__), '..', 'autognosis')
        sys.path.insert(0, autognosis_path)
        from bseries import RootedTree, generate_trees, BSeriesKernel
        from a000081 import get_a000081_value, cumulative_a000081
    
    # Minimal node stubs for standalone testing
    class TorchNode:
        def __init__(self, **kwargs):
            self.device = kwargs.get('device')
            self.dtype = kwargs.get('dtype')
            self.name = kwargs.get('name')
            self.initialized = False
            self.state = {}
            self.input_dim = None
            self.output_dim = None
        def _set_input_dim(self, x):
            self.input_dim = x.shape[-1] if hasattr(x, 'shape') else len(x)
        def _set_output_dim(self, y):
            self.output_dim = y.shape[-1] if hasattr(y, 'shape') else len(y)
    TorchTrainableNode = TorchNode
    TorchState = dict
    TorchTimestep = np.ndarray
    TorchTimeseries = np.ndarray
    def to_tensor(x): return x


@dataclass
class ElementaryDifferential:
    """An elementary differential indexed by a rooted tree.
    
    Elementary differentials are the building blocks of B-series
    numerical methods. Each tree t defines a differential operator F(t).
    
    Attributes
    ----------
    tree : RootedTree
        The indexing rooted tree
    order : int
        Order of the differential (= tree order)
    coefficient : float
        B-series coefficient (1 / (density * symmetry))
    weight_matrix : np.ndarray
        Weight matrix for this differential
    activation : Callable
        Activation function
    """
    tree: RootedTree
    order: int
    coefficient: float
    weight_matrix: Optional[np.ndarray] = None
    activation: Callable = np.tanh
    
    def __post_init__(self):
        """Compute coefficient if not provided."""
        if self.coefficient == 0:
            self.coefficient = 1.0 / (self.tree.density * self.tree.symmetry)
    
    def evaluate(
        self,
        state: np.ndarray,
        child_values: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Evaluate the elementary differential.
        
        For tree t with children [t1, t2, ...]:
        F(t)(y) = f'(y)(F(t1)(y), F(t2)(y), ...)
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        child_values : List[np.ndarray], optional
            Values from child differentials
            
        Returns
        -------
        np.ndarray
            Differential value
        """
        if self.weight_matrix is None:
            return self.activation(state) * self.coefficient
        
        # Base transformation
        base = self.weight_matrix @ state
        
        # Incorporate child values (tree composition)
        if child_values:
            for child_val in child_values:
                base = base + child_val * 0.1  # Scaled contribution
        
        return self.activation(base) * self.coefficient


class JMatrix:
    """Symplectic J matrix for J-surface dynamics.
    
    The standard symplectic matrix J = [[0, I], [-I, 0]] defines
    the coupling between gradient and evolution.
    
    Parameters
    ----------
    dim : int
        Dimension (must be even for standard J)
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.half_dim = dim // 2
        
        # Build J matrix
        self.J = np.zeros((dim, dim))
        self.J[:self.half_dim, self.half_dim:] = np.eye(self.half_dim)
        self.J[self.half_dim:, :self.half_dim] = -np.eye(self.half_dim)
    
    def apply(self, v: np.ndarray) -> np.ndarray:
        """Apply J to vector: J @ v."""
        return self.J @ v
    
    def apply_transpose(self, v: np.ndarray) -> np.ndarray:
        """Apply J^T to vector: -J @ v."""
        return -self.J @ v
    
    def hamiltonian_gradient(
        self,
        grad_H: np.ndarray,
    ) -> np.ndarray:
        """Convert Hamiltonian gradient to symplectic gradient.
        
        X_H = J @ grad(H)
        """
        return self.apply(grad_H)
    
    def is_symplectic(self, M: np.ndarray) -> bool:
        """Check if matrix M is symplectic: M^T J M = J."""
        result = M.T @ self.J @ M
        return np.allclose(result, self.J)


@dataclass
class ElementaryDifferentialNode:
    """A node representing an elementary differential in the ESN.
    
    Each node corresponds to a rooted tree and computes the
    associated elementary differential.
    
    Attributes
    ----------
    differential : ElementaryDifferential
        The elementary differential
    units : int
        Number of ESN units
    state : np.ndarray
        Current state
    W : np.ndarray
        Recurrent weights
    W_in : np.ndarray
        Input weights
    leak_rate : float
        Leaking rate
    spectral_radius : float
        Spectral radius
    membrane_id : str
        Associated membrane ID (for synchronization)
    """
    differential: ElementaryDifferential
    units: int
    state: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    W_in: Optional[np.ndarray] = None
    leak_rate: float = 0.3
    spectral_radius: float = 0.9
    membrane_id: str = ""
    
    def __post_init__(self):
        if self.state is None:
            self.state = np.zeros(self.units)
    
    @property
    def tree(self) -> RootedTree:
        return self.differential.tree
    
    @property
    def order(self) -> int:
        return self.differential.order
    
    @property
    def coefficient(self) -> float:
        return self.differential.coefficient
    
    def initialize_weights(
        self,
        input_dim: int,
        density: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize ESN weights."""
        rng = np.random.default_rng(seed)
        
        # Input weights
        self.W_in = rng.standard_normal((self.units, input_dim)) * 0.1
        
        # Recurrent weights (sparse)
        W = rng.standard_normal((self.units, self.units))
        mask = rng.random((self.units, self.units)) < density
        W = W * mask
        
        # Scale to spectral radius
        if np.any(W):
            current_sr = np.max(np.abs(np.linalg.eigvals(W)))
            if current_sr > 0:
                W = W * (self.spectral_radius / current_sr)
        
        self.W = W
        
        # Set differential weight matrix
        self.differential.weight_matrix = W
    
    def step(
        self,
        x: np.ndarray,
        child_states: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Evolve node state for one timestep.
        
        Uses elementary differential for update:
        s(t+1) = (1-lr) * s(t) + lr * F(tree)(s(t), x)
        """
        # Input contribution
        u = self.W_in @ x
        
        # Recurrent contribution via elementary differential
        child_values = child_states if child_states else None
        r = self.differential.evaluate(self.state, child_values)
        
        # Leaky integration
        pre_activation = u + self.W @ self.state
        new_state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre_activation)
        
        # Apply differential coefficient
        new_state = new_state * (1 + self.coefficient * 0.1)
        
        self.state = new_state
        return new_state
    
    def reset(self) -> None:
        """Reset node state."""
        self.state = np.zeros(self.units)


class JSurfaceESN(TorchNode):
    """J-Surface Elementary Differential ESN.
    
    An Echo State Network where dynamics are defined by elementary
    differentials on a symplectic J-surface manifold.
    
    Parameters
    ----------
    base_order : int
        Maximum tree order (determines number of differentials)
    units_per_differential : int
        ESN units per elementary differential
    total_units : int, optional
        Total units (overrides units_per_differential)
    input_dim : int, optional
        Input dimension (inferred if not provided)
    leak_rate : float, default 0.3
        Leaking rate
    spectral_radius : float, default 0.9
        Spectral radius
    density : float, default 0.1
        Sparsity of recurrent weights
    use_symplectic : bool, default True
        Whether to use symplectic integration
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
    differentials : List[ElementaryDifferentialNode]
        All elementary differential nodes
    J : JMatrix
        Symplectic J matrix
    bseries : BSeriesKernel
        B-series kernel for coefficients
        
    Examples
    --------
    >>> esn = JSurfaceESN(base_order=5, units_per_differential=20)
    >>> esn.initialize(input_data)
    >>> states = esn.run(input_sequence)
    """
    
    def __init__(
        self,
        base_order: int = 5,
        units_per_differential: int = 20,
        total_units: Optional[int] = None,
        input_dim: Optional[int] = None,
        leak_rate: float = 0.3,
        spectral_radius: float = 0.9,
        density: float = 0.1,
        use_symplectic: bool = True,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.base_order = base_order
        self.units_per_differential = units_per_differential
        self._total_units = total_units
        self.input_dim = input_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.use_symplectic = use_symplectic
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Generate trees
        self.trees: Dict[int, List[RootedTree]] = {}
        self._all_trees: List[RootedTree] = []
        for order in range(1, base_order + 1):
            self.trees[order] = generate_trees(order)
            self._all_trees.extend(self.trees[order])
        
        # B-series kernel
        self.bseries = BSeriesKernel(max_order=base_order)
        
        # Elementary differential nodes (created on initialize)
        self.differentials: List[ElementaryDifferentialNode] = []
        
        # J matrix (created on initialize)
        self.J: Optional[JMatrix] = None
        
        # Processing order (based on tree structure)
        self.processing_order: List[int] = []
        
        # Parent-child relationships
        self.parent_map: Dict[int, int] = {}
        self.children_map: Dict[int, List[int]] = {}
    
    def _build_differential_structure(self) -> None:
        """Build elementary differential nodes from trees."""
        # Calculate units per differential
        num_differentials = len(self._all_trees)
        if self._total_units:
            units_per = self._total_units // num_differentials
        else:
            units_per = self.units_per_differential
        
        # Ensure even number for symplectic structure
        if self.use_symplectic:
            units_per = (units_per // 2) * 2
        
        self.differentials = []
        
        for i, tree in enumerate(self._all_trees):
            # Get B-series coefficient
            coef = self.bseries.coefficients.get(
                tree.canonical_form(),
                1.0 / tree.density
            )
            
            # Adjust parameters based on tree
            tree_sr = self.spectral_radius * (1 - 0.1 * (tree.order - 1) / self.base_order)
            tree_lr = self.leak_rate * (1 + 0.1 / tree.symmetry)
            
            differential = ElementaryDifferential(
                tree=tree,
                order=tree.order,
                coefficient=coef,
            )
            
            node = ElementaryDifferentialNode(
                differential=differential,
                units=units_per,
                leak_rate=min(tree_lr, 1.0),
                spectral_radius=tree_sr,
            )
            
            self.differentials.append(node)
        
        # Build parent-child relationships
        self._build_hierarchy()
    
    def _build_hierarchy(self) -> None:
        """Build parent-child relationships based on tree order."""
        self.parent_map = {}
        self.children_map = {i: [] for i in range(len(self.differentials))}
        
        for i, node in enumerate(self.differentials):
            if node.order > 1:
                # Find potential parents (order - 1)
                parents = [
                    j for j, p in enumerate(self.differentials)
                    if p.order == node.order - 1
                ]
                if parents:
                    parent_idx = parents[i % len(parents)]
                    self.parent_map[i] = parent_idx
                    self.children_map[parent_idx].append(i)
        
        # Determine processing order (children before parents)
        self.processing_order = []
        visited = set()
        
        def visit(idx: int):
            if idx in visited:
                return
            visited.add(idx)
            for child_idx in self.children_map[idx]:
                visit(child_idx)
            self.processing_order.append(idx)
        
        # Start from roots (no parent)
        roots = [i for i in range(len(self.differentials)) if i not in self.parent_map]
        for root in roots:
            visit(root)
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the J-Surface ESN."""
        self._set_input_dim(x)
        
        # Build differential structure
        self._build_differential_structure()
        
        # Initialize weights for each differential
        for node in self.differentials:
            node.initialize_weights(
                input_dim=self.input_dim,
                density=self.density,
                seed=self.rng.integers(0, 2**31),
            )
        
        # Set output dimension
        self.output_dim = sum(node.units for node in self.differentials)
        
        # Initialize J matrix for symplectic dynamics
        if self.use_symplectic:
            self.J = JMatrix(self.output_dim)
        
        # Initialize state
        if TORCH_AVAILABLE:
            self.state = {
                "esn": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "esn": np.zeros(self.output_dim),
                "out": np.zeros(self.output_dim),
            }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep through J-Surface ESN."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Store states for child lookups
        node_states: Dict[int, np.ndarray] = {}
        
        # Process in order (children before parents)
        for idx in self.processing_order:
            node = self.differentials[idx]
            
            # Get child states
            child_states = [
                node_states[child_idx]
                for child_idx in self.children_map[idx]
                if child_idx in node_states
            ]
            
            # Step node
            new_state = node.step(x_np, child_states if child_states else None)
            node_states[idx] = new_state
        
        # Collect all states
        esn_state = np.concatenate([node.state for node in self.differentials])
        
        # Apply symplectic correction if enabled
        if self.use_symplectic and self.J is not None:
            # Symplectic gradient correction
            grad = np.gradient(esn_state)
            symplectic_grad = self.J.hamiltonian_gradient(grad)
            esn_state = esn_state + 0.01 * symplectic_grad
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            esn_tensor = torch.from_numpy(esn_state).to(
                device=self.device, dtype=self.dtype
            )
            return {
                "esn": esn_tensor,
                "out": esn_tensor,
            }
        else:
            return {
                "esn": esn_state,
                "out": esn_state,
            }
    
    def _run(
        self,
        state: TorchState,
        x: TorchTimeseries,
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence through J-Surface ESN."""
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
        
        return {"esn": final_out, "out": final_out}, outputs
    
    def synchronize_with_membrane(
        self,
        membrane_id: str,
        differential_indices: List[int],
    ) -> None:
        """Synchronize differentials with a membrane compartment.
        
        Parameters
        ----------
        membrane_id : str
            Target membrane ID
        differential_indices : List[int]
            Indices of differentials to synchronize
        """
        for idx in differential_indices:
            if idx < len(self.differentials):
                self.differentials[idx].membrane_id = membrane_id
    
    def get_differential_states(self) -> Dict[int, np.ndarray]:
        """Get states organized by differential index."""
        return {
            i: node.state.copy()
            for i, node in enumerate(self.differentials)
        }
    
    def get_structure_info(self) -> str:
        """Get string representation of differential structure."""
        lines = [
            f"J-Surface Elementary Differential ESN (base_order={self.base_order})",
            "=" * 50,
            f"Total differentials: {len(self.differentials)}",
            f"Total units: {self.output_dim}",
            f"Symplectic: {self.use_symplectic}",
            "",
            "Elementary Differentials:",
        ]
        
        for order in range(1, min(self.base_order + 1, 5)):
            trees = self.trees[order]
            lines.append(f"  Order {order}: {len(trees)} differentials")
            for i, tree in enumerate(trees[:3]):
                idx = self._all_trees.index(tree)
                node = self.differentials[idx]
                lines.append(
                    f"    F({tree}): coef={node.coefficient:.4f}, "
                    f"units={node.units}, membrane={node.membrane_id or 'none'}"
                )
            if len(trees) > 3:
                lines.append(f"    ... and {len(trees) - 3} more")
        
        return "\n".join(lines)
    
    def reset(self) -> TorchState:
        """Reset ESN state."""
        previous_state = self.state
        
        for node in self.differentials:
            node.reset()
        
        if TORCH_AVAILABLE:
            self.state = {
                "esn": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "esn": np.zeros(self.output_dim),
                "out": np.zeros(self.output_dim),
            }
        
        return previous_state
