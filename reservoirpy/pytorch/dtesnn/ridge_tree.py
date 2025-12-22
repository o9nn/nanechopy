"""
====================================
B-Series Ridge Tree (:mod:`reservoirpy.pytorch.dtesnn.ridge_tree`)
====================================

B-Series Rooted Trees as Ridge/Readout nodes.

In the DTESNN architecture, the readout layer is structured as a forest
of rooted trees, where each tree corresponds to an A000081 enumerated
structure. The ridge regression weights are derived from tree properties
(density, symmetry, order).

Key Concepts
------------
- **Ridge Tree**: A rooted tree where each node performs ridge regression
- **Tree Planting**: Trees are "planted" in membrane compartments
- **Synchronized Weights**: Weights derived from A000081 isomorphism

The B-series framework provides:
1. Tree-indexed coefficients for numerical integration
2. Elementary differentials for gradient computation
3. Order conditions for accuracy guarantees
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
    torch = None
    nn = None

try:
    from ..autognosis.bseries import RootedTree, generate_trees, BSeriesKernel
    from ..autognosis.a000081 import get_a000081_value, cumulative_a000081
    from ..node import TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
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
    class TorchTrainableNode:
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
    TorchState = dict
    TorchTimestep = np.ndarray
    TorchTimeseries = np.ndarray
    def to_tensor(x): return x


@dataclass
class RidgeTreeNode:
    """A node in the ridge tree structure.
    
    Each node performs a weighted linear transformation with ridge
    regularization, where weights are derived from tree properties.
    
    Attributes
    ----------
    tree : RootedTree
        The rooted tree this node represents
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    weight : np.ndarray
        Ridge regression weight matrix
    bias : np.ndarray
        Bias vector
    ridge_param : float
        Ridge regularization parameter
    children : List[RidgeTreeNode]
        Child nodes in the tree
    parent : Optional[RidgeTreeNode]
        Parent node (None for root)
    membrane_id : str
        ID of membrane this tree is planted in
    """
    tree: RootedTree
    input_dim: int
    output_dim: int
    weight: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    ridge_param: float = 1e-6
    children: List["RidgeTreeNode"] = field(default_factory=list)
    parent: Optional["RidgeTreeNode"] = None
    membrane_id: str = ""
    
    def __post_init__(self):
        """Initialize weights from tree properties."""
        if self.weight is None:
            # Weight scale from tree density
            scale = 1.0 / np.sqrt(self.tree.density * self.input_dim)
            self.weight = np.random.randn(self.output_dim, self.input_dim) * scale
        
        if self.bias is None:
            self.bias = np.zeros(self.output_dim)
    
    @property
    def order(self) -> int:
        return self.tree.order
    
    @property
    def symmetry(self) -> int:
        return self.tree.symmetry
    
    @property
    def density(self) -> float:
        return self.tree.density
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through this node."""
        return x @ self.weight.T + self.bias
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ridge: Optional[float] = None,
    ) -> None:
        """Fit ridge regression weights.
        
        Uses tree-derived regularization:
        ridge_effective = ridge_param * density / symmetry
        """
        ridge_param = ridge or self.ridge_param
        
        # Tree-derived regularization
        ridge_effective = ridge_param * self.density / self.symmetry
        
        # Ridge regression: W = (X^T X + λI)^{-1} X^T y
        XtX = X.T @ X
        Xty = X.T @ y
        
        reg = ridge_effective * np.eye(self.input_dim)
        self.weight = np.linalg.solve(XtX + reg, Xty).T
        
        # Fit bias
        self.bias = np.mean(y - X @ self.weight.T, axis=0)


class BSeriesRidgeTree(TorchTrainableNode):
    """B-Series Ridge Tree readout layer.
    
    A forest of rooted trees performing hierarchical ridge regression.
    Each tree is "planted" in a membrane compartment, creating a
    synchronized connection between readout and reservoir.
    
    Parameters
    ----------
    base_order : int
        Maximum tree order (determines forest size)
    input_dim : int, optional
        Input dimension (inferred if not provided)
    output_dim : int, optional
        Output dimension (inferred from training)
    ridge : float, default 1e-6
        Base ridge regularization parameter
    aggregation : str, default 'weighted_sum'
        How to aggregate tree outputs ('weighted_sum', 'concat', 'attention')
    use_bseries : bool, default True
        Whether to use B-series coefficients for weighting
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
    forest : List[RidgeTreeNode]
        All ridge tree nodes
    tree_weights : np.ndarray
        B-series coefficients for each tree
    planted_trees : Dict[str, List[RidgeTreeNode]]
        Trees organized by membrane ID
        
    Examples
    --------
    >>> ridge_tree = BSeriesRidgeTree(base_order=5)
    >>> ridge_tree.fit(reservoir_states, targets)
    >>> predictions = ridge_tree.run(new_states)
    """
    
    def __init__(
        self,
        base_order: int = 5,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        ridge: float = 1e-6,
        aggregation: str = 'weighted_sum',
        use_bseries: bool = True,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.base_order = base_order
        self.input_dim = input_dim
        self._output_dim = output_dim
        self.ridge = ridge
        self.aggregation = aggregation
        self.use_bseries = use_bseries
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Generate tree structure
        self.trees: Dict[int, List[RootedTree]] = {}
        self._all_trees: List[RootedTree] = []
        for order in range(1, base_order + 1):
            self.trees[order] = generate_trees(order)
            self._all_trees.extend(self.trees[order])
        
        # B-series kernel for coefficients
        if use_bseries:
            self.bseries = BSeriesKernel(max_order=base_order)
        else:
            self.bseries = None
        
        # Forest of ridge tree nodes (created on initialize)
        self.forest: List[RidgeTreeNode] = []
        self.tree_weights: Optional[np.ndarray] = None
        self.planted_trees: Dict[str, List[RidgeTreeNode]] = {}
        
        # Aggregation weights
        self.W_agg: Optional[np.ndarray] = None
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the ridge tree forest."""
        self._set_input_dim(x)
        if y is not None:
            self._set_output_dim(y)
        
        # Create ridge tree nodes
        self.forest = []
        for tree in self._all_trees:
            node = RidgeTreeNode(
                tree=tree,
                input_dim=self.input_dim,
                output_dim=self._output_dim or self.input_dim,
                ridge_param=self.ridge,
            )
            self.forest.append(node)
        
        # Compute B-series weights
        self._compute_tree_weights()
        
        # Set output dimension
        if self.aggregation == 'concat':
            self.output_dim = len(self.forest) * (self._output_dim or self.input_dim)
        else:
            self.output_dim = self._output_dim or self.input_dim
        
        # Initialize aggregation weights
        if self.aggregation == 'attention':
            self.W_agg = self.rng.standard_normal((len(self.forest),)) * 0.1
        
        # Initialize state
        if TORCH_AVAILABLE:
            self.state = {
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)
            }
        else:
            self.state = {"out": np.zeros(self.output_dim)}
        
        self.initialized = True
    
    def _compute_tree_weights(self) -> None:
        """Compute B-series weights for each tree."""
        self.tree_weights = np.zeros(len(self._all_trees))
        
        for i, tree in enumerate(self._all_trees):
            if self.use_bseries and self.bseries is not None:
                # B-series coefficient: 1 / (gamma(t) * sigma(t))
                coef = self.bseries.coefficients.get(
                    tree.canonical_form(),
                    1.0 / tree.density
                )
                self.tree_weights[i] = coef / tree.symmetry
            else:
                # Simple weight based on tree properties
                self.tree_weights[i] = 1.0 / (tree.density * tree.symmetry)
        
        # Normalize
        self.tree_weights /= np.sum(self.tree_weights)
    
    def plant_in_membrane(
        self,
        tree_index: int,
        membrane_id: str,
    ) -> None:
        """Plant a tree's root in a membrane compartment.
        
        Parameters
        ----------
        tree_index : int
            Index of tree in forest
        membrane_id : str
            ID of target membrane
        """
        if tree_index >= len(self.forest):
            return
        
        node = self.forest[tree_index]
        node.membrane_id = membrane_id
        
        if membrane_id not in self.planted_trees:
            self.planted_trees[membrane_id] = []
        self.planted_trees[membrane_id].append(node)
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep through ridge tree forest."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Forward through each tree
        outputs = []
        for i, node in enumerate(self.forest):
            out = node.forward(x_np)
            outputs.append(out * self.tree_weights[i])
        
        # Aggregate
        if self.aggregation == 'weighted_sum':
            result = np.sum(outputs, axis=0)
        elif self.aggregation == 'concat':
            result = np.concatenate(outputs)
        elif self.aggregation == 'attention':
            # Attention-weighted sum
            attn = np.softmax(self.W_agg)
            result = np.sum([o * a for o, a in zip(outputs, attn)], axis=0)
        else:
            result = np.mean(outputs, axis=0)
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            result = torch.from_numpy(result).to(device=self.device, dtype=self.dtype)
        
        return {"out": result}
    
    def _run(
        self,
        state: TorchState,
        x: TorchTimeseries,
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence through ridge tree forest."""
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
        
        return {"out": final_out}, outputs
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        ridge: Optional[float] = None,
        **kwargs,
    ) -> "BSeriesRidgeTree":
        """Train all ridge tree nodes.
        
        Parameters
        ----------
        x : array-like
            Input data (reservoir states)
        y : array-like
            Target data
        warmup : int
            Warmup steps to discard
        ridge : float, optional
            Ridge parameter (overrides default)
            
        Returns
        -------
        BSeriesRidgeTree
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
            return self
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Apply warmup
        x_train = x_np[warmup:]
        y_train = y_np[warmup:]
        
        # Update output dimension
        self._output_dim = y_train.shape[-1]
        
        # Train each tree node
        ridge_param = ridge or self.ridge
        for node in self.forest:
            # Tree-specific ridge parameter
            tree_ridge = ridge_param * node.density / node.symmetry
            node.output_dim = self._output_dim
            node.fit(x_train, y_train, ridge=tree_ridge)
        
        # Update output dimension
        if self.aggregation == 'concat':
            self.output_dim = len(self.forest) * self._output_dim
        else:
            self.output_dim = self._output_dim
        
        return self
    
    def get_tree_structure(self) -> str:
        """Get string representation of tree structure."""
        lines = [
            f"B-Series Ridge Tree Forest (base_order={self.base_order})",
            "=" * 50,
            f"Total trees: {len(self.forest)}",
            "",
        ]
        
        for order in range(1, min(self.base_order + 1, 5)):
            trees = self.trees[order]
            lines.append(f"Order {order}: {len(trees)} trees")
            for tree in trees[:3]:
                idx = self._all_trees.index(tree)
                node = self.forest[idx]
                lines.append(
                    f"  {tree} (w={self.tree_weights[idx]:.4f}, "
                    f"membrane={node.membrane_id or 'unplanted'})"
                )
            if len(trees) > 3:
                lines.append(f"  ... and {len(trees) - 3} more")
        
        return "\n".join(lines)
    
    def reset(self) -> TorchState:
        """Reset node state."""
        previous_state = self.state
        if TORCH_AVAILABLE:
            self.state = {
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)
            }
        else:
            self.state = {"out": np.zeros(self.output_dim)}
        return previous_state


class HierarchicalRidgeTree(BSeriesRidgeTree):
    """Hierarchical ridge tree with parent-child relationships.
    
    Extends BSeriesRidgeTree to maintain explicit tree hierarchy,
    where child nodes receive input from parent nodes.
    
    Parameters
    ----------
    base_order : int
        Maximum tree order
    hierarchy_type : str, default 'depth_first'
        How to organize hierarchy ('depth_first', 'breadth_first', 'order_based')
    propagate_residuals : bool, default True
        Whether to propagate residuals down the tree
    """
    
    def __init__(
        self,
        base_order: int = 5,
        hierarchy_type: str = 'depth_first',
        propagate_residuals: bool = True,
        **kwargs,
    ):
        super().__init__(base_order=base_order, **kwargs)
        
        self.hierarchy_type = hierarchy_type
        self.propagate_residuals = propagate_residuals
        
        # Parent-child relationships (built on initialize)
        self.parent_map: Dict[int, int] = {}
        self.children_map: Dict[int, List[int]] = {}
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize with hierarchy."""
        super().initialize(x, y)
        self._build_hierarchy()
    
    def _build_hierarchy(self) -> None:
        """Build parent-child relationships based on tree structure."""
        self.parent_map = {}
        self.children_map = {i: [] for i in range(len(self.forest))}
        
        if self.hierarchy_type == 'order_based':
            # Trees of order n are children of trees of order n-1
            for i, node in enumerate(self.forest):
                if node.order > 1:
                    # Find potential parents (order - 1)
                    parents = [
                        j for j, p in enumerate(self.forest)
                        if p.order == node.order - 1
                    ]
                    if parents:
                        parent_idx = parents[0]  # Simplified: use first
                        self.parent_map[i] = parent_idx
                        self.children_map[parent_idx].append(i)
                        node.parent = self.forest[parent_idx]
                        self.forest[parent_idx].children.append(node)
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process with hierarchical propagation."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Process in hierarchical order (roots first)
        outputs = [None] * len(self.forest)
        residuals = [None] * len(self.forest)
        
        # Find roots (no parent)
        roots = [i for i in range(len(self.forest)) if i not in self.parent_map]
        
        # BFS through hierarchy
        queue = list(roots)
        while queue:
            idx = queue.pop(0)
            node = self.forest[idx]
            
            # Input is original x plus parent residual
            node_input = x_np.copy()
            if self.propagate_residuals and idx in self.parent_map:
                parent_idx = self.parent_map[idx]
                if residuals[parent_idx] is not None:
                    node_input = node_input + residuals[parent_idx]
            
            # Forward
            out = node.forward(node_input)
            outputs[idx] = out * self.tree_weights[idx]
            
            # Compute residual for children
            if self.propagate_residuals:
                residuals[idx] = out - np.mean(out)
            
            # Add children to queue
            queue.extend(self.children_map[idx])
        
        # Aggregate
        valid_outputs = [o for o in outputs if o is not None]
        if self.aggregation == 'weighted_sum':
            result = np.sum(valid_outputs, axis=0)
        elif self.aggregation == 'concat':
            result = np.concatenate(valid_outputs)
        else:
            result = np.mean(valid_outputs, axis=0)
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            result = torch.from_numpy(result).to(device=self.device, dtype=self.dtype)
        
        return {"out": result}
