"""
====================================
B-Series Kernel (:mod:`reservoirpy.pytorch.autognosis.bseries`)
====================================

B-Series computational ridges for numerical integration.

B-series are formal power series used to represent numerical methods
for ordinary differential equations. They are indexed by rooted trees
and provide a unified framework for analyzing integration methods.

Key Concepts
------------
- **Rooted Trees**: Fundamental structures for B-series
- **Elementary Differentials**: Derivatives indexed by trees
- **B-Series Coefficients**: Weights for each tree
- **Order Conditions**: Constraints for method accuracy

The B-series kernel provides:
- Tree generation and manipulation
- Elementary differential computation
- B-series evaluation and integration
- Order condition verification
"""

# License: MIT License
# Copyright: nanechopy contributors (adapted from echo-jnn)

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any

try:
    from .a000081 import get_a000081_value, A000081_SEQUENCE
except ImportError:
    from a000081 import get_a000081_value, A000081_SEQUENCE


@dataclass
class RootedTree:
    """Representation of a rooted tree.
    
    A rooted tree is represented as a list of children trees.
    The empty list [] represents a single node (the root).
    
    Attributes
    ----------
    children : List[RootedTree]
        List of child subtrees
    order : int
        Number of nodes in the tree
    symmetry : int
        Symmetry factor (automorphism count)
    density : float
        Density coefficient for B-series
        
    Examples
    --------
    >>> # Single node (order 1)
    >>> t1 = RootedTree([])
    >>> 
    >>> # Two nodes (order 2)
    >>> t2 = RootedTree([RootedTree([])])
    >>> 
    >>> # Three nodes, linear (order 3)
    >>> t3_linear = RootedTree([RootedTree([RootedTree([])])])
    >>> 
    >>> # Three nodes, branching (order 3)
    >>> t3_branch = RootedTree([RootedTree([]), RootedTree([])])
    """
    children: List["RootedTree"] = field(default_factory=list)
    _order: Optional[int] = None
    _symmetry: Optional[int] = None
    _density: Optional[float] = None
    
    @property
    def order(self) -> int:
        """Number of nodes in the tree."""
        if self._order is None:
            self._order = 1 + sum(child.order for child in self.children)
        return self._order
    
    @property
    def symmetry(self) -> int:
        """Symmetry factor (number of automorphisms)."""
        if self._symmetry is None:
            if not self.children:
                self._symmetry = 1
            else:
                # Count multiplicities of identical children
                child_counts: Dict[str, int] = {}
                for child in self.children:
                    key = child.canonical_form()
                    child_counts[key] = child_counts.get(key, 0) + 1
                
                # Symmetry = product of factorials * product of child symmetries
                from math import factorial
                self._symmetry = 1
                for count in child_counts.values():
                    self._symmetry *= factorial(count)
                for child in self.children:
                    self._symmetry *= child.symmetry
        
        return self._symmetry
    
    @property
    def density(self) -> float:
        """Density coefficient gamma(t) for B-series."""
        if self._density is None:
            if not self.children:
                self._density = 1.0
            else:
                self._density = self.order * np.prod([child.density for child in self.children])
        return self._density
    
    def canonical_form(self) -> str:
        """Get canonical string representation."""
        if not self.children:
            return "[]"
        child_forms = sorted(child.canonical_form() for child in self.children)
        return "[" + ",".join(child_forms) + "]"
    
    def __str__(self) -> str:
        return self.canonical_form()
    
    def __repr__(self) -> str:
        return f"RootedTree({self.canonical_form()}, order={self.order})"
    
    def __eq__(self, other: "RootedTree") -> bool:
        return self.canonical_form() == other.canonical_form()
    
    def __hash__(self) -> int:
        return hash(self.canonical_form())


def generate_trees(order: int) -> List[RootedTree]:
    """Generate all rooted trees of given order.
    
    Parameters
    ----------
    order : int
        Number of nodes in trees
        
    Returns
    -------
    List[RootedTree]
        All distinct rooted trees with `order` nodes
        
    Examples
    --------
    >>> trees = generate_trees(4)
    >>> len(trees)
    4  # A000081[4] = 4
    """
    if order < 1:
        return []
    if order == 1:
        return [RootedTree([])]
    
    # Generate trees recursively using partitions
    trees = []
    _generate_trees_recursive(order - 1, order - 1, [], trees)
    return trees


def _generate_trees_recursive(
    remaining: int,
    max_child_order: int,
    current_children: List[RootedTree],
    result: List[RootedTree],
) -> None:
    """Recursively generate trees by partitioning remaining nodes."""
    if remaining == 0:
        result.append(RootedTree(list(current_children)))
        return
    
    # Try each possible child order
    for child_order in range(min(remaining, max_child_order), 0, -1):
        child_trees = generate_trees(child_order)
        for child_tree in child_trees:
            current_children.append(child_tree)
            _generate_trees_recursive(
                remaining - child_order,
                child_order,
                current_children,
                result
            )
            current_children.pop()


@dataclass
class BSeriesKernel:
    """B-Series computational kernel.
    
    Provides B-series evaluation and integration for reservoir computing.
    
    Parameters
    ----------
    max_order : int
        Maximum tree order to consider
    coefficients : Dict[str, float], optional
        B-series coefficients indexed by tree canonical form
    dtype : type, optional
        Data type for computations
        
    Attributes
    ----------
    trees : Dict[int, List[RootedTree]]
        Trees organized by order
    coefficients : Dict[str, float]
        B-series coefficients
        
    Examples
    --------
    >>> kernel = BSeriesKernel(max_order=5)
    >>> 
    >>> # Evaluate B-series at a point
    >>> f = lambda y: -y  # dy/dt = -y
    >>> y0 = np.array([1.0])
    >>> y1 = kernel.evaluate(f, y0, dt=0.1)
    """
    max_order: int
    coefficients: Dict[str, float] = field(default_factory=dict)
    dtype: type = float
    
    def __post_init__(self):
        """Initialize trees and default coefficients."""
        self.trees: Dict[int, List[RootedTree]] = {}
        for order in range(1, self.max_order + 1):
            self.trees[order] = generate_trees(order)
        
        # Initialize with Euler method coefficients if not provided
        if not self.coefficients:
            self._init_euler_coefficients()
    
    def _init_euler_coefficients(self) -> None:
        """Initialize coefficients for Euler method."""
        for order, trees in self.trees.items():
            for tree in trees:
                # Euler method: a(t) = 1/gamma(t) for all trees
                self.coefficients[tree.canonical_form()] = 1.0 / tree.density
    
    def set_rk4_coefficients(self) -> None:
        """Set coefficients for classical RK4 method."""
        # RK4 is order 4 accurate
        for order, trees in self.trees.items():
            for tree in trees:
                if order <= 4:
                    # Exact coefficients for RK4
                    self.coefficients[tree.canonical_form()] = 1.0 / tree.density
                else:
                    # Higher order terms have errors
                    self.coefficients[tree.canonical_form()] = 0.0
    
    def elementary_differential(
        self,
        tree: RootedTree,
        f: Callable,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute elementary differential F(t)(y).
        
        Parameters
        ----------
        tree : RootedTree
            Tree indexing the differential
        f : Callable
            Vector field f(y)
        y : np.ndarray
            Point at which to evaluate
            
        Returns
        -------
        np.ndarray
            Elementary differential value
        """
        if not tree.children:
            # F(τ)(y) = f(y) for single node
            return np.asarray(f(y))
        
        # Recursive: F(t)(y) = f^(k)(y)(F(t1)(y), ..., F(tk)(y))
        # For simplicity, use finite differences for higher derivatives
        child_diffs = [self.elementary_differential(child, f, y) for child in tree.children]
        
        # Approximate higher derivative using finite differences
        eps = 1e-6
        result = np.zeros_like(y)
        
        if len(tree.children) == 1:
            # First derivative: f'(y) * F(t1)(y)
            f_y = np.asarray(f(y))
            for i in range(len(y)):
                y_plus = y.copy()
                y_plus[i] += eps
                f_plus = np.asarray(f(y_plus))
                result += ((f_plus - f_y) / eps) * child_diffs[0][i]
        else:
            # Higher derivatives - simplified approximation
            result = np.asarray(f(y)) * np.prod([np.linalg.norm(d) for d in child_diffs])
        
        return result
    
    def evaluate(
        self,
        f: Callable,
        y: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Evaluate B-series at a point.
        
        Parameters
        ----------
        f : Callable
            Vector field f(y)
        y : np.ndarray
            Current state
        dt : float
            Time step
            
        Returns
        -------
        np.ndarray
            B-series value (next state approximation)
        """
        result = y.copy()
        
        for order in range(1, self.max_order + 1):
            for tree in self.trees[order]:
                coef = self.coefficients.get(tree.canonical_form(), 0.0)
                if abs(coef) > 1e-12:
                    elem_diff = self.elementary_differential(tree, f, y)
                    result = result + coef * (dt ** order) * elem_diff / tree.symmetry
        
        return result
    
    def integrate(
        self,
        f: Callable,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate ODE using B-series method.
        
        Parameters
        ----------
        f : Callable
            Vector field dy/dt = f(y)
        y0 : np.ndarray
            Initial condition
        t_span : Tuple[float, float]
            (t_start, t_end)
        n_steps : int
            Number of integration steps
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (times, states) arrays
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        
        times = np.linspace(t_start, t_end, n_steps + 1)
        states = np.zeros((n_steps + 1, len(y0)))
        states[0] = y0
        
        y = y0.copy()
        for i in range(n_steps):
            y = self.evaluate(f, y, dt)
            states[i + 1] = y
        
        return times, states
    
    def order_conditions(self, target_order: int) -> List[Tuple[RootedTree, float, float]]:
        """Check order conditions up to target order.
        
        Parameters
        ----------
        target_order : int
            Order to check
            
        Returns
        -------
        List[Tuple[RootedTree, float, float]]
            List of (tree, expected, actual) for each condition
        """
        conditions = []
        
        for order in range(1, min(target_order + 1, self.max_order + 1)):
            for tree in self.trees[order]:
                expected = 1.0 / tree.density
                actual = self.coefficients.get(tree.canonical_form(), 0.0)
                conditions.append((tree, expected, actual))
        
        return conditions


class BSeriesReservoir:
    """Reservoir computing layer using B-series dynamics.
    
    Combines Echo State Network dynamics with B-series integration
    for improved numerical stability and accuracy.
    
    Parameters
    ----------
    units : int
        Number of reservoir units
    max_order : int
        Maximum B-series order
    spectral_radius : float
        Spectral radius of reservoir
    leak_rate : float
        Leaking rate
    """
    
    def __init__(
        self,
        units: int,
        max_order: int = 3,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.units = units
        self.max_order = max_order
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        
        self.kernel = BSeriesKernel(max_order=max_order)
        
        # Initialize weights
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((units, units))
        
        # Scale to spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        current_sr = np.max(np.abs(eigenvalues))
        if current_sr > 0:
            self.W = self.W * (spectral_radius / current_sr)
        
        self.state = np.zeros(units)
    
    def _reservoir_dynamics(self, x: np.ndarray) -> np.ndarray:
        """Reservoir dynamics as vector field."""
        return np.tanh(self.W @ x) - x
    
    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance reservoir state by one step.
        
        Parameters
        ----------
        u : np.ndarray
            Input vector
            
        Returns
        -------
        np.ndarray
            New reservoir state
        """
        # Use B-series integration for reservoir update
        augmented_state = self.state + u[:self.units] if len(u) >= self.units else self.state
        
        new_state = self.kernel.evaluate(
            self._reservoir_dynamics,
            augmented_state,
            dt=self.leak_rate
        )
        
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state
    
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run reservoir on input sequence.
        
        Parameters
        ----------
        inputs : np.ndarray
            Input sequence of shape (timesteps, input_dim)
            
        Returns
        -------
        np.ndarray
            Reservoir states of shape (timesteps, units)
        """
        n_steps = inputs.shape[0]
        states = np.zeros((n_steps, self.units))
        
        for t in range(n_steps):
            states[t] = self.step(inputs[t])
        
        return states
    
    def reset(self) -> None:
        """Reset reservoir state to zeros."""
        self.state = np.zeros(self.units)
