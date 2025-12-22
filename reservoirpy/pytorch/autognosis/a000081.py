"""
====================================
A000081 Parameter Derivation (:mod:`reservoirpy.pytorch.autognosis.a000081`)
====================================

Parameter derivation from the OEIS A000081 sequence.

The A000081 sequence counts the number of rooted trees with n nodes:
A000081: {1, 1, 2, 4, 9, 20, 48, 115, 286, 719, 1842, 4766, 12486, ...}

This sequence provides mathematically grounded parameters for reservoir
computing architectures, ensuring consistency with rooted tree topology.

Key Functions
-------------
- get_a000081_value: Get sequence value at index
- cumulative_a000081: Get cumulative sum up to index
- derive_parameters: Derive all parameters from base order

Parameter Mappings
------------------
- reservoir_size: Cumulative A000081 up to base_order
- max_tree_order: 2 * base_order - 2
- num_membranes: A000081[membrane_order]
- growth_rate: A000081[n+1] / A000081[n]
- mutation_rate: 1 / A000081[base_order]
"""

# License: MIT License
# Copyright: nanechopy contributors (adapted from echo-jnn)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = None


# OEIS A000081: Number of rooted trees with n nodes
# https://oeis.org/A000081
A000081_SEQUENCE = [
    0,      # a(0) = 0 (placeholder for 0-indexing)
    1,      # a(1) = 1
    1,      # a(2) = 1
    2,      # a(3) = 2
    4,      # a(4) = 4
    9,      # a(5) = 9
    20,     # a(6) = 20
    48,     # a(7) = 48
    115,    # a(8) = 115
    286,    # a(9) = 286
    719,    # a(10) = 719
    1842,   # a(11) = 1842
    4766,   # a(12) = 4766
    12486,  # a(13) = 12486
    32973,  # a(14) = 32973
    87811,  # a(15) = 87811
    235381, # a(16) = 235381
    634847, # a(17) = 634847
    1721159,# a(18) = 1721159
    4688676,# a(19) = 4688676
    12826228,# a(20) = 12826228
]


def get_a000081_value(n: int) -> int:
    """Get the n-th value of the A000081 sequence.
    
    Parameters
    ----------
    n : int
        Index in the sequence (1-indexed, n >= 1)
        
    Returns
    -------
    int
        Number of rooted trees with n nodes
        
    Examples
    --------
    >>> get_a000081_value(5)
    9
    >>> get_a000081_value(10)
    719
    """
    if n < 0:
        raise ValueError(f"Index must be non-negative, got {n}")
    if n == 0:
        return 0
    if n < len(A000081_SEQUENCE):
        return A000081_SEQUENCE[n]
    
    # Compute using recurrence relation for larger n
    # This is computationally expensive for large n
    return _compute_a000081(n)


def _compute_a000081(n: int) -> int:
    """Compute A000081(n) using the recurrence relation.
    
    Uses the Euler transform recurrence:
    a(n) = (1/n) * sum_{k=1}^{n-1} (sum_{d|k} d * a(d)) * a(n-k)
    """
    if n < len(A000081_SEQUENCE):
        return A000081_SEQUENCE[n]
    
    # Extend sequence
    a = list(A000081_SEQUENCE)
    while len(a) <= n:
        m = len(a)
        # Compute sum over divisors
        total = 0
        for k in range(1, m):
            divisor_sum = sum(d * a[d] for d in range(1, k + 1) if k % d == 0)
            total += divisor_sum * a[m - k]
        a.append(total // m)
    
    return a[n]


def cumulative_a000081(n: int) -> int:
    """Get cumulative sum of A000081 up to index n.
    
    Parameters
    ----------
    n : int
        Upper index (inclusive)
        
    Returns
    -------
    int
        Sum of A000081[1] + A000081[2] + ... + A000081[n]
        
    Examples
    --------
    >>> cumulative_a000081(5)
    17  # 1 + 1 + 2 + 4 + 9
    """
    return sum(get_a000081_value(i) for i in range(1, n + 1))


@dataclass
class A000081Parameters:
    """Parameters derived from A000081 sequence.
    
    Attributes
    ----------
    base_order : int
        Base order used for derivation
    reservoir_size : int
        Recommended reservoir size (cumulative trees)
    max_tree_order : int
        Maximum tree order for B-series
    num_membranes : int
        Number of membrane compartments
    growth_rate : float
        Growth rate for ontogenetic evolution
    mutation_rate : float
        Mutation rate for evolution
    spectral_radius : float
        Recommended spectral radius
    leak_rate : float
        Recommended leak rate
    input_scaling : float
        Recommended input scaling
    ridge_param : float
        Recommended ridge parameter
    """
    base_order: int
    reservoir_size: int
    max_tree_order: int
    num_membranes: int
    growth_rate: float
    mutation_rate: float
    spectral_radius: float
    leak_rate: float
    input_scaling: float
    ridge_param: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "base_order": self.base_order,
            "reservoir_size": self.reservoir_size,
            "max_tree_order": self.max_tree_order,
            "num_membranes": self.num_membranes,
            "growth_rate": self.growth_rate,
            "mutation_rate": self.mutation_rate,
            "spectral_radius": self.spectral_radius,
            "leak_rate": self.leak_rate,
            "input_scaling": self.input_scaling,
            "ridge_param": self.ridge_param,
        }
    
    def explain(self) -> str:
        """Generate explanation of parameter derivation."""
        lines = [
            f"A000081 Parameter Derivation (base_order={self.base_order})",
            "=" * 60,
            "",
            f"reservoir_size = {self.reservoir_size}",
            f"  → Cumulative A000081 up to order {self.base_order}",
            f"  → Sum of rooted trees: {' + '.join(str(get_a000081_value(i)) for i in range(1, self.base_order + 1))}",
            "",
            f"max_tree_order = {self.max_tree_order}",
            f"  → 2 * base_order - 2 = 2 * {self.base_order} - 2",
            "",
            f"num_membranes = {self.num_membranes}",
            f"  → A000081[{min(self.base_order - 1, 4)}] = {self.num_membranes}",
            "",
            f"growth_rate = {self.growth_rate:.4f}",
            f"  → A000081[{self.base_order + 1}] / A000081[{self.base_order}]",
            f"  → {get_a000081_value(self.base_order + 1)} / {get_a000081_value(self.base_order)}",
            "",
            f"mutation_rate = {self.mutation_rate:.6f}",
            f"  → 1 / A000081[{self.base_order}] = 1 / {get_a000081_value(self.base_order)}",
            "",
            f"spectral_radius = {self.spectral_radius:.4f}",
            f"  → 1 - mutation_rate",
            "",
            f"leak_rate = {self.leak_rate:.4f}",
            f"  → 1 / base_order = 1 / {self.base_order}",
            "",
            f"input_scaling = {self.input_scaling:.4f}",
            f"  → sqrt(1 / reservoir_size)",
            "",
            f"ridge_param = {self.ridge_param:.2e}",
            f"  → 1 / (reservoir_size * A000081[base_order])",
        ]
        return "\n".join(lines)


def derive_parameters(
    base_order: int = 5,
    membrane_order: Optional[int] = None,
) -> A000081Parameters:
    """Derive all parameters from A000081 sequence.
    
    Parameters
    ----------
    base_order : int, default 5
        Base order for parameter derivation (recommended: 5-8)
    membrane_order : int, optional
        Order for membrane count (defaults to base_order - 1)
        
    Returns
    -------
    A000081Parameters
        Complete parameter set derived from A000081
        
    Examples
    --------
    >>> params = derive_parameters(base_order=5)
    >>> print(params.reservoir_size)
    17
    >>> print(params.explain())
    """
    if base_order < 2:
        raise ValueError(f"base_order must be >= 2, got {base_order}")
    
    membrane_order = membrane_order or min(base_order - 1, 4)
    
    # Core parameters
    reservoir_size = cumulative_a000081(base_order)
    max_tree_order = 2 * base_order - 2
    num_membranes = get_a000081_value(membrane_order)
    
    # Rates
    a_n = get_a000081_value(base_order)
    a_n_plus_1 = get_a000081_value(base_order + 1)
    
    growth_rate = a_n_plus_1 / a_n
    mutation_rate = 1.0 / a_n
    
    # Derived reservoir parameters
    spectral_radius = 1.0 - mutation_rate
    leak_rate = 1.0 / base_order
    input_scaling = np.sqrt(1.0 / reservoir_size)
    ridge_param = 1.0 / (reservoir_size * a_n)
    
    return A000081Parameters(
        base_order=base_order,
        reservoir_size=reservoir_size,
        max_tree_order=max_tree_order,
        num_membranes=num_membranes,
        growth_rate=growth_rate,
        mutation_rate=mutation_rate,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        input_scaling=input_scaling,
        ridge_param=ridge_param,
    )


def validate_parameters(
    reservoir_size: int,
    max_tree_order: int,
    num_membranes: int,
    growth_rate: float,
    mutation_rate: float,
) -> Tuple[bool, str]:
    """Validate if parameters align with A000081 sequence.
    
    Parameters
    ----------
    reservoir_size : int
        Reservoir size to validate
    max_tree_order : int
        Maximum tree order
    num_membranes : int
        Number of membranes
    growth_rate : float
        Growth rate
    mutation_rate : float
        Mutation rate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)
    """
    messages = []
    is_valid = True
    
    # Check if reservoir_size is a cumulative A000081
    found_cumulative = False
    for n in range(1, 15):
        if cumulative_a000081(n) == reservoir_size:
            found_cumulative = True
            break
    
    if not found_cumulative:
        is_valid = False
        messages.append(f"reservoir_size={reservoir_size} is not a cumulative A000081 value")
    
    # Check if num_membranes is in A000081
    found_membrane = False
    for n in range(1, 15):
        if get_a000081_value(n) == num_membranes:
            found_membrane = True
            break
    
    if not found_membrane:
        is_valid = False
        messages.append(f"num_membranes={num_membranes} is not in A000081 sequence")
    
    # Check growth rate
    for n in range(1, 14):
        expected_rate = get_a000081_value(n + 1) / get_a000081_value(n)
        if abs(growth_rate - expected_rate) < 0.01:
            break
    else:
        messages.append(f"growth_rate={growth_rate:.4f} does not match A000081 ratios")
    
    # Check mutation rate
    for n in range(1, 15):
        expected_rate = 1.0 / get_a000081_value(n)
        if abs(mutation_rate - expected_rate) < 0.001:
            break
    else:
        messages.append(f"mutation_rate={mutation_rate:.6f} does not match 1/A000081[n]")
    
    message = "\n".join(messages) if messages else "All parameters align with A000081"
    return is_valid, message


def get_recommended_orders() -> List[Tuple[int, int, str]]:
    """Get recommended base orders with descriptions.
    
    Returns
    -------
    List[Tuple[int, int, str]]
        List of (base_order, reservoir_size, description)
    """
    recommendations = [
        (3, cumulative_a000081(3), "Minimal: Good for testing"),
        (4, cumulative_a000081(4), "Small: Simple tasks"),
        (5, cumulative_a000081(5), "Standard: Balanced performance"),
        (6, cumulative_a000081(6), "Medium: Complex patterns"),
        (7, cumulative_a000081(7), "Large: High capacity"),
        (8, cumulative_a000081(8), "Very Large: Maximum expressiveness"),
    ]
    return recommendations
