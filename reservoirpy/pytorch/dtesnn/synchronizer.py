"""
====================================
A000081 Synchronizer (:mod:`reservoirpy.pytorch.dtesnn.synchronizer`)
====================================

Synchronization mechanism for A000081-isomorphic components.

The A000081 sequence enumerates rooted trees, and this synchronizer
ensures that all three DTESNN components (Ridge Trees, Membrane Nests,
J-Surface ESNs) share consistent structure through this enumeration.

Key Insight: Rooted trees, membrane hierarchies, and elementary differentials
are all enumerated by A000081, making them isomorphic structures that can
share parameters in a mathematically coherent way.
"""

# License: MIT License
# Copyright: nanechopy contributors

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    torch = None

# Import A000081 utilities
try:
    from ..autognosis.a000081 import (
        get_a000081_value, 
        cumulative_a000081, 
        A000081_SEQUENCE,
        derive_parameters,
        A000081Parameters
    )
    from ..autognosis.bseries import RootedTree, generate_trees
except ImportError:
    try:
        from a000081 import (
            get_a000081_value,
            cumulative_a000081,
            A000081_SEQUENCE,
            derive_parameters,
            A000081Parameters
        )
        from bseries import RootedTree, generate_trees
    except ImportError:
        import sys
        import os
        autognosis_path = os.path.join(os.path.dirname(__file__), '..', 'autognosis')
        sys.path.insert(0, autognosis_path)
        from a000081 import (
            get_a000081_value,
            cumulative_a000081,
            A000081_SEQUENCE,
            derive_parameters,
            A000081Parameters
        )
        from bseries import RootedTree, generate_trees


class IsomorphismType(Enum):
    """Types of A000081 isomorphisms."""
    TREE_TO_MEMBRANE = "tree_to_membrane"
    TREE_TO_DIFFERENTIAL = "tree_to_differential"
    MEMBRANE_TO_DIFFERENTIAL = "membrane_to_differential"
    FULL_TRIPARTITE = "full_tripartite"


@dataclass
class TreeIsomorphism:
    """Isomorphism between a rooted tree and other A000081 structures.
    
    Attributes
    ----------
    tree : RootedTree
        The rooted tree
    membrane_id : str
        Corresponding membrane compartment ID
    differential_index : int
        Index in elementary differential sequence
    weight : float
        Synchronized weight derived from tree properties
    depth : int
        Depth in the hierarchy
    """
    tree: RootedTree
    membrane_id: str
    differential_index: int
    weight: float
    depth: int
    
    @property
    def order(self) -> int:
        return self.tree.order
    
    @property
    def symmetry(self) -> int:
        return self.tree.symmetry
    
    @property
    def density(self) -> float:
        return self.tree.density


@dataclass
class SynchronizedParameters:
    """Parameters synchronized across all three systems.
    
    Attributes
    ----------
    base_order : int
        Base order for A000081 derivation
    total_trees : int
        Total number of trees up to base_order
    tree_counts : List[int]
        A000081[n] for n in 1..base_order
    cumulative_counts : List[int]
        Cumulative sums for indexing
    isomorphisms : List[TreeIsomorphism]
        All tree isomorphisms
    weight_matrix : np.ndarray
        Synchronized weight matrix
    adjacency_matrix : np.ndarray
        Tree adjacency for connectivity
    """
    base_order: int
    total_trees: int
    tree_counts: List[int] = field(default_factory=list)
    cumulative_counts: List[int] = field(default_factory=list)
    isomorphisms: List[TreeIsomorphism] = field(default_factory=list)
    weight_matrix: Optional[np.ndarray] = None
    adjacency_matrix: Optional[np.ndarray] = None


class A000081Synchronizer:
    """Synchronizer for A000081-isomorphic DTESNN components.
    
    This class creates and maintains the isomorphism between:
    1. B-Series Rooted Trees (Ridge/Readout)
    2. P-System Membrane Nests (Reservoir)
    3. J-Surface Elementary Differentials (ESN)
    
    All three are enumerated by A000081, allowing synchronized parameters.
    
    Parameters
    ----------
    base_order : int
        Maximum tree order to consider
    seed : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    params : SynchronizedParameters
        All synchronized parameters
    trees : Dict[int, List[RootedTree]]
        Trees organized by order
    membrane_map : Dict[str, TreeIsomorphism]
        Membrane ID to isomorphism mapping
    differential_map : Dict[int, TreeIsomorphism]
        Differential index to isomorphism mapping
        
    Examples
    --------
    >>> sync = A000081Synchronizer(base_order=6)
    >>> 
    >>> # Get synchronized parameters
    >>> params = sync.get_parameters()
    >>> print(f"Total structures: {params.total_trees}")
    >>> 
    >>> # Get isomorphism for a specific tree
    >>> iso = sync.get_isomorphism(tree_index=5)
    >>> print(f"Tree {iso.tree} -> Membrane {iso.membrane_id}")
    """
    
    def __init__(
        self,
        base_order: int = 6,
        seed: Optional[int] = None,
    ):
        self.base_order = base_order
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Generate all trees
        self.trees: Dict[int, List[RootedTree]] = {}
        self._all_trees: List[RootedTree] = []
        for order in range(1, base_order + 1):
            self.trees[order] = generate_trees(order)
            self._all_trees.extend(self.trees[order])
        
        # Build isomorphisms
        self.isomorphisms: List[TreeIsomorphism] = []
        self.membrane_map: Dict[str, TreeIsomorphism] = {}
        self.differential_map: Dict[int, TreeIsomorphism] = {}
        
        self._build_isomorphisms()
        self._build_weight_matrix()
        self._build_adjacency_matrix()
        
        # Create synchronized parameters
        self.params = SynchronizedParameters(
            base_order=base_order,
            total_trees=len(self._all_trees),
            tree_counts=[get_a000081_value(n) for n in range(1, base_order + 1)],
            cumulative_counts=[cumulative_a000081(n) for n in range(1, base_order + 1)],
            isomorphisms=self.isomorphisms,
            weight_matrix=self._weight_matrix,
            adjacency_matrix=self._adjacency_matrix,
        )
    
    def _build_isomorphisms(self) -> None:
        """Build isomorphisms between trees, membranes, and differentials."""
        tree_index = 0
        
        for order in range(1, self.base_order + 1):
            for local_idx, tree in enumerate(self.trees[order]):
                # Create membrane ID based on tree structure
                membrane_id = self._tree_to_membrane_id(tree, order, local_idx)
                
                # Compute synchronized weight
                weight = self._compute_synchronized_weight(tree)
                
                # Compute depth (tree height)
                depth = self._compute_tree_depth(tree)
                
                iso = TreeIsomorphism(
                    tree=tree,
                    membrane_id=membrane_id,
                    differential_index=tree_index,
                    weight=weight,
                    depth=depth,
                )
                
                self.isomorphisms.append(iso)
                self.membrane_map[membrane_id] = iso
                self.differential_map[tree_index] = iso
                
                tree_index += 1
    
    def _tree_to_membrane_id(self, tree: RootedTree, order: int, local_idx: int) -> str:
        """Convert tree to membrane compartment ID.
        
        The membrane ID encodes the tree structure:
        - Root membrane: "M_0"
        - Child membranes: "M_0.1", "M_0.2", etc.
        - Nested: "M_0.1.1", "M_0.1.2", etc.
        """
        if order == 1:
            return "M_root"
        
        # Build hierarchical ID from tree structure
        def build_id(t: RootedTree, prefix: str, idx: int) -> str:
            current = f"{prefix}.{idx}" if prefix else f"M_{idx}"
            return current
        
        return f"M_{order}_{local_idx}"
    
    def _compute_synchronized_weight(self, tree: RootedTree) -> float:
        """Compute synchronized weight from tree properties.
        
        Weight formula: w = 1 / (density * sqrt(symmetry))
        
        This ensures:
        - Higher order trees have smaller weights (regularization)
        - Symmetric trees have reduced weights (avoid redundancy)
        """
        return 1.0 / (tree.density * np.sqrt(tree.symmetry))
    
    def _compute_tree_depth(self, tree: RootedTree) -> int:
        """Compute depth (height) of tree."""
        if not tree.children:
            return 1
        return 1 + max(self._compute_tree_depth(child) for child in tree.children)
    
    def _build_weight_matrix(self) -> None:
        """Build synchronized weight matrix.
        
        Matrix W[i,j] represents the interaction weight between
        tree i and tree j, based on their structural relationship.
        """
        n = len(self._all_trees)
        self._weight_matrix = np.zeros((n, n))
        
        for i, iso_i in enumerate(self.isomorphisms):
            for j, iso_j in enumerate(self.isomorphisms):
                if i == j:
                    # Self-weight
                    self._weight_matrix[i, j] = iso_i.weight
                else:
                    # Cross-weight based on order difference
                    order_diff = abs(iso_i.order - iso_j.order)
                    if order_diff <= 1:
                        # Adjacent orders can interact
                        self._weight_matrix[i, j] = (
                            iso_i.weight * iso_j.weight * 
                            np.exp(-order_diff)
                        )
    
    def _build_adjacency_matrix(self) -> None:
        """Build adjacency matrix for tree connectivity.
        
        Trees are connected if:
        1. One is a subtree of another
        2. They share the same order (siblings)
        3. They differ by exactly one node (parent-child)
        """
        n = len(self._all_trees)
        self._adjacency_matrix = np.zeros((n, n), dtype=np.int32)
        
        for i, iso_i in enumerate(self.isomorphisms):
            for j, iso_j in enumerate(self.isomorphisms):
                if i == j:
                    continue
                
                # Same order = siblings
                if iso_i.order == iso_j.order:
                    self._adjacency_matrix[i, j] = 1
                
                # Adjacent orders = potential parent-child
                elif abs(iso_i.order - iso_j.order) == 1:
                    self._adjacency_matrix[i, j] = 1
    
    def get_parameters(self) -> SynchronizedParameters:
        """Get all synchronized parameters."""
        return self.params
    
    def get_isomorphism(self, tree_index: int) -> TreeIsomorphism:
        """Get isomorphism by tree index."""
        return self.isomorphisms[tree_index]
    
    def get_isomorphism_by_membrane(self, membrane_id: str) -> Optional[TreeIsomorphism]:
        """Get isomorphism by membrane ID."""
        return self.membrane_map.get(membrane_id)
    
    def get_trees_by_order(self, order: int) -> List[RootedTree]:
        """Get all trees of a specific order."""
        return self.trees.get(order, [])
    
    def get_weight_for_tree(self, tree_index: int) -> float:
        """Get synchronized weight for a tree."""
        return self.isomorphisms[tree_index].weight
    
    def get_membrane_hierarchy(self) -> Dict[str, List[str]]:
        """Get membrane hierarchy as adjacency list.
        
        Returns
        -------
        Dict[str, List[str]]
            Parent membrane ID -> list of child membrane IDs
        """
        hierarchy = {"M_root": []}
        
        for iso in self.isomorphisms:
            if iso.order == 1:
                continue
            
            # Find parent (tree of order-1)
            parent_order = iso.order - 1
            parent_membranes = [
                other.membrane_id 
                for other in self.isomorphisms 
                if other.order == parent_order
            ]
            
            if parent_membranes:
                parent = parent_membranes[0]  # Simplified: use first
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(iso.membrane_id)
        
        return hierarchy
    
    def get_differential_sequence(self) -> List[Tuple[int, RootedTree, float]]:
        """Get elementary differential sequence.
        
        Returns
        -------
        List[Tuple[int, RootedTree, float]]
            List of (index, tree, coefficient) for differentials
        """
        return [
            (iso.differential_index, iso.tree, 1.0 / iso.tree.density)
            for iso in self.isomorphisms
        ]
    
    def synchronize_weights(
        self,
        ridge_weights: np.ndarray,
        reservoir_weights: np.ndarray,
        esn_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Synchronize weights across all three systems.
        
        Applies the A000081 isomorphism to ensure consistent structure.
        
        Parameters
        ----------
        ridge_weights : np.ndarray
            Ridge tree weights
        reservoir_weights : np.ndarray
            Membrane reservoir weights
        esn_weights : np.ndarray
            ESN weights
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Synchronized (ridge, reservoir, esn) weights
        """
        n = len(self.isomorphisms)
        
        # Apply synchronized weight matrix
        sync_factor = self._weight_matrix / (np.sum(self._weight_matrix) + 1e-8)
        
        # Synchronize each system
        if ridge_weights.shape[0] >= n:
            ridge_sync = ridge_weights[:n] * np.diag(sync_factor)
        else:
            ridge_sync = ridge_weights
        
        if reservoir_weights.shape[0] >= n:
            reservoir_sync = reservoir_weights[:n, :n] * sync_factor
        else:
            reservoir_sync = reservoir_weights
        
        if esn_weights.shape[0] >= n:
            esn_sync = esn_weights[:n, :n] * sync_factor
        else:
            esn_sync = esn_weights
        
        return ridge_sync, reservoir_sync, esn_sync
    
    def plant_tree_in_membrane(
        self,
        tree_index: int,
    ) -> Tuple[str, Dict]:
        """Plant a ridge tree's root in a membrane compartment.
        
        This is the key operation that connects the readout (ridge tree)
        to the reservoir (membrane).
        
        Parameters
        ----------
        tree_index : int
            Index of the tree to plant
            
        Returns
        -------
        Tuple[str, Dict]
            (membrane_id, connection_info)
        """
        iso = self.isomorphisms[tree_index]
        
        connection_info = {
            "tree": str(iso.tree),
            "membrane_id": iso.membrane_id,
            "weight": iso.weight,
            "depth": iso.depth,
            "order": iso.order,
            "differential_index": iso.differential_index,
        }
        
        return iso.membrane_id, connection_info
    
    def to_tensor_weights(self) -> Tuple[Tensor, Tensor]:
        """Convert weight and adjacency matrices to PyTorch tensors.
        
        Returns
        -------
        Tuple[Tensor, Tensor]
            (weight_tensor, adjacency_tensor)
        """
        if not TORCH_AVAILABLE:
            # Return numpy arrays when torch not available
            return self._weight_matrix.astype(np.float32), self._adjacency_matrix.astype(np.float32)
        
        weight_tensor = torch.from_numpy(self._weight_matrix).float()
        adj_tensor = torch.from_numpy(self._adjacency_matrix).float()
        
        return weight_tensor, adj_tensor
    
    def explain(self) -> str:
        """Generate explanation of the synchronization."""
        lines = [
            f"A000081 Synchronizer (base_order={self.base_order})",
            "=" * 60,
            "",
            f"Total trees: {len(self._all_trees)}",
            f"Tree counts by order: {self.params.tree_counts}",
            "",
            "Isomorphism Structure:",
            "-" * 40,
        ]
        
        for order in range(1, min(self.base_order + 1, 5)):
            trees = self.trees[order]
            lines.append(f"  Order {order}: {len(trees)} trees")
            for i, tree in enumerate(trees[:3]):
                iso = next(
                    iso for iso in self.isomorphisms 
                    if iso.tree == tree
                )
                lines.append(
                    f"    {tree} -> {iso.membrane_id} "
                    f"(w={iso.weight:.4f}, d={iso.depth})"
                )
            if len(trees) > 3:
                lines.append(f"    ... and {len(trees) - 3} more")
        
        lines.extend([
            "",
            "Weight Matrix Properties:",
            f"  Shape: {self._weight_matrix.shape}",
            f"  Sparsity: {np.mean(self._weight_matrix == 0):.2%}",
            f"  Max weight: {np.max(self._weight_matrix):.4f}",
            "",
            "Adjacency Matrix Properties:",
            f"  Connections: {np.sum(self._adjacency_matrix)}",
            f"  Density: {np.mean(self._adjacency_matrix):.2%}",
        ])
        
        return "\n".join(lines)
