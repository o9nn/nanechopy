"""
====================================
P-System Membrane Reservoir (:mod:`reservoirpy.pytorch.dtesnn.membrane_reservoir`)
====================================

P-System Membrane Nests as Reservoir nodes.

In the DTESNN architecture, the reservoir layer is structured as a
hierarchical P-system of membrane compartments. Each membrane contains
reservoir units, and communication between membranes follows rules
derived from the A000081 tree structure.

Key Concepts
------------
- **Membrane Compartment**: A reservoir region with internal dynamics
- **Membrane Nest**: Hierarchical organization of compartments
- **Communication Rules**: Tree-based message passing between membranes
- **Tree Planting**: Ridge trees are planted in membrane compartments

P-System Properties
-------------------
1. Hierarchical structure (nested membranes)
2. Parallel evolution of compartments
3. Communication via symbol objects
4. Dissolution and division rules

A000081 Correspondence
----------------------
- A000081[n] membranes at nesting level n
- Tree structure defines membrane hierarchy
- Communication rules from tree adjacency
"""

# License: MIT License
# Copyright: nanechopy contributors

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
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
    torch = None
    nn = None

try:
    from ..autognosis.bseries import RootedTree, generate_trees
    from ..autognosis.a000081 import get_a000081_value, cumulative_a000081
    from ..node import TorchNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor
except ImportError:
    try:
        from bseries import RootedTree, generate_trees
        from a000081 import get_a000081_value, cumulative_a000081
    except ImportError:
        import sys
        import os
        autognosis_path = os.path.join(os.path.dirname(__file__), '..', 'autognosis')
        sys.path.insert(0, autognosis_path)
        from bseries import RootedTree, generate_trees
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
    TorchState = dict
    TorchTimestep = np.ndarray
    TorchTimeseries = np.ndarray
    def to_tensor(x): return x


class CommunicationRule(Enum):
    """Types of membrane communication rules."""
    SYMPORT = "symport"      # Objects move together
    ANTIPORT = "antiport"    # Objects exchange
    DISSOLVE = "dissolve"    # Membrane dissolves
    DIVIDE = "divide"        # Membrane divides
    BROADCAST = "broadcast"  # Send to all children


@dataclass
class MembraneCompartment:
    """A single membrane compartment in the P-system.
    
    Each compartment contains reservoir units and maintains internal
    state that evolves according to ESN dynamics.
    
    Attributes
    ----------
    membrane_id : str
        Unique identifier for this membrane
    order : int
        Nesting level (1 = outermost)
    units : int
        Number of reservoir units
    parent_id : Optional[str]
        ID of parent membrane (None for skin membrane)
    children_ids : List[str]
        IDs of child membranes
    tree : Optional[RootedTree]
        Associated rooted tree from A000081
    state : np.ndarray
        Current reservoir state
    W : np.ndarray
        Internal recurrent weights
    W_in : np.ndarray
        Input weights
    W_comm : Dict[str, np.ndarray]
        Communication weights to other membranes
    leak_rate : float
        Leaking rate for ESN dynamics
    spectral_radius : float
        Spectral radius of W
    planted_trees : List[int]
        Indices of ridge trees planted here
    """
    membrane_id: str
    order: int
    units: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    tree: Optional[RootedTree] = None
    state: Optional[np.ndarray] = None
    W: Optional[np.ndarray] = None
    W_in: Optional[np.ndarray] = None
    W_comm: Dict[str, np.ndarray] = field(default_factory=dict)
    leak_rate: float = 0.3
    spectral_radius: float = 0.9
    planted_trees: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize state if not provided."""
        if self.state is None:
            self.state = np.zeros(self.units)
    
    def initialize_weights(
        self,
        input_dim: int,
        density: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize reservoir weights.
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        density : float
            Sparsity of recurrent weights
        seed : int, optional
            Random seed
        """
        rng = np.random.default_rng(seed)
        
        # Input weights (dense)
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
    
    def step(
        self,
        x: np.ndarray,
        comm_inputs: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Evolve membrane state for one timestep.
        
        Parameters
        ----------
        x : np.ndarray
            External input
        comm_inputs : Dict[str, np.ndarray], optional
            Inputs from other membranes
            
        Returns
        -------
        np.ndarray
            New state
        """
        # Input contribution
        u = self.W_in @ x
        
        # Recurrent contribution
        r = self.W @ self.state
        
        # Communication contribution
        c = np.zeros(self.units)
        if comm_inputs:
            for membrane_id, comm_state in comm_inputs.items():
                if membrane_id in self.W_comm:
                    c += self.W_comm[membrane_id] @ comm_state
        
        # ESN update with leaky integration
        pre_activation = u + r + c
        new_state = (1 - self.leak_rate) * self.state + self.leak_rate * np.tanh(pre_activation)
        
        self.state = new_state
        return new_state
    
    def reset(self) -> None:
        """Reset membrane state."""
        self.state = np.zeros(self.units)


@dataclass
class MembraneNest:
    """A hierarchical nest of membrane compartments.
    
    Represents the full P-system structure with nested membranes
    organized according to A000081 tree enumeration.
    
    Attributes
    ----------
    skin_id : str
        ID of outermost (skin) membrane
    membranes : Dict[str, MembraneCompartment]
        All membrane compartments
    hierarchy : Dict[str, List[str]]
        Parent -> children mapping
    trees : Dict[int, List[RootedTree]]
        Trees by order
    communication_rules : List[Tuple[str, str, CommunicationRule]]
        Active communication rules
    """
    skin_id: str = "M_skin"
    membranes: Dict[str, MembraneCompartment] = field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    trees: Dict[int, List[RootedTree]] = field(default_factory=dict)
    communication_rules: List[Tuple[str, str, CommunicationRule]] = field(default_factory=list)
    
    @property
    def total_units(self) -> int:
        """Total reservoir units across all membranes."""
        return sum(m.units for m in self.membranes.values())
    
    def get_membrane(self, membrane_id: str) -> Optional[MembraneCompartment]:
        """Get membrane by ID."""
        return self.membranes.get(membrane_id)
    
    def get_children(self, membrane_id: str) -> List[MembraneCompartment]:
        """Get child membranes."""
        child_ids = self.hierarchy.get(membrane_id, [])
        return [self.membranes[cid] for cid in child_ids if cid in self.membranes]
    
    def get_parent(self, membrane_id: str) -> Optional[MembraneCompartment]:
        """Get parent membrane."""
        membrane = self.membranes.get(membrane_id)
        if membrane and membrane.parent_id:
            return self.membranes.get(membrane.parent_id)
        return None
    
    def get_all_states(self) -> np.ndarray:
        """Get concatenated state from all membranes."""
        states = [m.state for m in self.membranes.values()]
        return np.concatenate(states)
    
    def set_all_states(self, state: np.ndarray) -> None:
        """Set states for all membranes from concatenated vector."""
        offset = 0
        for membrane in self.membranes.values():
            membrane.state = state[offset:offset + membrane.units]
            offset += membrane.units
    
    def reset(self) -> None:
        """Reset all membrane states."""
        for membrane in self.membranes.values():
            membrane.reset()


class MembraneReservoir(TorchNode):
    """P-System Membrane Reservoir node.
    
    A reservoir computing layer structured as a P-system of nested
    membrane compartments. The membrane hierarchy is derived from
    A000081 rooted tree enumeration.
    
    Parameters
    ----------
    base_order : int
        Maximum tree order (determines membrane structure)
    units_per_membrane : int
        Reservoir units per membrane compartment
    total_units : int, optional
        Total units (overrides units_per_membrane)
    input_dim : int, optional
        Input dimension (inferred if not provided)
    leak_rate : float, default 0.3
        Leaking rate for ESN dynamics
    spectral_radius : float, default 0.9
        Spectral radius of recurrent weights
    density : float, default 0.1
        Sparsity of recurrent weights
    communication_strength : float, default 0.1
        Strength of inter-membrane communication
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
    nest : MembraneNest
        The P-system membrane structure
    membrane_order : List[str]
        Order of membrane processing
        
    Examples
    --------
    >>> reservoir = MembraneReservoir(base_order=5, units_per_membrane=50)
    >>> reservoir.initialize(input_data)
    >>> states = reservoir.run(input_sequence)
    """
    
    def __init__(
        self,
        base_order: int = 5,
        units_per_membrane: int = 50,
        total_units: Optional[int] = None,
        input_dim: Optional[int] = None,
        leak_rate: float = 0.3,
        spectral_radius: float = 0.9,
        density: float = 0.1,
        communication_strength: float = 0.1,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.base_order = base_order
        self.units_per_membrane = units_per_membrane
        self._total_units = total_units
        self.input_dim = input_dim
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.communication_strength = communication_strength
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        
        # Generate trees
        self.trees: Dict[int, List[RootedTree]] = {}
        self._all_trees: List[RootedTree] = []
        for order in range(1, base_order + 1):
            self.trees[order] = generate_trees(order)
            self._all_trees.extend(self.trees[order])
        
        # Membrane nest (created on initialize)
        self.nest: Optional[MembraneNest] = None
        self.membrane_order: List[str] = []
    
    def _build_membrane_structure(self) -> MembraneNest:
        """Build P-system membrane structure from A000081 trees."""
        nest = MembraneNest()
        nest.trees = self.trees
        
        # Calculate units per membrane
        num_membranes = len(self._all_trees)
        if self._total_units:
            units_per = self._total_units // num_membranes
        else:
            units_per = self.units_per_membrane
        
        # Create skin membrane (contains all others)
        skin = MembraneCompartment(
            membrane_id="M_skin",
            order=0,
            units=units_per,
            leak_rate=self.leak_rate,
            spectral_radius=self.spectral_radius,
        )
        nest.membranes["M_skin"] = skin
        nest.hierarchy["M_skin"] = []
        
        # Create membrane for each tree
        tree_to_membrane: Dict[int, str] = {}
        
        for i, tree in enumerate(self._all_trees):
            membrane_id = f"M_{tree.order}_{i}"
            
            # Adjust parameters based on tree properties
            tree_sr = self.spectral_radius * (1 - 1.0 / (tree.density + 1))
            tree_lr = self.leak_rate * (1 + 1.0 / tree.symmetry)
            
            membrane = MembraneCompartment(
                membrane_id=membrane_id,
                order=tree.order,
                units=units_per,
                tree=tree,
                leak_rate=min(tree_lr, 1.0),
                spectral_radius=tree_sr,
            )
            
            nest.membranes[membrane_id] = membrane
            tree_to_membrane[i] = membrane_id
        
        # Build hierarchy based on tree order
        # Trees of order n are children of trees of order n-1
        for i, tree in enumerate(self._all_trees):
            membrane_id = tree_to_membrane[i]
            membrane = nest.membranes[membrane_id]
            
            if tree.order == 1:
                # Order 1 trees are children of skin
                membrane.parent_id = "M_skin"
                nest.hierarchy["M_skin"].append(membrane_id)
            else:
                # Find parent (tree of order-1)
                parent_order = tree.order - 1
                parent_trees = [
                    j for j, t in enumerate(self._all_trees)
                    if t.order == parent_order
                ]
                if parent_trees:
                    parent_idx = parent_trees[i % len(parent_trees)]
                    parent_id = tree_to_membrane[parent_idx]
                    membrane.parent_id = parent_id
                    
                    if parent_id not in nest.hierarchy:
                        nest.hierarchy[parent_id] = []
                    nest.hierarchy[parent_id].append(membrane_id)
        
        # Set children IDs
        for membrane_id, children in nest.hierarchy.items():
            if membrane_id in nest.membranes:
                nest.membranes[membrane_id].children_ids = children
        
        return nest
    
    def _initialize_communication(self) -> None:
        """Initialize communication weights between membranes."""
        for membrane_id, membrane in self.nest.membranes.items():
            # Communication with parent
            if membrane.parent_id and membrane.parent_id in self.nest.membranes:
                parent = self.nest.membranes[membrane.parent_id]
                W_comm = self.rng.standard_normal((membrane.units, parent.units))
                W_comm *= self.communication_strength / np.sqrt(parent.units)
                membrane.W_comm[membrane.parent_id] = W_comm
            
            # Communication with children
            for child_id in membrane.children_ids:
                if child_id in self.nest.membranes:
                    child = self.nest.membranes[child_id]
                    W_comm = self.rng.standard_normal((membrane.units, child.units))
                    W_comm *= self.communication_strength / np.sqrt(child.units)
                    membrane.W_comm[child_id] = W_comm
            
            # Communication with siblings (same order)
            siblings = [
                m_id for m_id, m in self.nest.membranes.items()
                if m.order == membrane.order and m_id != membrane_id
            ]
            for sibling_id in siblings[:2]:  # Limit to 2 siblings
                sibling = self.nest.membranes[sibling_id]
                W_comm = self.rng.standard_normal((membrane.units, sibling.units))
                W_comm *= self.communication_strength * 0.5 / np.sqrt(sibling.units)
                membrane.W_comm[sibling_id] = W_comm
    
    def _determine_processing_order(self) -> List[str]:
        """Determine order for processing membranes.
        
        Uses topological sort based on hierarchy.
        """
        order = []
        visited = set()
        
        def visit(membrane_id: str):
            if membrane_id in visited:
                return
            visited.add(membrane_id)
            
            # Visit children first (bottom-up)
            for child_id in self.nest.hierarchy.get(membrane_id, []):
                visit(child_id)
            
            order.append(membrane_id)
        
        visit("M_skin")
        return order
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize the membrane reservoir."""
        self._set_input_dim(x)
        
        # Build membrane structure
        self.nest = self._build_membrane_structure()
        
        # Initialize weights for each membrane
        for membrane in self.nest.membranes.values():
            membrane.initialize_weights(
                input_dim=self.input_dim,
                density=self.density,
                seed=self.rng.integers(0, 2**31),
            )
        
        # Initialize communication weights
        self._initialize_communication()
        
        # Determine processing order
        self.membrane_order = self._determine_processing_order()
        
        # Set output dimension
        self.output_dim = self.nest.total_units
        
        # Initialize state
        if TORCH_AVAILABLE:
            self.state = {
                "reservoir": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "reservoir": np.zeros(self.output_dim),
                "out": np.zeros(self.output_dim),
            }
        
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep through membrane reservoir."""
        # Convert to numpy
        if TORCH_AVAILABLE and isinstance(x, Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        
        # Collect current states for communication
        current_states = {
            m_id: m.state.copy()
            for m_id, m in self.nest.membranes.items()
        }
        
        # Process each membrane in order
        for membrane_id in self.membrane_order:
            membrane = self.nest.membranes[membrane_id]
            
            # Gather communication inputs
            comm_inputs = {
                other_id: current_states[other_id]
                for other_id in membrane.W_comm.keys()
                if other_id in current_states
            }
            
            # Step membrane
            membrane.step(x_np, comm_inputs)
        
        # Collect all states
        reservoir_state = self.nest.get_all_states()
        
        # Convert back to tensor
        if TORCH_AVAILABLE:
            reservoir_tensor = torch.from_numpy(reservoir_state).to(
                device=self.device, dtype=self.dtype
            )
            return {
                "reservoir": reservoir_tensor,
                "out": reservoir_tensor,
            }
        else:
            return {
                "reservoir": reservoir_state,
                "out": reservoir_state,
            }
    
    def _run(
        self,
        state: TorchState,
        x: TorchTimeseries,
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Process sequence through membrane reservoir."""
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
        
        return {"reservoir": final_out, "out": final_out}, outputs
    
    def plant_tree(self, tree_index: int, membrane_id: str) -> bool:
        """Plant a ridge tree in a membrane compartment.
        
        Parameters
        ----------
        tree_index : int
            Index of tree to plant
        membrane_id : str
            Target membrane ID
            
        Returns
        -------
        bool
            True if successful
        """
        if membrane_id not in self.nest.membranes:
            return False
        
        membrane = self.nest.membranes[membrane_id]
        if tree_index not in membrane.planted_trees:
            membrane.planted_trees.append(tree_index)
        
        return True
    
    def get_membrane_states(self) -> Dict[str, np.ndarray]:
        """Get states organized by membrane."""
        return {
            m_id: m.state.copy()
            for m_id, m in self.nest.membranes.items()
        }
    
    def get_structure_info(self) -> str:
        """Get string representation of membrane structure."""
        lines = [
            f"P-System Membrane Reservoir (base_order={self.base_order})",
            "=" * 50,
            f"Total membranes: {len(self.nest.membranes)}",
            f"Total units: {self.nest.total_units}",
            "",
            "Membrane Hierarchy:",
        ]
        
        def print_membrane(membrane_id: str, indent: int = 0):
            membrane = self.nest.membranes.get(membrane_id)
            if not membrane:
                return
            
            prefix = "  " * indent
            tree_str = str(membrane.tree) if membrane.tree else "N/A"
            lines.append(
                f"{prefix}{membrane_id}: {membrane.units} units, "
                f"tree={tree_str}, planted={len(membrane.planted_trees)}"
            )
            
            for child_id in self.nest.hierarchy.get(membrane_id, []):
                print_membrane(child_id, indent + 1)
        
        print_membrane("M_skin")
        
        return "\n".join(lines)
    
    def reset(self) -> TorchState:
        """Reset reservoir state."""
        previous_state = self.state
        self.nest.reset()
        
        if TORCH_AVAILABLE:
            self.state = {
                "reservoir": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
                "out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype),
            }
        else:
            self.state = {
                "reservoir": np.zeros(self.output_dim),
                "out": np.zeros(self.output_dim),
            }
        
        return previous_state
