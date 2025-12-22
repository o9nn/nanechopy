"""
====================================
Ontogenetic Engine (:mod:`reservoirpy.pytorch.autognosis.ontogenetic`)
====================================

Ontogenetic evolution engine for self-organizing reservoir computing.

The ontogenetic engine implements evolutionary parameter adaptation
based on the A000081 sequence and rooted tree structures. It enables
reservoirs to self-organize and adapt their topology over time.

Key Components
--------------
- OntogeneticState: Current state of evolution
- OntogeneticEngine: Main evolution controller
- MembraneGarden: Container for evolving tree populations
- JSurfaceReactor: Gradient-evolution dynamics
"""

# License: MIT License
# Copyright: nanechopy contributors (adapted from echo-jnn)

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any

try:
    from .a000081 import A000081Parameters, derive_parameters, get_a000081_value
    from .bseries import RootedTree, generate_trees, BSeriesKernel
except ImportError:
    from a000081 import A000081Parameters, derive_parameters, get_a000081_value
    from bseries import RootedTree, generate_trees, BSeriesKernel


class EvolutionPhase(Enum):
    """Phases of ontogenetic evolution."""
    INITIALIZATION = "initialization"
    GROWTH = "growth"
    PRUNING = "pruning"
    STABILIZATION = "stabilization"
    MATURATION = "maturation"


@dataclass
class OntogeneticState:
    """State of ontogenetic evolution.
    
    Attributes
    ----------
    generation : int
        Current generation number
    phase : EvolutionPhase
        Current evolution phase
    tree_population : List[RootedTree]
        Current population of trees
    fitness_history : List[float]
        History of fitness values
    best_fitness : float
        Best fitness achieved
    best_tree : Optional[RootedTree]
        Tree with best fitness
    parameters : A000081Parameters
        Current parameter set
    """
    generation: int = 0
    phase: EvolutionPhase = EvolutionPhase.INITIALIZATION
    tree_population: List[RootedTree] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    best_fitness: float = float('-inf')
    best_tree: Optional[RootedTree] = None
    parameters: Optional[A000081Parameters] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "phase": self.phase.value,
            "population_size": len(self.tree_population),
            "best_fitness": self.best_fitness,
            "best_tree": str(self.best_tree) if self.best_tree else None,
        }


@dataclass
class MembraneCompartment:
    """A membrane compartment in the P-system.
    
    Attributes
    ----------
    id : str
        Unique identifier
    trees : List[RootedTree]
        Trees in this compartment
    rules : List[Callable]
        Evolution rules
    parent : Optional[str]
        Parent compartment ID
    children : List[str]
        Child compartment IDs
    """
    id: str
    trees: List[RootedTree] = field(default_factory=list)
    rules: List[Callable] = field(default_factory=list)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


class MembraneGarden:
    """Garden of membrane compartments for tree evolution.
    
    Implements P-system membrane computing for evolving
    tree populations in separate compartments.
    
    Parameters
    ----------
    num_membranes : int
        Number of membrane compartments
    max_trees_per_membrane : int
        Maximum trees per compartment
    """
    
    def __init__(
        self,
        num_membranes: int = 4,
        max_trees_per_membrane: int = 10,
    ):
        self.num_membranes = num_membranes
        self.max_trees_per_membrane = max_trees_per_membrane
        
        # Create membrane structure
        self.membranes: Dict[str, MembraneCompartment] = {}
        self._create_membrane_structure()
    
    def _create_membrane_structure(self) -> None:
        """Create hierarchical membrane structure."""
        # Root membrane
        self.membranes["root"] = MembraneCompartment(id="root")
        
        # Child membranes
        for i in range(self.num_membranes - 1):
            membrane_id = f"m{i+1}"
            self.membranes[membrane_id] = MembraneCompartment(
                id=membrane_id,
                parent="root"
            )
            self.membranes["root"].children.append(membrane_id)
    
    def plant_tree(self, tree: RootedTree, membrane_id: str = "root") -> bool:
        """Plant a tree in a membrane.
        
        Parameters
        ----------
        tree : RootedTree
            Tree to plant
        membrane_id : str
            Target membrane
            
        Returns
        -------
        bool
            True if planted successfully
        """
        if membrane_id not in self.membranes:
            return False
        
        membrane = self.membranes[membrane_id]
        if len(membrane.trees) >= self.max_trees_per_membrane:
            return False
        
        membrane.trees.append(tree)
        return True
    
    def harvest_trees(self, membrane_id: str) -> List[RootedTree]:
        """Harvest all trees from a membrane.
        
        Parameters
        ----------
        membrane_id : str
            Source membrane
            
        Returns
        -------
        List[RootedTree]
            Harvested trees
        """
        if membrane_id not in self.membranes:
            return []
        
        trees = self.membranes[membrane_id].trees
        self.membranes[membrane_id].trees = []
        return trees
    
    def get_all_trees(self) -> List[RootedTree]:
        """Get all trees from all membranes."""
        all_trees = []
        for membrane in self.membranes.values():
            all_trees.extend(membrane.trees)
        return all_trees
    
    def apply_rules(self) -> None:
        """Apply evolution rules in all membranes."""
        for membrane in self.membranes.values():
            for rule in membrane.rules:
                membrane.trees = rule(membrane.trees)


class JSurfaceReactor:
    """J-Surface reactor for gradient-evolution dynamics.
    
    Combines gradient descent with evolutionary dynamics
    on a J-surface (symplectic manifold).
    
    Parameters
    ----------
    dim : int
        Dimension of the J-surface
    symplectic : bool
        Whether to use symplectic integration
    """
    
    def __init__(
        self,
        dim: int,
        symplectic: bool = True,
    ):
        self.dim = dim
        self.symplectic = symplectic
        
        # State variables
        self.position = np.zeros(dim)
        self.momentum = np.zeros(dim)
        
        # J-matrix (symplectic structure)
        self.J = np.zeros((2 * dim, 2 * dim))
        self.J[:dim, dim:] = np.eye(dim)
        self.J[dim:, :dim] = -np.eye(dim)
    
    def hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """Compute Hamiltonian H(q, p).
        
        Default: H = 0.5 * ||p||^2 + V(q)
        """
        kinetic = 0.5 * np.sum(p ** 2)
        potential = 0.5 * np.sum(q ** 2)  # Harmonic potential
        return kinetic + potential
    
    def gradient(self, q: np.ndarray) -> np.ndarray:
        """Compute gradient of potential."""
        return q  # For harmonic potential
    
    def step(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Advance state by one step.
        
        Uses symplectic Euler if symplectic=True.
        """
        if self.symplectic:
            # Symplectic Euler
            self.momentum = self.momentum - dt * self.gradient(self.position)
            self.position = self.position + dt * self.momentum
        else:
            # Standard Euler
            grad = self.gradient(self.position)
            self.position = self.position + dt * self.momentum
            self.momentum = self.momentum - dt * grad
        
        return self.position.copy(), self.momentum.copy()
    
    def evolve(self, n_steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve for multiple steps.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (positions, momenta) trajectories
        """
        positions = np.zeros((n_steps + 1, self.dim))
        momenta = np.zeros((n_steps + 1, self.dim))
        
        positions[0] = self.position
        momenta[0] = self.momentum
        
        for i in range(n_steps):
            q, p = self.step(dt)
            positions[i + 1] = q
            momenta[i + 1] = p
        
        return positions, momenta


class OntogeneticEngine:
    """Main ontogenetic evolution engine.
    
    Orchestrates the evolution of reservoir computing architectures
    using A000081-derived parameters and rooted tree structures.
    
    Parameters
    ----------
    base_order : int
        Base order for A000081 parameters
    population_size : int
        Size of tree population
    fitness_fn : Callable, optional
        Custom fitness function
    seed : int, optional
        Random seed
        
    Examples
    --------
    >>> engine = OntogeneticEngine(base_order=5)
    >>> engine.initialize()
    >>> 
    >>> # Evolve for 100 generations
    >>> for _ in range(100):
    ...     engine.evolve_step()
    >>> 
    >>> # Get best configuration
    >>> best = engine.get_best_configuration()
    """
    
    def __init__(
        self,
        base_order: int = 5,
        population_size: int = 20,
        fitness_fn: Optional[Callable] = None,
        seed: Optional[int] = None,
    ):
        self.base_order = base_order
        self.population_size = population_size
        self.fitness_fn = fitness_fn or self._default_fitness
        
        # Initialize RNG
        self.rng = np.random.default_rng(seed)
        
        # Derive parameters
        self.params = derive_parameters(base_order)
        
        # Create components
        self.garden = MembraneGarden(num_membranes=self.params.num_membranes)
        self.reactor = JSurfaceReactor(dim=self.params.reservoir_size)
        self.bseries = BSeriesKernel(max_order=self.params.max_tree_order)
        
        # State
        self.state = OntogeneticState(parameters=self.params)
    
    def _default_fitness(self, tree: RootedTree) -> float:
        """Default fitness function based on tree properties."""
        # Prefer trees with moderate order and low symmetry
        order_score = 1.0 / (1.0 + abs(tree.order - self.base_order))
        symmetry_score = 1.0 / tree.symmetry
        density_score = 1.0 / tree.density
        
        return order_score + 0.5 * symmetry_score + 0.3 * density_score
    
    def initialize(self) -> None:
        """Initialize the evolution."""
        self.state.phase = EvolutionPhase.INITIALIZATION
        self.state.generation = 0
        
        # Generate initial population
        for order in range(1, self.base_order + 1):
            trees = generate_trees(order)
            for tree in trees[:self.population_size // self.base_order]:
                self.state.tree_population.append(tree)
                # Plant in random membrane
                membrane_id = self.rng.choice(list(self.garden.membranes.keys()))
                self.garden.plant_tree(tree, membrane_id)
        
        # Ensure minimum population
        while len(self.state.tree_population) < self.population_size:
            order = self.rng.integers(1, self.base_order + 1)
            trees = generate_trees(order)
            if trees:
                tree = self.rng.choice(trees)
                self.state.tree_population.append(tree)
        
        self.state.phase = EvolutionPhase.GROWTH
    
    def evolve_step(self) -> float:
        """Perform one evolution step.
        
        Returns
        -------
        float
            Best fitness in this generation
        """
        self.state.generation += 1
        
        # Evaluate fitness
        fitness_values = [self.fitness_fn(tree) for tree in self.state.tree_population]
        
        # Update best
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.state.best_fitness:
            self.state.best_fitness = fitness_values[best_idx]
            self.state.best_tree = self.state.tree_population[best_idx]
        
        self.state.fitness_history.append(max(fitness_values))
        
        # Selection
        sorted_indices = np.argsort(fitness_values)[::-1]
        survivors = [self.state.tree_population[i] for i in sorted_indices[:self.population_size // 2]]
        
        # Mutation
        new_population = list(survivors)
        while len(new_population) < self.population_size:
            parent = self.rng.choice(survivors)
            child = self._mutate_tree(parent)
            new_population.append(child)
        
        self.state.tree_population = new_population
        
        # Update phase
        self._update_phase()
        
        return self.state.best_fitness
    
    def _mutate_tree(self, tree: RootedTree) -> RootedTree:
        """Mutate a tree to create offspring."""
        mutation_type = self.rng.choice(['grow', 'prune', 'swap'])
        
        if mutation_type == 'grow' and tree.order < self.params.max_tree_order:
            # Add a child
            new_child = RootedTree([])
            new_children = list(tree.children) + [new_child]
            return RootedTree(new_children)
        
        elif mutation_type == 'prune' and tree.children:
            # Remove a child
            idx = self.rng.integers(len(tree.children))
            new_children = tree.children[:idx] + tree.children[idx+1:]
            return RootedTree(new_children)
        
        elif mutation_type == 'swap' and len(tree.children) >= 2:
            # Swap two children
            new_children = list(tree.children)
            i, j = self.rng.choice(len(new_children), size=2, replace=False)
            new_children[i], new_children[j] = new_children[j], new_children[i]
            return RootedTree(new_children)
        
        return tree  # No mutation
    
    def _update_phase(self) -> None:
        """Update evolution phase based on progress."""
        gen = self.state.generation
        
        if gen < 10:
            self.state.phase = EvolutionPhase.GROWTH
        elif gen < 30:
            self.state.phase = EvolutionPhase.PRUNING
        elif gen < 50:
            self.state.phase = EvolutionPhase.STABILIZATION
        else:
            self.state.phase = EvolutionPhase.MATURATION
    
    def get_best_configuration(self) -> Dict:
        """Get the best configuration found.
        
        Returns
        -------
        Dict
            Configuration dictionary
        """
        return {
            "tree": str(self.state.best_tree) if self.state.best_tree else None,
            "fitness": self.state.best_fitness,
            "generation": self.state.generation,
            "phase": self.state.phase.value,
            "parameters": self.params.to_dict(),
        }
    
    def get_reservoir_config(self) -> Dict:
        """Get reservoir configuration based on evolution.
        
        Returns
        -------
        Dict
            Reservoir configuration
        """
        return {
            "units": self.params.reservoir_size,
            "spectral_radius": self.params.spectral_radius,
            "leak_rate": self.params.leak_rate,
            "input_scaling": self.params.input_scaling,
            "ridge": self.params.ridge_param,
        }
