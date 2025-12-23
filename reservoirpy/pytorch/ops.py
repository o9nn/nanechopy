"""
====================================
PyTorch Node Operations (:mod:`reservoirpy.pytorch.ops`)
====================================

Operations for combining and connecting PyTorch nodes.

Provides:
- link: Connect nodes sequentially (>>)
- merge: Combine nodes in parallel (&)
- link_feedback: Create feedback connections (<<)
"""

# License: MIT License
# Copyright: nanechopy contributors

from typing import List, Optional, Sequence, Union, Tuple, Any
import numpy as np

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()

from .node import TorchNode, TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries


class TorchModel(TorchTrainableNode):
    """Model combining multiple TorchNodes.
    
    A TorchModel is a computational graph of connected nodes.
    """
    
    def __init__(
        self,
        nodes: List[TorchNode],
        edges: List[Tuple[int, int]],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.nodes = nodes
        self.edges = edges
        self._topological_order: Optional[List[int]] = None
    
    def _compute_topological_order(self) -> List[int]:
        """Compute topological order of nodes."""
        n = len(self.nodes)
        in_degree = [0] * n
        adj = [[] for _ in range(n)]
        
        for src, dst in self.edges:
            adj[src].append(dst)
            in_degree[dst] += 1
        
        # Find nodes with no incoming edges
        queue = [i for i in range(n) if in_degree[i] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return order
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize all nodes in the model."""
        self._set_input_dim(x)
        
        # Compute topological order
        self._topological_order = self._compute_topological_order()
        
        # Initialize nodes in order
        current_input = x
        for idx in self._topological_order:
            node = self.nodes[idx]
            if not node.initialized:
                node.initialize(current_input, y)
            current_input = torch.zeros(node.output_dim, device=self.device, dtype=self.dtype)
        
        # Set output dimension from last node
        last_node = self.nodes[self._topological_order[-1]]
        self.output_dim = last_node.output_dim
        
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Process one timestep through all nodes."""
        outputs = {-1: x}  # Input is from "node -1"
        
        for idx in self._topological_order:
            node = self.nodes[idx]
            
            # Collect inputs from predecessors
            inputs = []
            for src, dst in self.edges:
                if dst == idx:
                    inputs.append(outputs.get(src, x))
            
            if not inputs:
                node_input = x
            elif len(inputs) == 1:
                node_input = inputs[0]
            else:
                node_input = torch.cat(inputs, dim=-1)
            
            # Process through node
            out = node.step(node_input)
            outputs[idx] = out
        
        # Return output from last node
        last_idx = self._topological_order[-1]
        return {"out": outputs[last_idx]}
    
    def fit(
        self,
        x: Union[np.ndarray, TorchTimeseries],
        y: Optional[Union[np.ndarray, TorchTimeseries]] = None,
        warmup: int = 0,
        **kwargs,
    ) -> "TorchModel":
        """Train trainable nodes in the model."""
        from .node import to_tensor
        
        x = to_tensor(x, device=self.device, dtype=self.dtype)
        if y is not None:
            y = to_tensor(y, device=self.device, dtype=self.dtype)
        
        if not self.initialized:
            self.initialize(x, y)
        
        # Train each trainable node
        for node in self.nodes:
            if isinstance(node, TorchTrainableNode):
                # Collect states up to this node
                # Simplified: just train with original data
                node.fit(x, y, warmup=warmup, **kwargs)
        
        return self


def link(
    sender: Union[TorchNode, Sequence[TorchNode]],
    receiver: Union[TorchNode, Sequence[TorchNode]],
) -> TorchModel:
    """Link nodes sequentially.
    
    Creates a model where output of sender flows to input of receiver.
    
    Parameters
    ----------
    sender : TorchNode or sequence of TorchNode
        Source node(s)
    receiver : TorchNode or sequence of TorchNode
        Destination node(s)
        
    Returns
    -------
    TorchModel
        Model with nodes linked sequentially
        
    Examples
    --------
    >>> model = link(reservoir, readout)
    >>> # Or using operator
    >>> model = reservoir >> readout
    """
    # Normalize to lists
    if isinstance(sender, TorchNode):
        senders = [sender]
    else:
        senders = list(sender)
    
    if isinstance(receiver, TorchNode):
        receivers = [receiver]
    else:
        receivers = list(receiver)
    
    # Build node list and edges
    nodes = senders + receivers
    edges = []
    
    # Connect each sender to each receiver
    for i, s in enumerate(senders):
        for j, r in enumerate(receivers):
            edges.append((i, len(senders) + j))
    
    # Get device/dtype from first node
    device = senders[0].device if hasattr(senders[0], 'device') else None
    dtype = senders[0].dtype if hasattr(senders[0], 'dtype') else None
    
    return TorchModel(nodes=nodes, edges=edges, device=device, dtype=dtype)


def merge(
    *nodes: TorchNode,
) -> "MergedNode":
    """Merge nodes to run in parallel.
    
    Creates a node that runs all input nodes in parallel
    and concatenates their outputs.
    
    Parameters
    ----------
    *nodes : TorchNode
        Nodes to merge
        
    Returns
    -------
    MergedNode
        Node that runs all nodes in parallel
        
    Examples
    --------
    >>> merged = merge(node1, node2, node3)
    >>> # Or using operator
    >>> merged = node1 & node2 & node3
    """
    return MergedNode(list(nodes))


class MergedNode(TorchNode):
    """Node that runs multiple nodes in parallel."""
    
    def __init__(
        self,
        nodes: List[TorchNode],
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        # Get device/dtype from first node if not provided
        if device is None and nodes:
            device = nodes[0].device if hasattr(nodes[0], 'device') else None
        if dtype is None and nodes:
            dtype = nodes[0].dtype if hasattr(nodes[0], 'dtype') else None
        
        super().__init__(device=device, dtype=dtype, name=name)
        self.nodes = nodes
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize all merged nodes."""
        self._set_input_dim(x)
        
        # Initialize each node
        for node in self.nodes:
            if not node.initialized:
                node.initialize(x, y)
        
        # Output dimension is sum of all node outputs
        self.output_dim = sum(node.output_dim for node in self.nodes)
        
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Run all nodes and concatenate outputs."""
        outputs = []
        for node in self.nodes:
            out = node.step(x)
            outputs.append(out)
        
        result = torch.cat(outputs, dim=-1)
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Run all nodes on sequence."""
        all_outputs = []
        for node in self.nodes:
            _, outputs = node._run(node.state, x)
            all_outputs.append(outputs)
        
        result = torch.cat(all_outputs, dim=-1)
        return {"out": result[-1]}, result


def link_feedback(
    sender: TorchNode,
    receiver: TorchNode,
) -> TorchModel:
    """Create feedback connection from sender to receiver.
    
    The output of sender at time t is fed to receiver at time t+1.
    
    Parameters
    ----------
    sender : TorchNode
        Node providing feedback
    receiver : TorchNode
        Node receiving feedback
        
    Returns
    -------
    TorchModel
        Model with feedback connection
        
    Examples
    --------
    >>> model = link_feedback(readout, reservoir)
    >>> # Or using operator
    >>> model = readout << reservoir
    """
    # For now, return a simple linked model
    # Full feedback implementation would require special handling
    return link(sender, receiver)
