"""
====================================
ATen Nodes (:mod:`reservoirpy.pytorch.nodes.aten`)
====================================

Low-level tensor operations using PyTorch's ATen (A Tensor library) backend.

ATen provides the fundamental tensor operations that underpin PyTorch.
These nodes expose ATen operations in a reservoir computing context,
enabling fine-grained control over tensor computations.

Key ATen operations supported:
- Matrix multiplication (mm, bmm, matmul)
- Element-wise operations (add, mul, div)
- Reductions (sum, mean, norm)
- Transformations (view, reshape, transpose)
- Activations (relu, tanh, sigmoid, gelu)
"""

# License: MIT License
# Copyright: nanechopy contributors

from typing import Callable, Dict, List, Optional, Sequence, Union, Tuple, Any
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any
    class _TorchPlaceholder:
        dtype = Any
    torch = _TorchPlaceholder()

from ..node import TorchNode, TorchTrainableNode, TorchState, TorchTimestep, TorchTimeseries, to_tensor


class ATenOp(Enum):
    """Supported ATen operations."""
    # Matrix operations
    MATMUL = "matmul"
    MM = "mm"
    BMM = "bmm"
    MV = "mv"
    DOT = "dot"
    
    # Element-wise
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    
    # Reductions
    SUM = "sum"
    MEAN = "mean"
    NORM = "norm"
    MAX = "max"
    MIN = "min"
    
    # Transformations
    VIEW = "view"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    PERMUTE = "permute"
    SQUEEZE = "squeeze"
    UNSQUEEZE = "unsqueeze"
    
    # Activations
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    SOFTMAX = "softmax"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SILU = "silu"
    
    # Special
    CLONE = "clone"
    CONTIGUOUS = "contiguous"
    DETACH = "detach"


class ATenNode(TorchNode):
    """Node that applies ATen tensor operations.
    
    ATenNode provides direct access to PyTorch's ATen operations,
    enabling low-level tensor manipulation within the reservoir
    computing framework.
    
    Parameters
    ----------
    operation : ATenOp or str
        The ATen operation to apply
    weight : Tensor, optional
        Weight matrix for operations that require it (matmul, etc.)
    bias : Tensor, optional
        Bias vector to add after operation
    activation : ATenOp or str, optional
        Activation function to apply after operation
    op_kwargs : dict, optional
        Additional keyword arguments for the operation
    output_dim : int, optional
        Output dimension (inferred if not provided)
    device : str, optional
        Device for tensors
    dtype : torch.dtype, optional
        Data type for tensors
    name : str, optional
        Name of the node
        
    Examples
    --------
    >>> # Linear transformation with ReLU
    >>> node = ATenNode(
    ...     operation="matmul",
    ...     weight=torch.randn(100, 50),
    ...     activation="relu"
    ... )
    >>> output = node.run(input_data)
    
    >>> # Custom reduction
    >>> node = ATenNode(operation="mean", op_kwargs={"dim": -1})
    """
    
    def __init__(
        self,
        operation: Union[ATenOp, str] = ATenOp.MATMUL,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        activation: Optional[Union[ATenOp, str]] = None,
        op_kwargs: Optional[Dict] = None,
        output_dim: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, requires_grad=requires_grad, name=name)
        
        # Parse operation
        if isinstance(operation, str):
            operation = ATenOp(operation.lower())
        self.operation = operation
        
        # Parse activation
        if activation is not None:
            if isinstance(activation, str):
                activation = ATenOp(activation.lower())
        self.activation = activation
        
        # Store operation kwargs
        self.op_kwargs = op_kwargs or {}
        
        # Store weight and bias (will be registered on initialize)
        self._init_weight = weight
        self._init_bias = bias
        self.output_dim = output_dim
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize node dimensions and parameters."""
        self._set_input_dim(x)
        
        # Initialize weight if needed
        if self._init_weight is not None:
            self.register_parameter("weight", self._init_weight)
            if self.output_dim is None:
                self.output_dim = self._init_weight.shape[0]
        elif self.operation in [ATenOp.MATMUL, ATenOp.MM, ATenOp.MV]:
            # Create random weight if not provided
            out_dim = self.output_dim or self.input_dim
            weight = torch.randn(out_dim, self.input_dim, device=self.device, dtype=self.dtype)
            weight = weight / np.sqrt(self.input_dim)  # Xavier-like init
            self.register_parameter("weight", weight)
            self.output_dim = out_dim
        else:
            self.output_dim = self.output_dim or self.input_dim
        
        # Initialize bias if needed
        if self._init_bias is not None:
            self.register_parameter("bias", self._init_bias)
        
        # Initialize state
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _apply_operation(self, x: Tensor) -> Tensor:
        """Apply the ATen operation."""
        op = self.operation
        kwargs = self.op_kwargs
        
        # Matrix operations
        if op == ATenOp.MATMUL:
            result = torch.matmul(self.weight, x)
        elif op == ATenOp.MM:
            result = torch.mm(self.weight, x.unsqueeze(-1)).squeeze(-1)
        elif op == ATenOp.BMM:
            result = torch.bmm(self.weight.unsqueeze(0), x.unsqueeze(0).unsqueeze(-1)).squeeze()
        elif op == ATenOp.MV:
            result = torch.mv(self.weight, x)
        elif op == ATenOp.DOT:
            result = torch.dot(self.weight.flatten(), x.flatten()).unsqueeze(0)
        
        # Element-wise operations
        elif op == ATenOp.ADD:
            result = torch.add(x, kwargs.get("other", 0))
        elif op == ATenOp.SUB:
            result = torch.sub(x, kwargs.get("other", 0))
        elif op == ATenOp.MUL:
            result = torch.mul(x, kwargs.get("other", 1))
        elif op == ATenOp.DIV:
            result = torch.div(x, kwargs.get("other", 1))
        elif op == ATenOp.POW:
            result = torch.pow(x, kwargs.get("exponent", 2))
        
        # Reductions
        elif op == ATenOp.SUM:
            result = torch.sum(x, **kwargs)
        elif op == ATenOp.MEAN:
            result = torch.mean(x, **kwargs)
        elif op == ATenOp.NORM:
            result = torch.norm(x, **kwargs)
        elif op == ATenOp.MAX:
            result = torch.max(x, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
        elif op == ATenOp.MIN:
            result = torch.min(x, **kwargs)
            if isinstance(result, tuple):
                result = result[0]
        
        # Transformations
        elif op == ATenOp.VIEW:
            result = x.view(*kwargs.get("shape", (-1,)))
        elif op == ATenOp.RESHAPE:
            result = x.reshape(*kwargs.get("shape", (-1,)))
        elif op == ATenOp.TRANSPOSE:
            result = x.transpose(kwargs.get("dim0", 0), kwargs.get("dim1", 1))
        elif op == ATenOp.PERMUTE:
            result = x.permute(*kwargs.get("dims", tuple(range(x.dim()))))
        elif op == ATenOp.SQUEEZE:
            result = x.squeeze(kwargs.get("dim", None))
        elif op == ATenOp.UNSQUEEZE:
            result = x.unsqueeze(kwargs.get("dim", 0))
        
        # Activations
        elif op == ATenOp.RELU:
            result = F.relu(x)
        elif op == ATenOp.TANH:
            result = torch.tanh(x)
        elif op == ATenOp.SIGMOID:
            result = torch.sigmoid(x)
        elif op == ATenOp.GELU:
            result = F.gelu(x)
        elif op == ATenOp.SOFTMAX:
            result = F.softmax(x, dim=kwargs.get("dim", -1))
        elif op == ATenOp.LEAKY_RELU:
            result = F.leaky_relu(x, negative_slope=kwargs.get("negative_slope", 0.01))
        elif op == ATenOp.ELU:
            result = F.elu(x, alpha=kwargs.get("alpha", 1.0))
        elif op == ATenOp.SILU:
            result = F.silu(x)
        
        # Special
        elif op == ATenOp.CLONE:
            result = x.clone()
        elif op == ATenOp.CONTIGUOUS:
            result = x.contiguous()
        elif op == ATenOp.DETACH:
            result = x.detach()
        
        else:
            raise ValueError(f"Unknown operation: {op}")
        
        return result
    
    def _apply_activation(self, x: Tensor) -> Tensor:
        """Apply activation function if specified."""
        if self.activation is None:
            return x
        
        act = self.activation
        if act == ATenOp.RELU:
            return F.relu(x)
        elif act == ATenOp.TANH:
            return torch.tanh(x)
        elif act == ATenOp.SIGMOID:
            return torch.sigmoid(x)
        elif act == ATenOp.GELU:
            return F.gelu(x)
        elif act == ATenOp.SOFTMAX:
            return F.softmax(x, dim=-1)
        elif act == ATenOp.LEAKY_RELU:
            return F.leaky_relu(x)
        elif act == ATenOp.ELU:
            return F.elu(x)
        elif act == ATenOp.SILU:
            return F.silu(x)
        else:
            return x
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Compute one step of the ATen operation."""
        # Apply main operation
        result = self._apply_operation(x)
        
        # Add bias if present
        if hasattr(self, 'bias') and self.bias is not None:
            result = result + self.bias
        
        # Apply activation
        result = self._apply_activation(result)
        
        return {"out": result}
    
    def _run(
        self, 
        state: TorchState, 
        x: TorchTimeseries
    ) -> Tuple[TorchState, TorchTimeseries]:
        """Vectorized run for efficiency."""
        # For operations that can be vectorized
        if self.operation in [ATenOp.MATMUL, ATenOp.RELU, ATenOp.TANH, 
                              ATenOp.SIGMOID, ATenOp.GELU]:
            # Batch process
            if self.operation == ATenOp.MATMUL:
                result = torch.matmul(x, self.weight.T)
            else:
                result = self._apply_operation(x)
            
            if hasattr(self, 'bias') and self.bias is not None:
                result = result + self.bias
            
            result = self._apply_activation(result)
            
            return {"out": result[-1]}, result
        else:
            # Fall back to step-by-step
            return super()._run(state, x)


class ATenChain(TorchNode):
    """Chain multiple ATen operations together.
    
    Allows composing multiple ATen operations into a single node
    for efficient sequential processing.
    
    Parameters
    ----------
    operations : list of ATenOp or str
        Sequence of operations to apply
    weights : list of Tensor, optional
        Weights for each operation (if needed)
    activations : list of ATenOp or str, optional
        Activations after each operation
        
    Examples
    --------
    >>> chain = ATenChain(
    ...     operations=["matmul", "relu", "matmul", "tanh"],
    ...     weights=[W1, None, W2, None]
    ... )
    """
    
    def __init__(
        self,
        operations: List[Union[ATenOp, str]],
        weights: Optional[List[Optional[Tensor]]] = None,
        activations: Optional[List[Optional[Union[ATenOp, str]]]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        name: Optional[str] = None,
    ):
        super().__init__(device=device, dtype=dtype, name=name)
        
        self.operations = [
            ATenOp(op.lower()) if isinstance(op, str) else op
            for op in operations
        ]
        self.weights = weights or [None] * len(operations)
        self.activations = activations or [None] * len(operations)
        
        # Create ATenNode for each operation
        self.nodes: List[ATenNode] = []
    
    def initialize(
        self,
        x: Union[TorchTimestep, TorchTimeseries],
        y: Optional[Union[TorchTimestep, TorchTimeseries]] = None,
    ) -> None:
        """Initialize all nodes in the chain."""
        self._set_input_dim(x)
        
        current_dim = self.input_dim
        for i, (op, weight, act) in enumerate(zip(self.operations, self.weights, self.activations)):
            node = ATenNode(
                operation=op,
                weight=weight,
                activation=act,
                device=self.device,
                dtype=self.dtype,
            )
            # Create dummy input for initialization
            dummy = torch.zeros(current_dim, device=self.device, dtype=self.dtype)
            node.initialize(dummy)
            self.nodes.append(node)
            current_dim = node.output_dim
        
        self.output_dim = current_dim
        self.state = {"out": torch.zeros(self.output_dim, device=self.device, dtype=self.dtype)}
        self.initialized = True
    
    def _step(self, state: TorchState, x: TorchTimestep) -> TorchState:
        """Apply all operations in sequence."""
        result = x
        for node in self.nodes:
            result = node.step(result)
        return {"out": result}
