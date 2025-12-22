"""
Tests for PyTorch integration and autognosis engine.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")


class TestA000081Parameters:
    """Test A000081 parameter derivation."""
    
    def test_get_a000081_value(self):
        """Test getting A000081 sequence values."""
        from reservoirpy.pytorch.autognosis.a000081 import get_a000081_value
        
        # Known values from OEIS A000081
        assert get_a000081_value(1) == 1
        assert get_a000081_value(2) == 1
        assert get_a000081_value(3) == 2
        assert get_a000081_value(4) == 4
        assert get_a000081_value(5) == 9
        assert get_a000081_value(6) == 20
        assert get_a000081_value(7) == 48
        assert get_a000081_value(8) == 115
    
    def test_cumulative_a000081(self):
        """Test cumulative sum of A000081."""
        from reservoirpy.pytorch.autognosis.a000081 import cumulative_a000081
        
        # 1 + 1 + 2 + 4 + 9 = 17
        assert cumulative_a000081(5) == 17
        
        # 1 + 1 + 2 + 4 + 9 + 20 = 37
        assert cumulative_a000081(6) == 37
    
    def test_derive_parameters(self):
        """Test parameter derivation."""
        from reservoirpy.pytorch.autognosis.a000081 import derive_parameters
        
        params = derive_parameters(base_order=5)
        
        assert params.base_order == 5
        assert params.reservoir_size == 17  # cumulative A000081 up to 5
        assert params.max_tree_order == 8  # 2 * 5 - 2
        assert params.num_membranes == 4  # A000081[4]
        assert params.growth_rate == 20 / 9  # A000081[6] / A000081[5]
        assert params.mutation_rate == 1 / 9  # 1 / A000081[5]
    
    def test_explain_parameters(self):
        """Test parameter explanation."""
        from reservoirpy.pytorch.autognosis.a000081 import derive_parameters
        
        params = derive_parameters(base_order=5)
        explanation = params.explain()
        
        assert "A000081" in explanation
        assert "reservoir_size" in explanation
        assert "17" in explanation


class TestBSeriesKernel:
    """Test B-series computational kernel."""
    
    def test_generate_trees(self):
        """Test rooted tree generation."""
        from reservoirpy.pytorch.autognosis.bseries import generate_trees
        
        # A000081[1] = 1 tree of order 1
        trees_1 = generate_trees(1)
        assert len(trees_1) == 1
        
        # A000081[2] = 1 tree of order 2
        trees_2 = generate_trees(2)
        assert len(trees_2) == 1
        
        # A000081[3] = 2 trees of order 3
        trees_3 = generate_trees(3)
        assert len(trees_3) == 2
        
        # A000081[4] = 4 trees of order 4
        trees_4 = generate_trees(4)
        assert len(trees_4) == 4
    
    def test_tree_properties(self):
        """Test rooted tree properties."""
        from reservoirpy.pytorch.autognosis.bseries import RootedTree
        
        # Single node
        t1 = RootedTree([])
        assert t1.order == 1
        assert t1.symmetry == 1
        assert t1.density == 1.0
        
        # Two nodes
        t2 = RootedTree([RootedTree([])])
        assert t2.order == 2
        assert t2.symmetry == 1
        assert t2.density == 2.0
    
    def test_bseries_evaluate(self):
        """Test B-series evaluation."""
        from reservoirpy.pytorch.autognosis.bseries import BSeriesKernel
        
        kernel = BSeriesKernel(max_order=3)
        
        # Simple ODE: dy/dt = -y
        def f(y):
            return -y
        
        y0 = np.array([1.0])
        y1 = kernel.evaluate(f, y0, dt=0.1)
        
        # Should decrease
        assert y1[0] < y0[0]


class TestOntogeneticEngine:
    """Test ontogenetic evolution engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        from reservoirpy.pytorch.autognosis.ontogenetic import OntogeneticEngine
        
        engine = OntogeneticEngine(base_order=5, population_size=10, seed=42)
        engine.initialize()
        
        assert engine.state.generation == 0
        assert len(engine.state.tree_population) >= 10
    
    def test_evolution_step(self):
        """Test evolution step."""
        from reservoirpy.pytorch.autognosis.ontogenetic import OntogeneticEngine
        
        engine = OntogeneticEngine(base_order=5, population_size=10, seed=42)
        engine.initialize()
        
        fitness = engine.evolve_step()
        
        assert engine.state.generation == 1
        assert fitness > float('-inf')
        assert len(engine.state.fitness_history) == 1


class TestAutognosisEngine:
    """Test main autognosis engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        from reservoirpy.pytorch.autognosis.engine import AutognosisEngine
        
        engine = AutognosisEngine(base_order=5, seed=42)
        engine.initialize(input_dim=10, output_dim=5)
        
        assert engine.is_initialized
        assert engine.input_dim == 10
        assert engine.output_dim == 5
    
    def test_step(self):
        """Test single step."""
        from reservoirpy.pytorch.autognosis.engine import AutognosisEngine
        
        engine = AutognosisEngine(base_order=5, seed=42)
        engine.initialize(input_dim=10)
        
        x = np.random.randn(10)
        state = engine.step(x)
        
        assert state.shape == (engine.params.reservoir_size,)
    
    def test_process(self):
        """Test sequence processing."""
        from reservoirpy.pytorch.autognosis.engine import AutognosisEngine
        
        engine = AutognosisEngine(base_order=5, seed=42)
        engine.initialize(input_dim=10)
        
        X = np.random.randn(100, 10)
        states = engine.process(X)
        
        assert states.shape == (100, engine.params.reservoir_size)
    
    def test_fit_predict(self):
        """Test training and prediction."""
        from reservoirpy.pytorch.autognosis.engine import AutognosisEngine
        
        engine = AutognosisEngine(base_order=5, seed=42)
        
        # Generate simple data
        X = np.random.randn(200, 10)
        y = np.sin(np.linspace(0, 4 * np.pi, 200)).reshape(-1, 1)
        
        # Train
        engine.fit(X, y, warmup=10)
        
        # Predict
        engine.reset()
        predictions = engine.predict(X)
        
        assert predictions.shape == (200, 1)


class TestAutognosisNode:
    """Test autognosis node."""
    
    def test_initialization(self):
        """Test node initialization."""
        from reservoirpy.pytorch.autognosis.node import AutognosisNode
        
        node = AutognosisNode(base_order=5, seed=42)
        
        x = torch.randn(10)
        node.initialize(x)
        
        assert node.initialized
        assert node.input_dim == 10
    
    def test_step(self):
        """Test single step."""
        from reservoirpy.pytorch.autognosis.node import AutognosisNode
        
        node = AutognosisNode(base_order=5, seed=42)
        
        x = torch.randn(10)
        node.initialize(x)
        
        state = node._step(node.state, x)
        
        assert "out" in state
        assert "reservoir" in state
    
    def test_fit(self):
        """Test training."""
        from reservoirpy.pytorch.autognosis.node import AutognosisNode
        
        node = AutognosisNode(base_order=5, seed=42)
        
        X = torch.randn(100, 10)
        y = torch.randn(100, 3)
        
        node.fit(X, y, warmup=10)
        
        assert node.output_dim == 3


class TestTorchNodes:
    """Test PyTorch-based nodes."""
    
    def test_aten_matmul(self):
        """Test ATen matmul node."""
        from reservoirpy.pytorch.nodes.aten import ATenMatmulNode
        
        node = ATenMatmulNode(out_features=5)
        
        x = torch.randn(10)
        node.initialize(x)
        
        out = node.step(x)
        
        assert out.shape == (5,)
    
    def test_nn_linear(self):
        """Test NN linear node."""
        from reservoirpy.pytorch.nodes.nn_layers import NNLinearNode
        
        node = NNLinearNode(units=20, input_dim=10)
        
        x = torch.randn(10)
        node.initialize(x)
        
        out = node.step(x)
        
        assert out.shape == (20,)
    
    def test_lstm_node(self):
        """Test LSTM node."""
        from reservoirpy.pytorch.nodes.rnn import LSTMNode
        
        node = LSTMNode(hidden_size=32, input_dim=10)
        
        x = torch.randn(10)
        node.initialize(x)
        
        state = node._step(node.state, x)
        
        assert state["out"].shape == (32,)
        assert state["cell"].shape == (32,)
    
    def test_esn_torch_node(self):
        """Test ESN PyTorch node."""
        from reservoirpy.pytorch.nodes.rnn import ESNTorchNode
        
        node = ESNTorchNode(units=100, lr=0.3, sr=0.9, seed=42)
        
        x = torch.randn(10)
        node.initialize(x)
        
        state = node._step(node.state, x)
        
        assert state["reservoir"].shape == (100,)
    
    def test_esn_fit(self):
        """Test ESN training."""
        from reservoirpy.pytorch.nodes.rnn import ESNTorchNode
        
        node = ESNTorchNode(units=100, lr=0.3, sr=0.9, seed=42)
        
        X = torch.randn(200, 10)
        y = torch.randn(200, 3)
        
        node.fit(X, y, warmup=10)
        
        assert node.W_out is not None
        assert node.W_out.shape == (3, 100)


class TestModels:
    """Test high-level models."""
    
    def test_torch_reservoir_model(self):
        """Test TorchReservoirModel."""
        from reservoirpy.pytorch.model import TorchReservoirModel
        
        model = TorchReservoirModel(
            reservoir_size=100,
            lr=0.3,
            sr=0.9,
            seed=42
        )
        
        X = torch.randn(200, 10)
        y = torch.randn(200, 3)
        
        model.fit(X, y, warmup=10)
        
        model.reset()
        _, outputs = model._run(model.state, X)
        
        assert outputs.shape == (200, 3)
    
    def test_autognosis_model(self):
        """Test AutognosisModel."""
        from reservoirpy.pytorch.model import AutognosisModel
        
        model = AutognosisModel(
            base_order=5,
            enable_evolution=False,  # Disable for faster test
            seed=42
        )
        
        X = torch.randn(200, 10)
        y = torch.randn(200, 3)
        
        model.fit(X, y, warmup=10)
        
        model.reset()
        _, outputs = model._run(model.state, X)
        
        assert outputs.shape == (200, 3)
        
        # Check status
        status = model.get_status()
        assert "parameters" in status
        assert "base_order" in status


class TestOps:
    """Test node operations."""
    
    def test_link(self):
        """Test link operation."""
        from reservoirpy.pytorch.ops import link
        from reservoirpy.pytorch.nodes.nn_layers import NNLinearNode
        
        node1 = NNLinearNode(units=20, input_dim=10)
        node2 = NNLinearNode(units=5, input_dim=20)
        
        model = link(node1, node2)
        
        x = torch.randn(10)
        model.initialize(x)
        
        state = model._step(model.state, x)
        
        assert state["out"].shape == (5,)
    
    def test_merge(self):
        """Test merge operation."""
        from reservoirpy.pytorch.ops import merge
        from reservoirpy.pytorch.nodes.nn_layers import NNLinearNode
        
        node1 = NNLinearNode(units=10, input_dim=5)
        node2 = NNLinearNode(units=15, input_dim=5)
        
        merged = merge(node1, node2)
        
        x = torch.randn(5)
        merged.initialize(x)
        
        state = merged._step(merged.state, x)
        
        # Output should be concatenation: 10 + 15 = 25
        assert state["out"].shape == (25,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
