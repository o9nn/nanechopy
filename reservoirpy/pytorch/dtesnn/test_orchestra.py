"""
====================================
Test Autonomous Orchestra (:mod:`reservoirpy.pytorch.dtesnn.test_orchestra`)
====================================

Tests for the Deep Tree Echo Self Autonomous Orchestra.
"""

import pytest
import numpy as np
from datetime import datetime

from orchestra import (
    AutonomousOrchestra,
    OrchestraConfig,
    PersonaConfig,
    CognitiveStyle,
    AffectiveBaseline,
    MemoryHook,
    RelevanceRealizationConfig,
)


class TestPersonaConfig:
    """Test persona configuration."""
    
    def test_contemplative_scholar_creation(self):
        """Test creating contemplative scholar persona."""
        persona = PersonaConfig.contemplative_scholar()
        
        assert persona.style == CognitiveStyle.CONTEMPLATIVE_SCHOLAR
        assert persona.spectral_radius == 0.95
        assert persona.input_scaling == 0.3
        assert persona.leak_rate == 0.2
        assert persona.affective_baseline == AffectiveBaseline.WONDER
        assert "deep geodesics" in persona.wisdom_approach.lower()
    
    def test_dynamic_explorer_creation(self):
        """Test creating dynamic explorer persona."""
        persona = PersonaConfig.dynamic_explorer()
        
        assert persona.style == CognitiveStyle.DYNAMIC_EXPLORER
        assert persona.spectral_radius == 0.7
        assert persona.input_scaling == 0.8
        assert persona.leak_rate == 0.8
        assert persona.affective_baseline == AffectiveBaseline.EXCITEMENT
    
    def test_cautious_analyst_creation(self):
        """Test creating cautious analyst persona."""
        persona = PersonaConfig.cautious_analyst()
        
        assert persona.style == CognitiveStyle.CAUTIOUS_ANALYST
        assert persona.spectral_radius == 0.99
        assert persona.input_scaling == 0.2
        assert persona.leak_rate == 0.3
        assert persona.affective_baseline == AffectiveBaseline.INTEREST
    
    def test_creative_visionary_creation(self):
        """Test creating creative visionary persona."""
        persona = PersonaConfig.creative_visionary()
        
        assert persona.style == CognitiveStyle.CREATIVE_VISIONARY
        assert persona.spectral_radius == 0.85
        assert persona.input_scaling == 0.7
        assert persona.leak_rate == 0.6
        assert persona.affective_baseline == AffectiveBaseline.JOY
    
    def test_custom_persona(self):
        """Test creating custom persona."""
        persona = PersonaConfig(
            style=CognitiveStyle.CUSTOM,
            spectral_radius=0.88,
            input_scaling=0.5,
            leak_rate=0.4,
            affective_baseline=AffectiveBaseline.CURIOSITY,
            wisdom_approach="Custom approach",
            description="Custom persona",
        )
        
        assert persona.style == CognitiveStyle.CUSTOM
        assert persona.spectral_radius == 0.88
        assert persona.input_scaling == 0.5


class TestMemoryHook:
    """Test memory hook functionality."""
    
    def test_memory_creation(self):
        """Test creating a memory with hooks."""
        memory = MemoryHook(
            content="Test content",
            emotional_tone="wonder",
            pattern_recognition="recursive pattern",
            echo_signature="sig_123",
        )
        
        assert memory.content == "Test content"
        assert memory.emotional_tone == "wonder"
        assert memory.pattern_recognition == "recursive pattern"
        assert memory.echo_signature == "sig_123"
        assert memory.timestamp  # Should be auto-generated
    
    def test_memory_to_dict(self):
        """Test memory conversion to dictionary."""
        memory = MemoryHook(
            content="Test",
            emotional_tone="joy",
        )
        
        data = memory.to_dict()
        
        assert isinstance(data, dict)
        assert data['content'] == "Test"
        assert data['emotional_tone'] == "joy"
        assert 'timestamp' in data


class TestRelevanceRealizationConfig:
    """Test relevance realization configuration."""
    
    def test_default_config(self):
        """Test default relevance config."""
        config = RelevanceRealizationConfig()
        
        assert config.attention_temperature == 1.0
        assert config.salience_threshold == 0.1
        assert config.filter_strength == 0.5
        assert config.frame_adaptation_rate == 0.3
        assert config.feedforward_depth == 3
        assert config.feedback_decay == 0.9
    
    def test_custom_config(self):
        """Test custom relevance config."""
        config = RelevanceRealizationConfig(
            attention_temperature=0.5,
            salience_threshold=0.2,
            feedback_decay=0.8,
        )
        
        assert config.attention_temperature == 0.5
        assert config.salience_threshold == 0.2
        assert config.feedback_decay == 0.8


class TestOrchestraConfig:
    """Test orchestra configuration."""
    
    def test_default_config(self):
        """Test default orchestra config with auto-created personas."""
        config = OrchestraConfig()
        
        assert len(config.personas) == 3  # Default: 3 personas
        assert config.memory_capacity == 1000
        assert config.triangulation_threshold == 0.3
        assert config.enable_autonomous_coordination is True
        assert config.base_order == 6
    
    def test_custom_personas(self):
        """Test orchestra config with custom personas."""
        personas = [
            PersonaConfig.contemplative_scholar(),
            PersonaConfig.dynamic_explorer(),
        ]
        config = OrchestraConfig(
            personas=personas,
            memory_capacity=500,
        )
        
        assert len(config.personas) == 2
        assert config.memory_capacity == 500


class TestAutonomousOrchestra:
    """Test autonomous orchestra functionality."""
    
    def test_orchestra_creation(self):
        """Test creating an orchestra."""
        orchestra = AutonomousOrchestra()
        
        assert orchestra is not None
        assert len(orchestra.personas) >= 3  # At least 3 default personas
        assert len(orchestra.memories) == 0  # No memories initially
    
    def test_orchestra_with_custom_config(self):
        """Test orchestra with custom configuration."""
        config = OrchestraConfig(
            personas=[PersonaConfig.contemplative_scholar()],
            memory_capacity=100,
        )
        orchestra = AutonomousOrchestra(config=config)
        
        assert len(orchestra.personas) == 1
        assert orchestra.config.memory_capacity == 100
    
    def test_memory_cultivation(self):
        """Test cultivating memories."""
        orchestra = AutonomousOrchestra()
        
        memory = orchestra.cultivate_memory(
            content="Test memory",
            emotional_tone="wonder",
            pattern_recognition="recursive self-reference",
        )
        
        assert len(orchestra.memories) == 1
        assert memory.content == "Test memory"
        assert memory.emotional_tone == "wonder"
    
    def test_memory_retrieval(self):
        """Test retrieving memories."""
        orchestra = AutonomousOrchestra()
        
        # Add multiple memories
        orchestra.cultivate_memory(content="Memory 1", emotional_tone="joy")
        orchestra.cultivate_memory(content="Memory 2", emotional_tone="wonder")
        orchestra.cultivate_memory(content="Memory 3", emotional_tone="joy")
        
        # Retrieve all
        all_memories = orchestra.retrieve_memories(limit=10)
        assert len(all_memories) == 3
        
        # Retrieve by emotion
        joy_memories = orchestra.retrieve_memories(emotional_tone="joy", limit=10)
        assert len(joy_memories) == 2
        
        # Retrieve by query
        query_memories = orchestra.retrieve_memories(query="Memory 1", limit=10)
        assert len(query_memories) == 1
    
    def test_memory_capacity_pruning(self):
        """Test memory pruning when capacity exceeded."""
        config = OrchestraConfig(memory_capacity=5)
        orchestra = AutonomousOrchestra(config=config)
        
        # Add more memories than capacity
        for i in range(10):
            orchestra.cultivate_memory(content=f"Memory {i}")
        
        # Should only keep most recent 5
        assert len(orchestra.memories) == 5
        assert orchestra.memories[-1].content == "Memory 9"
    
    def test_process_single_persona(self):
        """Test processing with a single persona."""
        orchestra = AutonomousOrchestra()
        input_data = np.random.randn(10)
        
        persona_id = list(orchestra.personas.keys())[0]
        response = orchestra.process(input_data, persona_id=persona_id)
        
        assert response is not None
        assert isinstance(response, np.ndarray)
    
    def test_process_all_personas(self):
        """Test processing with all personas."""
        orchestra = AutonomousOrchestra()
        input_data = np.random.randn(10)
        
        responses = orchestra.process(input_data)
        
        assert isinstance(responses, dict)
        assert len(responses) == len(orchestra.personas)
    
    def test_realize_consensus_weighted_attention(self):
        """Test consensus with weighted attention."""
        orchestra = AutonomousOrchestra()
        
        # Create mock responses
        responses = {
            'persona_1': np.array([1.0, 2.0, 3.0]),
            'persona_2': np.array([2.0, 3.0, 4.0]),
            'persona_3': np.array([3.0, 4.0, 5.0]),
        }
        
        consensus = orchestra.realize_consensus(responses, method='weighted_attention')
        
        assert consensus is not None
        assert len(consensus) == 3
        assert np.all(np.isfinite(consensus))
    
    def test_realize_consensus_mean(self):
        """Test consensus with simple mean."""
        orchestra = AutonomousOrchestra()
        
        responses = {
            'persona_1': np.array([1.0, 2.0, 3.0]),
            'persona_2': np.array([3.0, 4.0, 5.0]),
        }
        
        consensus = orchestra.realize_consensus(responses, method='mean')
        
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(consensus, expected)
    
    def test_detect_misalignment(self):
        """Test misalignment detection."""
        orchestra = AutonomousOrchestra()
        
        responses = {
            'persona_1': np.array([1.0, 1.0, 1.0]),
            'persona_2': np.array([1.1, 0.9, 1.0]),
            'persona_3': np.array([10.0, 10.0, 10.0]),  # Outlier
        }
        
        misalignments = orchestra.detect_misalignment(responses)
        
        assert len(misalignments) == 3
        # Persona 3 should have highest misalignment
        assert misalignments['persona_3'] > misalignments['persona_1']
        assert misalignments['persona_3'] > misalignments['persona_2']
    
    def test_update_relevance_states(self):
        """Test updating relevance states."""
        orchestra = AutonomousOrchestra()
        
        input_data = np.random.randn(10)
        responses = {
            pid: np.random.randn(10) for pid in orchestra.personas.keys()
        }
        
        # Get initial states
        initial_states = {
            pid: state.copy() for pid, state in orchestra.relevance_states.items()
        }
        
        # Update
        orchestra.update_relevance_states(input_data, responses)
        
        # States should have changed
        for pid in orchestra.personas.keys():
            assert not np.array_equal(
                orchestra.relevance_states[pid],
                initial_states[pid]
            )
    
    def test_get_persona_info(self):
        """Test getting persona information."""
        orchestra = AutonomousOrchestra()
        
        persona_id = list(orchestra.personas.keys())[0]
        info = orchestra.get_persona_info(persona_id)
        
        assert 'id' in info
        assert 'style' in info
        assert 'spectral_radius' in info
        assert 'leak_rate' in info
        assert 'affective_baseline' in info
        assert 'wisdom_approach' in info
        assert 'current_relevance' in info
    
    def test_get_orchestra_state(self):
        """Test getting complete orchestra state."""
        orchestra = AutonomousOrchestra()
        
        # Add some memories
        orchestra.cultivate_memory(content="Test memory 1")
        orchestra.cultivate_memory(content="Test memory 2")
        
        state = orchestra.get_orchestra_state()
        
        assert 'personas' in state
        assert 'memory_count' in state
        assert 'memory_capacity' in state
        assert 'recent_memories' in state
        assert 'autonomous_coordination_enabled' in state
        
        assert state['memory_count'] == 2
        assert len(state['personas']) == len(orchestra.personas)
    
    def test_explain_self(self):
        """Test self-explanation generation."""
        orchestra = AutonomousOrchestra()
        
        # Add a memory
        orchestra.cultivate_memory(content="Test")
        
        explanation = orchestra.explain_self()
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Deep Tree Echo Self" in explanation
        assert "emergent process" in explanation
        assert str(len(orchestra.personas)) in explanation
        assert str(len(orchestra.memories)) in explanation
    
    def test_persona_specific_parameters(self):
        """Test that each persona has distinct parameters."""
        config = OrchestraConfig(
            personas=[
                PersonaConfig.contemplative_scholar(),
                PersonaConfig.dynamic_explorer(),
            ]
        )
        orchestra = AutonomousOrchestra(config=config)
        
        personas_list = list(orchestra.personas.values())
        p1 = personas_list[0]
        p2 = personas_list[1]
        
        # They should have different parameters
        assert p1.spectral_radius != p2.spectral_radius
        assert p1.leak_rate != p2.leak_rate
        assert p1.input_scaling != p2.input_scaling


class TestIntegration:
    """Integration tests for full orchestra workflow."""
    
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline."""
        orchestra = AutonomousOrchestra()
        
        # Generate input
        input_data = np.random.randn(10)
        
        # Process with all personas
        responses = orchestra.process(input_data)
        
        # Realize consensus
        consensus = orchestra.realize_consensus(responses)
        
        # Detect misalignment
        misalignments = orchestra.detect_misalignment(responses)
        
        # Update relevance states
        orchestra.update_relevance_states(input_data, responses)
        
        # Cultivate memory
        memory = orchestra.cultivate_memory(
            content=input_data,
            emotional_tone="curiosity",
            pattern_recognition="processing pipeline completed",
        )
        
        # Verify all steps completed
        assert consensus is not None
        assert len(misalignments) == len(orchestra.personas)
        assert len(orchestra.memories) == 1
    
    def test_temporal_memory_sequence(self):
        """Test temporal sequence of memory cultivation."""
        orchestra = AutonomousOrchestra()
        
        # Create sequence of memories
        memories = []
        for i in range(5):
            mem = orchestra.cultivate_memory(
                content=f"Step {i}",
                emotional_tone="contemplation",
                echo_signature=f"echo_{i}",
            )
            memories.append(mem)
        
        # Verify temporal ordering
        timestamps = [m.timestamp for m in memories]
        # Timestamps should be in order (or very close due to execution speed)
        assert len(timestamps) == 5
        
        # Retrieve recent memories
        recent = orchestra.retrieve_memories(limit=3)
        assert len(recent) == 3
        assert recent[-1].content == "Step 4"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
