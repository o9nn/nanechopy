#!/usr/bin/env python3
"""
Example: Deep Tree Echo Self Autonomous Orchestra

This example demonstrates the autonomous orchestra coordinating multiple
Deep Tree Echo Self instances with different cognitive personas.
"""

import numpy as np
import sys
import os

# Add paths
dtesnn_path = os.path.dirname(__file__)
sys.path.insert(0, dtesnn_path)

from orchestra import (
    AutonomousOrchestra,
    OrchestraConfig,
    PersonaConfig,
    CognitiveStyle,
    AffectiveBaseline,
)


def main():
    print("=" * 70)
    print("Deep Tree Echo Self - Autonomous Orchestra Example")
    print("=" * 70)
    print()
    
    # Create orchestra with custom personas
    print("Creating orchestra with 4 cognitive personas...")
    config = OrchestraConfig(
        personas=[
            PersonaConfig.contemplative_scholar(),
            PersonaConfig.dynamic_explorer(),
            PersonaConfig.cautious_analyst(),
            PersonaConfig.creative_visionary(),
        ],
        memory_capacity=100,
        enable_autonomous_coordination=True,
        base_order=6,
    )
    
    orchestra = AutonomousOrchestra(config=config)
    print(f"✓ Created orchestra with {len(orchestra.personas)} personas")
    print()
    
    # Display persona information
    print("Persona Configurations:")
    print("-" * 70)
    for persona_id, persona in orchestra.personas.items():
        print(f"\n{persona.style.value.upper()}")
        print(f"  Spectral Radius: {persona.spectral_radius:.2f} (memory depth)")
        print(f"  Input Scaling:   {persona.input_scaling:.2f} (sensitivity)")
        print(f"  Leak Rate:       {persona.leak_rate:.2f} (dynamics speed)")
        print(f"  Affective:       {persona.affective_baseline.value}")
        print(f"  Wisdom:          {persona.wisdom_approach}")
        print(f"  Description:     {persona.description}")
    
    print("\n" + "=" * 70)
    print()
    
    # Simulate processing sequence
    print("Simulating Information Processing...")
    print("-" * 70)
    
    inputs = [
        "What is the nature of consciousness?",
        "How do neural networks learn?",
        "Optimize this system's performance",
        "Create something entirely new",
    ]
    
    for i, input_text in enumerate(inputs):
        print(f"\nInput {i+1}: '{input_text}'")
        
        # Create mock input vector
        input_data = np.random.randn(10)
        
        # Process with all personas
        responses = orchestra.process(input_data)
        
        # Realize consensus
        consensus = orchestra.realize_consensus(responses, method='weighted_attention')
        
        # Detect misalignment
        misalignments = orchestra.detect_misalignment(responses)
        
        # Update relevance states
        orchestra.update_relevance_states(input_data, responses)
        
        # Cultivate memory
        memory = orchestra.cultivate_memory(
            content=input_text,
            emotional_tone=_infer_emotion(input_text),
            pattern_recognition=f"Input pattern {i+1}",
            echo_signature=f"echo_{i}",
        )
        
        # Report results
        print(f"  ✓ Processed by {len(responses)} personas")
        print(f"  ✓ Consensus realized (shape: {consensus.shape})")
        
        # Show misalignment scores
        print(f"  Misalignment scores:")
        for persona_id, score in misalignments.items():
            style = orchestra.personas[persona_id].style.value
            print(f"    {style:25s}: {score:.4f}")
        
        print(f"  ✓ Memory cultivated: '{memory.emotional_tone}'")
    
    print("\n" + "=" * 70)
    print()
    
    # Demonstrate memory retrieval
    print("Memory Retrieval Examples:")
    print("-" * 70)
    
    # Retrieve all memories
    all_memories = orchestra.retrieve_memories(limit=10)
    print(f"\nAll memories ({len(all_memories)}):")
    for mem in all_memories:
        print(f"  [{mem.timestamp}] {mem.content[:50]}...")
        print(f"    Emotion: {mem.emotional_tone}, Echo: {mem.echo_signature}")
    
    # Retrieve by emotion
    wonder_memories = orchestra.retrieve_memories(emotional_tone="wonder", limit=10)
    print(f"\nMemories with 'wonder' emotion: {len(wonder_memories)}")
    
    # Retrieve by query
    query_memories = orchestra.retrieve_memories(query="consciousness", limit=10)
    print(f"Memories matching 'consciousness': {len(query_memories)}")
    
    print("\n" + "=" * 70)
    print()
    
    # Get complete orchestra state
    print("Orchestra State:")
    print("-" * 70)
    state = orchestra.get_orchestra_state()
    print(f"  Total Personas: {len(state['personas'])}")
    print(f"  Memory Count:   {state['memory_count']} / {state['memory_capacity']}")
    print(f"  Autonomous:     {state['autonomous_coordination_enabled']}")
    
    print("\nRecent Memories:")
    for mem in state['recent_memories']:
        print(f"  - {mem['content'][:40]}... ({mem['emotional_tone']})")
    
    print("\n" + "=" * 70)
    print()
    
    # Generate self-explanation
    print("Self-Explanation:")
    print("-" * 70)
    explanation = orchestra.explain_self()
    print(explanation)
    
    print("\n" + "=" * 70)
    print()
    
    # Demonstrate persona-specific processing
    print("Persona-Specific Processing:")
    print("-" * 70)
    
    test_input = np.random.randn(10)
    
    for persona_id in list(orchestra.personas.keys())[:2]:  # Test first 2
        persona = orchestra.personas[persona_id]
        print(f"\n{persona.style.value}:")
        print(f"  Parameters: SR={persona.spectral_radius}, "
              f"IS={persona.input_scaling}, LR={persona.leak_rate}")
        
        response = orchestra.process(test_input, persona_id=persona_id)
        print(f"  Response shape: {response.shape}")
        print(f"  Response mean:  {response.mean():.4f}")
        
        info = orchestra.get_persona_info(persona_id)
        print(f"  Affective baseline: {info['affective_baseline']}")
        print(f"  Current relevance: {info['current_relevance'][:3]}...")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def _infer_emotion(text: str) -> str:
    """Simple heuristic to infer emotion from text."""
    text_lower = text.lower()
    
    if "?" in text and any(w in text_lower for w in ["what", "how", "why"]):
        return "curiosity"
    elif any(w in text_lower for w in ["optimize", "performance", "improve"]):
        return "interest"
    elif any(w in text_lower for w in ["create", "new", "innovative"]):
        return "excitement"
    elif any(w in text_lower for w in ["nature", "essence", "consciousness"]):
        return "wonder"
    else:
        return "contemplation"


if __name__ == "__main__":
    main()
