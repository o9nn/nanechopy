#!/usr/bin/env python3
"""
Train and save DTESNN chatbot model.

This script is designed to be run by the GitHub Action workflow.
It reads configuration from environment variables and saves the trained
model to a timestamped file in the models/ directory.

Environment Variables:
    ORDER: A000081 order for the model (default: 6)
    EMBEDDING_DIM: Embedding dimension (default: 32)
    UNITS_PER_COMPONENT: Units per component (default: 16)
    CUSTOM_TRAINING_PAIRS: JSON string of additional training pairs (optional)
"""

import os
import sys
import json
from datetime import datetime

# Add the package to path
script_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)
sys.path.insert(0, os.path.join(repo_dir, 'reservoirpy', 'pytorch', 'dtesnn'))
sys.path.insert(0, os.path.join(repo_dir, 'reservoirpy', 'pytorch', 'autognosis'))

from chatbot import DTESNNChatbot, ChatbotConfig
from persistence import save_chatbot, get_model_info


def main():
    # Read configuration from environment
    order = int(os.environ.get('ORDER', '6'))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', '32'))
    units_per_component = int(os.environ.get('UNITS_PER_COMPONENT', '16'))
    custom_pairs_json = os.environ.get('CUSTOM_TRAINING_PAIRS', '')
    
    print("=" * 60)
    print("DTESNN Chatbot Training")
    print("=" * 60)
    print(f"Order: {order}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Units per Component: {units_per_component}")
    
    # Parse custom training pairs if provided
    custom_pairs = None
    if custom_pairs_json and custom_pairs_json.strip():
        try:
            custom_pairs = json.loads(custom_pairs_json)
            print(f"Custom Training Pairs: {len(custom_pairs)} pairs")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse custom training pairs: {e}")
            custom_pairs = None
    
    # Create configuration
    config = ChatbotConfig(
        base_order=order,
        embedding_dim=embedding_dim,
        units_per_component=units_per_component,
        seed=42,
        temperature=0.8,
        top_k=50,
        max_response_length=50,
    )
    
    # Create chatbot
    print("\nCreating chatbot...")
    chatbot = DTESNNChatbot(config)
    
    # Train
    print("Training chatbot...")
    import time
    start_time = time.time()
    
    if custom_pairs:
        # Convert to list of tuples if it's a list of lists
        if isinstance(custom_pairs, list) and len(custom_pairs) > 0:
            if isinstance(custom_pairs[0], list):
                custom_pairs = [tuple(pair) for pair in custom_pairs]
        chatbot.train(additional_pairs=custom_pairs)
    else:
        chatbot.train()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # Get model info
    print("\nModel Info:")
    print(chatbot.get_model_info())
    
    # Test the model
    print("\nTest Responses:")
    test_inputs = ["hello", "what is intelligence", "how do you work"]
    for test_input in test_inputs:
        response = chatbot.respond(test_input)
        print(f"  '{test_input}': {response[:80]}...")
    
    # Generate timestamped filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"models/dtesnn_order{order}_{timestamp}.pkl.gz"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    print(f"\nSaving model to {filename}...")
    saved_path = save_chatbot(chatbot, filename, compress=True)
    
    # Verify the saved model
    file_size = os.path.getsize(saved_path)
    print(f"Model saved successfully!")
    print(f"  File: {saved_path}")
    print(f"  Size: {file_size / 1024:.2f} KB")
    
    # Get and display model info
    info = get_model_info(saved_path)
    print(f"  Version: {info.get('version')}")
    print(f"  Vocab Size: {info.get('vocab_size')}")
    print(f"  Is Trained: {info.get('is_trained')}")
    
    # Create a metadata file
    metadata = {
        "filename": os.path.basename(saved_path),
        "order": order,
        "embedding_dim": embedding_dim,
        "units_per_component": units_per_component,
        "training_time_seconds": training_time,
        "file_size_bytes": file_size,
        "vocab_size": info.get('vocab_size'),
        "timestamp": timestamp,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    
    metadata_file = saved_path.replace('.pkl.gz', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_file}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return saved_path


if __name__ == "__main__":
    main()
