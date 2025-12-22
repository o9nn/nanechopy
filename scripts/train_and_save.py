#!/usr/bin/env python3
"""
Train and save DTESNN chatbot model.

This script is designed to be run by the GitHub Action workflow.
It reads configuration from environment variables and saves the trained
model to a timestamped file in the models/ directory.

The script loads training data from data/training_dataset.jsonl if available.

Environment Variables:
    ORDER: A000081 order for the model (default: 6)
    EMBEDDING_DIM: Embedding dimension (default: 32)
    UNITS_PER_COMPONENT: Units per component (default: 16)
    CUSTOM_TRAINING_PAIRS: JSON string of additional training pairs (optional)
    TRAINING_DATA_PATH: Path to training data JSONL file (default: data/training_dataset.jsonl)
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the package to path
script_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(script_dir)
sys.path.insert(0, repo_dir)
sys.path.insert(0, os.path.join(repo_dir, 'reservoirpy', 'pytorch', 'dtesnn'))
sys.path.insert(0, os.path.join(repo_dir, 'reservoirpy', 'pytorch', 'autognosis'))

from chatbot import DTESNNChatbot, ChatbotConfig
from persistence import save_chatbot, get_model_info


def load_training_data(filepath: str) -> list:
    """
    Load training pairs from JSONL file.
    
    Expected format:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    Returns list of (input, response) tuples.
    """
    training_pairs = []
    
    if not os.path.exists(filepath):
        print(f"Training data file not found: {filepath}")
        return training_pairs
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                
                if len(messages) >= 2:
                    user_msg = messages[0].get('content', '')
                    asst_msg = messages[1].get('content', '')
                    
                    if user_msg and asst_msg:
                        # Clean up the messages - extract key content
                        # Remove markdown formatting and code blocks
                        user_text = clean_message(user_msg)
                        asst_text = clean_message(asst_msg)
                        
                        if user_text and asst_text:
                            training_pairs.append((user_text, asst_text))
                            
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    return training_pairs


def clean_message(text: str) -> str:
    """Clean message text for training."""
    import re
    
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'#{1,6}\s*', '', text)  # Headers
    
    # Remove special markers
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Template markers
    text = re.sub(r'\[\[[^\]]+\]\]', '', text)  # Wiki links
    
    # Clean up whitespace
    text = ' '.join(text.split())
    
    # Truncate to reasonable length
    words = text.split()[:100]  # Max 100 words
    
    return ' '.join(words)


def main():
    # Read configuration from environment
    order = int(os.environ.get('ORDER', '6'))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', '32'))
    units_per_component = int(os.environ.get('UNITS_PER_COMPONENT', '16'))
    custom_pairs_json = os.environ.get('CUSTOM_TRAINING_PAIRS', '')
    training_data_path = os.environ.get('TRAINING_DATA_PATH', 'data/training_dataset.jsonl')
    
    print("=" * 60)
    print("DTESNN Chatbot Training")
    print("=" * 60)
    print(f"Order: {order}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Units per Component: {units_per_component}")
    print(f"Training Data Path: {training_data_path}")
    
    # Load training data from file
    training_pairs = []
    if os.path.exists(training_data_path):
        print(f"\nLoading training data from: {training_data_path}")
        training_pairs = load_training_data(training_data_path)
        print(f"Loaded {len(training_pairs)} training pairs from file")
    else:
        print(f"\nTraining data file not found, using default pairs only")
    
    # Parse custom training pairs if provided
    if custom_pairs_json and custom_pairs_json.strip():
        try:
            custom_pairs = json.loads(custom_pairs_json)
            if isinstance(custom_pairs, list):
                # Convert to list of tuples if it's a list of lists
                if len(custom_pairs) > 0 and isinstance(custom_pairs[0], list):
                    custom_pairs = [tuple(pair) for pair in custom_pairs]
                training_pairs.extend(custom_pairs)
                print(f"Added {len(custom_pairs)} custom training pairs")
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse custom training pairs: {e}")
    
    print(f"Total training pairs: {len(training_pairs)}")
    
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
    
    if training_pairs:
        chatbot.train(additional_pairs=training_pairs)
    else:
        chatbot.train()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    # Get model info
    print("\nModel Info:")
    print(chatbot.get_model_info())
    
    # Test the model
    print("\nTest Responses:")
    test_inputs = [
        "hello",
        "Deep Tree Echo train your reservoirs",
        "what is intelligence",
        "how do you work",
        "optimize your system",
    ]
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
        "training_pairs_count": len(training_pairs),
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
