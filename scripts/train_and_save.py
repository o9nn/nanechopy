#!/usr/bin/env python3
"""
Train and save DTESNN chatbot model in multiple formats.

This script is designed to be run by the GitHub Action workflow.
It reads configuration from environment variables and saves the trained
model to multiple formats in the models/ directory.

Supported export formats:
- JSON (.json): Complete Python dictionary as JSON
- Pickle (.pkl.gz): Compressed Python object serialization
- NumPy (.npz): Compressed numpy arrays
- Config JSON (.config.json): Configuration and metadata only
- ONNX (.onnx): Open Neural Network Exchange format
- GGUF (.gguf): GPT-Generated Unified Format
- Guile Scheme (.scm): S-expression format

Environment Variables:
    ORDER: A000081 order for the model (default: 6)
    EMBEDDING_DIM: Embedding dimension (default: 32)
    UNITS_PER_COMPONENT: Units per component (default: 16)
    CUSTOM_TRAINING_PAIRS: JSON string of additional training pairs (optional)
    TRAINING_DATA_PATH: Path to training data JSONL file (default: data/training_dataset.jsonl)
    EXPORT_FORMATS: Comma-separated list of formats to export (default: all)
"""

import os
import sys
import json
import re
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
from export import export_all_formats


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
    export_formats = os.environ.get('EXPORT_FORMATS', 'all')
    
    print("=" * 60)
    print("DTESNN Chatbot Training & Multi-Format Export")
    print("=" * 60)
    print(f"Order: {order}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"Units per Component: {units_per_component}")
    print(f"Training Data Path: {training_data_path}")
    print(f"Export Formats: {export_formats}")
    
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
    
    # Generate timestamped base filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_filename = f"models/dtesnn_order{order}_{timestamp}"
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Export to all formats
    print("\n" + "=" * 60)
    print("Exporting to Multiple Formats")
    print("=" * 60)
    
    export_paths = export_all_formats(chatbot, base_filename)
    
    # Calculate total size
    total_size = 0
    for fmt, path in export_paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            total_size += size
            print(f"  {fmt}: {path} ({size / 1024:.2f} KB)")
    
    print(f"\nTotal export size: {total_size / 1024:.2f} KB")
    
    # Create comprehensive metadata file
    metadata = {
        "model_name": f"dtesnn_order{order}",
        "timestamp": timestamp,
        "created_at": datetime.utcnow().isoformat() + "Z",
        
        "configuration": {
            "order": order,
            "embedding_dim": embedding_dim,
            "units_per_component": units_per_component,
            "temperature": config.temperature,
            "top_k": config.top_k,
        },
        
        "training": {
            "training_time_seconds": training_time,
            "training_pairs_count": len(training_pairs),
            "training_data_path": training_data_path,
        },
        
        "model_stats": {
            "vocab_size": len(chatbot.vocab.words),
            "total_trees": chatbot.model.synchronizer.params.total_trees if chatbot.model.synchronizer else 0,
            "tree_counts": chatbot.model.synchronizer.params.tree_counts if chatbot.model.synchronizer else [],
        },
        
        "exports": {
            fmt: {
                "path": path,
                "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
            }
            for fmt, path in export_paths.items()
        },
        
        "total_size_bytes": total_size,
    }
    
    metadata_file = f"{base_filename}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {metadata_file}")
    
    print("\n" + "=" * 60)
    print("Training & Export Complete!")
    print("=" * 60)
    
    # Return the primary pkl.gz path for backward compatibility
    return export_paths.get('pkl', f"{base_filename}.pkl.gz")


if __name__ == "__main__":
    main()
