"""
====================================
DTESNN Multi-Format Export (:mod:`reservoirpy.pytorch.dtesnn.export`)
====================================

Export DTESNN models to multiple formats for maximum portability:

- **JSON** (.json): Complete Python dictionary as JSON (human-readable)
- **Pickle** (.pkl, .pkl.gz): Full Python object serialization
- **NumPy** (.npz): Compressed numpy arrays (weights only)
- **Config JSON** (.config.json): Configuration and metadata only
- **ONNX** (.onnx): Open Neural Network Exchange format
- **GGUF** (.gguf): GPT-Generated Unified Format (llama.cpp compatible)
- **Guile Scheme** (.scm): S-expression format for Scheme/Lisp

Examples
--------
>>> from reservoirpy.pytorch.dtesnn.export import export_all_formats
>>> 
>>> # Export to all formats at once
>>> paths = export_all_formats(chatbot, "models/my_model")
>>> print(paths)
{'json': 'models/my_model.json', 'pkl': 'models/my_model.pkl.gz', ...}
"""

# License: MIT License
# Copyright: nanechopy contributors

import os
import json
import struct
import pickle
import gzip
from dataclasses import asdict
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    from .chatbot import DTESNNChatbot, ChatbotConfig, Vocabulary
    from .model import DTESNN, DTESNNConfig
except ImportError:
    from chatbot import DTESNNChatbot, ChatbotConfig, Vocabulary
    from model import DTESNN, DTESNNConfig


# ============================================================================
# Helper Functions
# ============================================================================

def _numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(v) for v in obj]
    return obj


def _extract_full_state(chatbot: DTESNNChatbot) -> Dict[str, Any]:
    """Extract complete serializable state from chatbot."""
    state = {
        "version": "1.0",
        "format": "dtesnn_chatbot",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        
        # Configuration
        "chatbot_config": asdict(chatbot.config),
        
        # Vocabulary
        "vocabulary": {
            "words": chatbot.vocab.words,
            "embedding_dim": chatbot.vocab.embedding_dim,
            "embeddings": chatbot.vocab.embeddings.tolist(),
            "word_to_idx": chatbot.vocab.word_to_idx,
            "idx_to_word": {str(k): v for k, v in chatbot.vocab.idx_to_word.items()},
        },
        
        # Training state
        "is_trained": chatbot._is_trained,
        "training_pairs": chatbot._training_pairs,
        "conversation_history": chatbot.conversation_history,
        
        # Model configuration
        "model_config": asdict(chatbot.model.config),
        
        # A000081 synchronizer state
        "synchronizer": {
            "base_order": chatbot.model.synchronizer.base_order,
            "tree_counts": chatbot.model.synchronizer.params.tree_counts,
            "total_trees": chatbot.model.synchronizer.params.total_trees,
            "cumulative_trees": chatbot.model.synchronizer.params.cumulative_counts,
        } if chatbot.model.synchronizer else None,
        
        # Model dimensions
        "dimensions": {
            "input_dim": chatbot.model.input_dim,
            "output_dim": chatbot.model._output_dim,
            "jsurface_dim": chatbot.model._jsurface_dim,
            "reservoir_dim": chatbot.model._reservoir_dim,
            "combined_dim": chatbot.model._combined_dim,
        },
        
        # Weights
        "weights": {},
    }
    
    # Extract JSurface ESN weights
    if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
        state["weights"]["jsurface"] = {
            "W_in": chatbot.model.jsurface._W_in.tolist() if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None else None,
            "W": chatbot.model.jsurface._W.tolist() if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None else None,
            "bias": chatbot.model.jsurface._bias.tolist() if hasattr(chatbot.model.jsurface, '_bias') and chatbot.model.jsurface._bias is not None else None,
        }
    
    # Extract Membrane Reservoir weights
    if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
        state["weights"]["reservoir"] = {
            "W_in": chatbot.model.reservoir._W_in.tolist() if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None else None,
            "W": chatbot.model.reservoir._W.tolist() if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None else None,
            "bias": chatbot.model.reservoir._bias.tolist() if hasattr(chatbot.model.reservoir, '_bias') and chatbot.model.reservoir._bias is not None else None,
        }
    
    # Extract Ridge Tree weights
    if chatbot.model.ridge_tree:
        state["weights"]["ridge_tree"] = {
            "W_out": chatbot.model.ridge_tree._W_out.tolist() if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None else None,
            "bias_out": chatbot.model.ridge_tree._bias_out.tolist() if hasattr(chatbot.model.ridge_tree, '_bias_out') and chatbot.model.ridge_tree._bias_out is not None else None,
        }
    
    # Current model state
    if chatbot.model.state:
        state["model_state"] = {
            k: v.tolist() if hasattr(v, 'tolist') else v 
            for k, v in chatbot.model.state.items()
        }
    
    return state


# ============================================================================
# JSON Export (Complete Dictionary)
# ============================================================================

def export_json(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
    indent: int = 2,
    include_embeddings: bool = True,
) -> str:
    """
    Export chatbot to JSON format (complete Python dictionary).
    
    This is the most human-readable format, containing all model data
    as nested JSON objects.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
    indent : int, default=2
        JSON indentation level
    include_embeddings : bool, default=True
        Whether to include vocabulary embeddings (can be large)
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.json')
    
    state = _extract_full_state(chatbot)
    
    if not include_embeddings:
        state["vocabulary"]["embeddings"] = f"<{len(state['vocabulary']['embeddings'])} embeddings omitted>"
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=indent, ensure_ascii=False)
    
    return str(filepath)


# ============================================================================
# Pickle Export (PKL.gz)
# ============================================================================

def export_pickle(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
    compress: bool = True,
) -> str:
    """
    Export chatbot to Pickle format.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
    compress : bool, default=True
        Whether to gzip compress the output
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    
    state = _extract_full_state(chatbot)
    
    # Convert lists back to numpy for pickle efficiency
    if "weights" in state:
        for component, weights in state["weights"].items():
            if weights:
                for key, value in weights.items():
                    if isinstance(value, list):
                        state["weights"][component][key] = np.array(value)
    
    if "vocabulary" in state and "embeddings" in state["vocabulary"]:
        if isinstance(state["vocabulary"]["embeddings"], list):
            state["vocabulary"]["embeddings"] = np.array(state["vocabulary"]["embeddings"])
    
    if compress:
        if not str(filepath).endswith('.gz'):
            filepath = Path(str(filepath) + '.gz')
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if not filepath.suffix:
            filepath = filepath.with_suffix('.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return str(filepath)


# ============================================================================
# NumPy Export (NPZ)
# ============================================================================

def export_numpy(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
) -> str:
    """
    Export chatbot weights to NumPy NPZ format.
    
    This format is efficient for numerical data and can be loaded
    in any language with NumPy-compatible libraries.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.npz')
    
    arrays = {}
    
    # Vocabulary embeddings
    arrays['vocab_embeddings'] = chatbot.vocab.embeddings
    
    # JSurface weights
    if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
        if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None:
            arrays['jsurface_W_in'] = chatbot.model.jsurface._W_in
        if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None:
            arrays['jsurface_W'] = chatbot.model.jsurface._W
        if hasattr(chatbot.model.jsurface, '_bias') and chatbot.model.jsurface._bias is not None:
            arrays['jsurface_bias'] = chatbot.model.jsurface._bias
    
    # Reservoir weights
    if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
        if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None:
            arrays['reservoir_W_in'] = chatbot.model.reservoir._W_in
        if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None:
            arrays['reservoir_W'] = chatbot.model.reservoir._W
        if hasattr(chatbot.model.reservoir, '_bias') and chatbot.model.reservoir._bias is not None:
            arrays['reservoir_bias'] = chatbot.model.reservoir._bias
    
    # Ridge tree weights
    if chatbot.model.ridge_tree:
        if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None:
            arrays['ridge_W_out'] = chatbot.model.ridge_tree._W_out
        if hasattr(chatbot.model.ridge_tree, '_bias_out') and chatbot.model.ridge_tree._bias_out is not None:
            arrays['ridge_bias_out'] = chatbot.model.ridge_tree._bias_out
    
    # Store config as JSON string
    config = {
        "chatbot_config": asdict(chatbot.config),
        "model_config": asdict(chatbot.model.config),
        "vocabulary_words": chatbot.vocab.words,
        "is_trained": chatbot._is_trained,
    }
    arrays['config_json'] = np.array([json.dumps(config)], dtype=object)
    
    np.savez_compressed(filepath, **arrays)
    
    return str(filepath)


# ============================================================================
# Config-Only JSON Export
# ============================================================================

def export_config_json(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
) -> str:
    """
    Export chatbot configuration and metadata only (no weights).
    
    Useful for documenting model architecture without large weight files.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    if not str(filepath).endswith('.config.json'):
        filepath = Path(str(filepath).replace('.json', '') + '.config.json')
    
    config = {
        "version": "1.0",
        "format": "dtesnn_config",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        
        "chatbot_config": asdict(chatbot.config),
        "model_config": asdict(chatbot.model.config),
        
        "vocabulary": {
            "size": len(chatbot.vocab.words),
            "embedding_dim": chatbot.vocab.embedding_dim,
            "special_tokens": ["<PAD>", "<UNK>", "<START>", "<END>", "<SEP>"],
        },
        
        "synchronizer": {
            "base_order": chatbot.model.synchronizer.base_order,
            "tree_counts": chatbot.model.synchronizer.params.tree_counts,
            "total_trees": chatbot.model.synchronizer.params.total_trees,
        } if chatbot.model.synchronizer else None,
        
        "dimensions": {
            "input_dim": chatbot.model.input_dim,
            "output_dim": chatbot.model._output_dim,
            "jsurface_dim": chatbot.model._jsurface_dim,
            "reservoir_dim": chatbot.model._reservoir_dim,
            "combined_dim": chatbot.model._combined_dim,
        },
        
        "training": {
            "is_trained": chatbot._is_trained,
            "num_training_pairs": len(chatbot._training_pairs),
        },
        
        "architecture": {
            "type": "DTESNN",
            "components": [
                "JSurface ESN (Elementary Differentials)",
                "Membrane Reservoir (P-System)",
                "Ridge Tree (B-Series Readout)",
            ],
            "synchronization": "A000081 (Rooted Tree Enumeration)",
        },
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


# ============================================================================
# ONNX Export
# ============================================================================

def export_onnx(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
) -> str:
    """
    Export chatbot to ONNX format.
    
    ONNX (Open Neural Network Exchange) is an open format for representing
    machine learning models, enabling interoperability between frameworks.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
        
    Returns
    -------
    str
        Path to exported file
        
    Notes
    -----
    The ONNX export creates a simplified computational graph representing
    the DTESNN forward pass. Due to the dynamic nature of reservoir computing,
    some operations are approximated.
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.onnx')
    
    # Try to use onnx library if available
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        HAS_ONNX = True
    except ImportError:
        HAS_ONNX = False
    
    if not HAS_ONNX:
        # Create a custom ONNX-like format without the library
        return _export_onnx_manual(chatbot, filepath)
    
    # Build ONNX graph
    input_dim = chatbot.config.embedding_dim
    output_dim = len(chatbot.vocab.words)
    
    # Input tensor
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_dim])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_dim])
    
    nodes = []
    initializers = []
    
    # Vocabulary embeddings
    vocab_emb = numpy_helper.from_array(
        chatbot.vocab.embeddings.astype(np.float32),
        name='vocab_embeddings'
    )
    initializers.append(vocab_emb)
    
    # JSurface weights
    if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
        if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None:
            W_in = numpy_helper.from_array(
                chatbot.model.jsurface._W_in.astype(np.float32),
                name='jsurface_W_in'
            )
            initializers.append(W_in)
        
        if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None:
            W = numpy_helper.from_array(
                chatbot.model.jsurface._W.astype(np.float32),
                name='jsurface_W'
            )
            initializers.append(W)
    
    # Reservoir weights
    if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
        if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None:
            W_in = numpy_helper.from_array(
                chatbot.model.reservoir._W_in.astype(np.float32),
                name='reservoir_W_in'
            )
            initializers.append(W_in)
        
        if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None:
            W = numpy_helper.from_array(
                chatbot.model.reservoir._W.astype(np.float32),
                name='reservoir_W'
            )
            initializers.append(W)
    
    # Ridge tree output weights
    if chatbot.model.ridge_tree:
        if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None:
            W_out = numpy_helper.from_array(
                chatbot.model.ridge_tree._W_out.astype(np.float32),
                name='ridge_W_out'
            )
            initializers.append(W_out)
            
            # Create MatMul node for output
            matmul_node = helper.make_node(
                'MatMul',
                inputs=['input', 'ridge_W_out'],
                outputs=['output'],
                name='ridge_output'
            )
            nodes.append(matmul_node)
    
    # If no nodes created, create identity
    if not nodes:
        identity_node = helper.make_node(
            'Identity',
            inputs=['input'],
            outputs=['output'],
            name='identity'
        )
        nodes.append(identity_node)
    
    # Create graph
    graph = helper.make_graph(
        nodes,
        'dtesnn_chatbot',
        [X],
        [Y],
        initializers
    )
    
    # Create model with metadata
    model = helper.make_model(graph, producer_name='nanechopy')
    model.doc_string = f"DTESNN Chatbot - Order {chatbot.config.base_order}"
    
    # Add metadata
    meta = model.metadata_props.add()
    meta.key = "dtesnn_order"
    meta.value = str(chatbot.config.base_order)
    
    meta = model.metadata_props.add()
    meta.key = "vocab_size"
    meta.value = str(len(chatbot.vocab.words))
    
    # Save
    onnx.save(model, str(filepath))
    
    return str(filepath)


def _export_onnx_manual(chatbot: DTESNNChatbot, filepath: Path) -> str:
    """Export ONNX format without onnx library (custom binary format)."""
    
    # Create a simplified ONNX-compatible structure as JSON + binary weights
    onnx_data = {
        "format": "dtesnn_onnx_manual",
        "version": "1.0",
        "ir_version": 8,
        "producer_name": "nanechopy",
        "producer_version": "1.0",
        "domain": "ai.dtesnn",
        "model_version": 1,
        "doc_string": f"DTESNN Chatbot - Order {chatbot.config.base_order}",
        
        "graph": {
            "name": "dtesnn_chatbot",
            "inputs": [
                {"name": "input", "type": "float32", "shape": [-1, chatbot.config.embedding_dim]}
            ],
            "outputs": [
                {"name": "output", "type": "float32", "shape": [-1, len(chatbot.vocab.words)]}
            ],
            "nodes": [],
            "initializers": [],
        },
        
        "metadata": {
            "dtesnn_order": chatbot.config.base_order,
            "vocab_size": len(chatbot.vocab.words),
            "embedding_dim": chatbot.config.embedding_dim,
            "is_trained": chatbot._is_trained,
        },
    }
    
    # Add weight tensors as base64-encoded data
    import base64
    
    def encode_array(arr, name):
        return {
            "name": name,
            "type": "float32",
            "shape": list(arr.shape),
            "data_b64": base64.b64encode(arr.astype(np.float32).tobytes()).decode('ascii'),
        }
    
    # Vocabulary embeddings
    onnx_data["graph"]["initializers"].append(
        encode_array(chatbot.vocab.embeddings, "vocab_embeddings")
    )
    
    # JSurface weights
    if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
        if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None:
            onnx_data["graph"]["initializers"].append(
                encode_array(chatbot.model.jsurface._W_in, "jsurface_W_in")
            )
        if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None:
            onnx_data["graph"]["initializers"].append(
                encode_array(chatbot.model.jsurface._W, "jsurface_W")
            )
    
    # Reservoir weights
    if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
        if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None:
            onnx_data["graph"]["initializers"].append(
                encode_array(chatbot.model.reservoir._W_in, "reservoir_W_in")
            )
        if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None:
            onnx_data["graph"]["initializers"].append(
                encode_array(chatbot.model.reservoir._W, "reservoir_W")
            )
    
    # Ridge tree weights
    if chatbot.model.ridge_tree:
        if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None:
            onnx_data["graph"]["initializers"].append(
                encode_array(chatbot.model.ridge_tree._W_out, "ridge_W_out")
            )
    
    # Save as JSON (with .onnx.json extension to indicate manual format)
    output_path = Path(str(filepath).replace('.onnx', '.onnx.json'))
    with open(output_path, 'w') as f:
        json.dump(onnx_data, f, indent=2)
    
    return str(output_path)


# ============================================================================
# GGUF Export
# ============================================================================

# GGUF Magic and Version
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF Data Types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def export_gguf(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
    quantize: bool = False,
) -> str:
    """
    Export chatbot to GGUF format.
    
    GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
    and other efficient inference engines.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
    quantize : bool, default=False
        Whether to quantize weights to int8
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.gguf')
    
    with open(filepath, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', GGUF_MAGIC))  # Magic
        f.write(struct.pack('<I', GGUF_VERSION))  # Version
        
        # Collect tensors
        tensors = []
        
        # Vocabulary embeddings
        tensors.append(("vocab.embeddings", chatbot.vocab.embeddings.astype(np.float32)))
        
        # JSurface weights
        if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
            if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None:
                tensors.append(("jsurface.W_in", chatbot.model.jsurface._W_in.astype(np.float32)))
            if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None:
                tensors.append(("jsurface.W", chatbot.model.jsurface._W.astype(np.float32)))
            if hasattr(chatbot.model.jsurface, '_bias') and chatbot.model.jsurface._bias is not None:
                tensors.append(("jsurface.bias", chatbot.model.jsurface._bias.astype(np.float32)))
        
        # Reservoir weights
        if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
            if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None:
                tensors.append(("reservoir.W_in", chatbot.model.reservoir._W_in.astype(np.float32)))
            if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None:
                tensors.append(("reservoir.W", chatbot.model.reservoir._W.astype(np.float32)))
            if hasattr(chatbot.model.reservoir, '_bias') and chatbot.model.reservoir._bias is not None:
                tensors.append(("reservoir.bias", chatbot.model.reservoir._bias.astype(np.float32)))
        
        # Ridge tree weights
        if chatbot.model.ridge_tree:
            if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None:
                tensors.append(("ridge.W_out", chatbot.model.ridge_tree._W_out.astype(np.float32)))
            if hasattr(chatbot.model.ridge_tree, '_bias_out') and chatbot.model.ridge_tree._bias_out is not None:
                tensors.append(("ridge.bias_out", chatbot.model.ridge_tree._bias_out.astype(np.float32)))
        
        # Metadata key-value pairs
        metadata = {
            "general.architecture": "dtesnn",
            "general.name": f"DTESNN Order {chatbot.config.base_order}",
            "general.author": "nanechopy",
            "general.version": "1.0",
            "dtesnn.order": chatbot.config.base_order,
            "dtesnn.embedding_dim": chatbot.config.embedding_dim,
            "dtesnn.vocab_size": len(chatbot.vocab.words),
            "dtesnn.is_trained": chatbot._is_trained,
            "dtesnn.total_trees": chatbot.model.synchronizer.params.total_trees if chatbot.model.synchronizer else 0,
        }
        
        # Write tensor count and metadata count
        f.write(struct.pack('<Q', len(tensors)))  # n_tensors
        f.write(struct.pack('<Q', len(metadata)))  # n_kv
        
        # Write metadata
        for key, value in metadata.items():
            # Write key
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # Write value type and value
            if isinstance(value, bool):
                f.write(struct.pack('<I', GGUF_TYPE_BOOL))
                f.write(struct.pack('<?', value))
            elif isinstance(value, int):
                f.write(struct.pack('<I', GGUF_TYPE_INT64))
                f.write(struct.pack('<q', value))
            elif isinstance(value, float):
                f.write(struct.pack('<I', GGUF_TYPE_FLOAT64))
                f.write(struct.pack('<d', value))
            elif isinstance(value, str):
                f.write(struct.pack('<I', GGUF_TYPE_STRING))
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
        
        # Write tensor info
        tensor_data_offset = f.tell()
        tensor_infos = []
        current_offset = 0
        
        for name, tensor in tensors:
            # Quantize if requested
            if quantize:
                tensor = (tensor * 127).astype(np.int8)
                dtype = GGUF_TYPE_INT8
            else:
                dtype = GGUF_TYPE_FLOAT32
            
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Write dimensions
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<Q', dim))
            
            # Write type
            f.write(struct.pack('<I', dtype))
            
            # Write offset (will be filled later)
            f.write(struct.pack('<Q', current_offset))
            
            tensor_infos.append((tensor, current_offset))
            current_offset += tensor.nbytes
        
        # Align to 32 bytes
        alignment = 32
        current_pos = f.tell()
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)
        
        # Write tensor data
        for tensor, _ in tensor_infos:
            f.write(tensor.tobytes())
    
    return str(filepath)


# ============================================================================
# Guile Scheme Export
# ============================================================================

def export_scheme(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
    include_weights: bool = True,
) -> str:
    """
    Export chatbot to Guile Scheme format.
    
    Creates an S-expression representation of the model that can be
    loaded and manipulated in Scheme/Lisp environments.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    filepath : str or Path
        Output file path
    include_weights : bool, default=True
        Whether to include weight matrices (can be large)
        
    Returns
    -------
    str
        Path to exported file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.scm')
    
    def to_sexp(obj, indent=0):
        """Convert Python object to S-expression string."""
        prefix = "  " * indent
        
        if obj is None:
            return "#f"
        elif isinstance(obj, bool):
            return "#t" if obj else "#f"
        elif isinstance(obj, (int, np.integer)):
            return str(int(obj))
        elif isinstance(obj, (float, np.floating)):
            return f"{float(obj):.8g}"
        elif isinstance(obj, str):
            # Escape special characters
            escaped = obj.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        elif isinstance(obj, np.ndarray):
            if obj.size > 1000 and not include_weights:
                return f'#(array shape: {list(obj.shape)} size: {obj.size})'
            # Convert to nested list representation
            if obj.ndim == 1:
                elements = ' '.join(f"{x:.6g}" for x in obj.flatten()[:100])
                if obj.size > 100:
                    elements += " ..."
                return f"#({elements})"
            elif obj.ndim == 2:
                rows = []
                for i, row in enumerate(obj[:20]):  # Limit rows
                    row_str = ' '.join(f"{x:.6g}" for x in row[:20])
                    if len(row) > 20:
                        row_str += " ..."
                    rows.append(f"#({row_str})")
                if len(obj) > 20:
                    rows.append("; ... more rows")
                return f"#(\n{prefix}  " + f"\n{prefix}  ".join(rows) + f"\n{prefix})"
            else:
                return f"#(array shape: {list(obj.shape)})"
        elif isinstance(obj, (list, tuple)):
            if len(obj) == 0:
                return "'()"
            elements = [to_sexp(x, indent + 1) for x in obj]
            if all(isinstance(x, (int, float, str)) for x in obj) and len(obj) < 10:
                return "(" + " ".join(elements) + ")"
            return "(\n" + prefix + "  " + f"\n{prefix}  ".join(elements) + f"\n{prefix})"
        elif isinstance(obj, dict):
            pairs = []
            for k, v in obj.items():
                key_str = k.replace('_', '-')  # Scheme naming convention
                val_str = to_sexp(v, indent + 1)
                pairs.append(f"({key_str} . {val_str})")
            return "(\n" + prefix + "  " + f"\n{prefix}  ".join(pairs) + f"\n{prefix})"
        else:
            return str(obj)
    
    # Build the Scheme module
    lines = [
        ";;; DTESNN Chatbot Model Export",
        f";;; Generated: {datetime.utcnow().isoformat()}Z",
        f";;; Order: {chatbot.config.base_order}",
        ";;;",
        "",
        "(define-module (dtesnn model)",
        "  #:export (dtesnn-config",
        "            dtesnn-vocabulary",
        "            dtesnn-weights",
        "            dtesnn-synchronizer",
        "            dtesnn-forward))",
        "",
        ";;; ============================================================",
        ";;; Configuration",
        ";;; ============================================================",
        "",
        "(define dtesnn-config",
        f"  '((version . \"1.0\")",
        f"    (format . \"dtesnn-scheme\")",
        f"    (base-order . {chatbot.config.base_order})",
        f"    (embedding-dim . {chatbot.config.embedding_dim})",
        f"    (units-per-component . {chatbot.config.units_per_component})",
        f"    (max-response-length . {chatbot.config.max_response_length})",
        f"    (temperature . {chatbot.config.temperature})",
        f"    (top-k . {chatbot.config.top_k})",
        f"    (is-trained . {'#t' if chatbot._is_trained else '#f'})))",
        "",
        ";;; ============================================================",
        ";;; A000081 Synchronizer",
        ";;; ============================================================",
        "",
    ]
    
    if chatbot.model.synchronizer:
        sync = chatbot.model.synchronizer
        lines.extend([
            "(define dtesnn-synchronizer",
            f"  '((base-order . {sync.base_order})",
            f"    (tree-counts . #({' '.join(str(x) for x in sync.params.tree_counts)}))",
            f"    (total-trees . {sync.params.total_trees})",
            f"    (cumulative-trees . #({' '.join(str(x) for x in sync.params.cumulative_counts)}))))",
            "",
        ])
    else:
        lines.extend([
            "(define dtesnn-synchronizer #f)",
            "",
        ])
    
    lines.extend([
        ";;; ============================================================",
        ";;; Vocabulary",
        ";;; ============================================================",
        "",
        "(define dtesnn-vocabulary",
        f"  '((size . {len(chatbot.vocab.words)})",
        f"    (embedding-dim . {chatbot.vocab.embedding_dim})",
        "    (special-tokens . (\"<PAD>\" \"<UNK>\" \"<START>\" \"<END>\" \"<SEP>\"))",
    ])
    
    # Add vocabulary words (first 100)
    vocab_preview = chatbot.vocab.words[:100]
    lines.append(f"    (words . #({' '.join(repr(w) for w in vocab_preview)}")
    if len(chatbot.vocab.words) > 100:
        lines.append(f"              ; ... {len(chatbot.vocab.words) - 100} more words")
    lines.append("              ))")
    
    # Add embeddings if requested
    if include_weights:
        lines.append(f"    (embeddings . {to_sexp(chatbot.vocab.embeddings, 2)})")
    else:
        lines.append(f"    (embeddings . #(matrix {chatbot.vocab.embeddings.shape[0]} x {chatbot.vocab.embeddings.shape[1]}))")
    
    lines.append("    ))")
    lines.append("")
    
    lines.extend([
        ";;; ============================================================",
        ";;; Model Weights",
        ";;; ============================================================",
        "",
        "(define dtesnn-weights",
        "  '(",
    ])
    
    # JSurface weights
    if chatbot.model.jsurface and chatbot.model.jsurface.initialized:
        lines.append("    ;; JSurface ESN (Elementary Differentials)")
        lines.append("    (jsurface")
        if hasattr(chatbot.model.jsurface, '_W_in') and chatbot.model.jsurface._W_in is not None:
            if include_weights:
                lines.append(f"      (W-in . {to_sexp(chatbot.model.jsurface._W_in, 3)})")
            else:
                shape = chatbot.model.jsurface._W_in.shape
                lines.append(f"      (W-in . #(matrix {shape[0]} x {shape[1]}))")
        if hasattr(chatbot.model.jsurface, '_W') and chatbot.model.jsurface._W is not None:
            if include_weights:
                lines.append(f"      (W . {to_sexp(chatbot.model.jsurface._W, 3)})")
            else:
                shape = chatbot.model.jsurface._W.shape
                lines.append(f"      (W . #(matrix {shape[0]} x {shape[1]}))")
        lines.append("    )")
    
    # Reservoir weights
    if chatbot.model.reservoir and chatbot.model.reservoir.initialized:
        lines.append("    ;; Membrane Reservoir (P-System)")
        lines.append("    (reservoir")
        if hasattr(chatbot.model.reservoir, '_W_in') and chatbot.model.reservoir._W_in is not None:
            if include_weights:
                lines.append(f"      (W-in . {to_sexp(chatbot.model.reservoir._W_in, 3)})")
            else:
                shape = chatbot.model.reservoir._W_in.shape
                lines.append(f"      (W-in . #(matrix {shape[0]} x {shape[1]}))")
        if hasattr(chatbot.model.reservoir, '_W') and chatbot.model.reservoir._W is not None:
            if include_weights:
                lines.append(f"      (W . {to_sexp(chatbot.model.reservoir._W, 3)})")
            else:
                shape = chatbot.model.reservoir._W.shape
                lines.append(f"      (W . #(matrix {shape[0]} x {shape[1]}))")
        lines.append("    )")
    
    # Ridge tree weights
    if chatbot.model.ridge_tree:
        lines.append("    ;; Ridge Tree (B-Series Readout)")
        lines.append("    (ridge-tree")
        if hasattr(chatbot.model.ridge_tree, '_W_out') and chatbot.model.ridge_tree._W_out is not None:
            if include_weights:
                lines.append(f"      (W-out . {to_sexp(chatbot.model.ridge_tree._W_out, 3)})")
            else:
                shape = chatbot.model.ridge_tree._W_out.shape
                lines.append(f"      (W-out . #(matrix {shape[0]} x {shape[1]}))")
        lines.append("    )")
    
    lines.append("  ))")
    lines.append("")
    
    # Add forward pass function
    lines.extend([
        ";;; ============================================================",
        ";;; Forward Pass (Pseudocode)",
        ";;; ============================================================",
        "",
        "(define (dtesnn-forward input)",
        "  \"Compute DTESNN forward pass.",
        "   input: vector of embedding dimension",
        "   returns: vector of vocabulary size (logits)\"",
        "  ",
        "  ;; 1. JSurface ESN: Elementary differential computation",
        "  ;;    x_j = tanh(W_in_j * input + W_j * x_j_prev)",
        "  ",
        "  ;; 2. Membrane Reservoir: P-System dynamics",
        "  ;;    x_m = (1 - leak) * x_m_prev + leak * tanh(W_in_m * x_j + W_m * x_m_prev)",
        "  ",
        "  ;; 3. Ridge Tree: B-Series readout",
        "  ;;    output = W_out * [x_j; x_m] + bias",
        "  ",
        "  ;; 4. Apply softmax for token probabilities",
        "  ;;    probs = softmax(output / temperature)",
        "  ",
        "  'not-implemented-see-python)",
        "",
        ";;; End of DTESNN Model Export",
    ])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return str(filepath)


# ============================================================================
# Export All Formats
# ============================================================================

def export_all_formats(
    chatbot: DTESNNChatbot,
    base_path: Union[str, Path],
    include_weights_in_scheme: bool = False,
) -> Dict[str, str]:
    """
    Export chatbot to all supported formats.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to export
    base_path : str or Path
        Base path for output files (without extension)
    include_weights_in_scheme : bool, default=False
        Whether to include full weights in Scheme export
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping format names to file paths
    """
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # JSON (complete dictionary)
    try:
        results['json'] = export_json(chatbot, f"{base_path}.json")
        print(f"✓ JSON: {results['json']}")
    except Exception as e:
        print(f"✗ JSON export failed: {e}")
    
    # Pickle (compressed)
    try:
        results['pkl'] = export_pickle(chatbot, f"{base_path}.pkl", compress=True)
        print(f"✓ PKL.gz: {results['pkl']}")
    except Exception as e:
        print(f"✗ PKL export failed: {e}")
    
    # NumPy (NPZ)
    try:
        results['npz'] = export_numpy(chatbot, f"{base_path}.npz")
        print(f"✓ NPZ: {results['npz']}")
    except Exception as e:
        print(f"✗ NPZ export failed: {e}")
    
    # Config JSON (metadata only)
    try:
        results['config'] = export_config_json(chatbot, f"{base_path}.config.json")
        print(f"✓ Config JSON: {results['config']}")
    except Exception as e:
        print(f"✗ Config JSON export failed: {e}")
    
    # ONNX
    try:
        results['onnx'] = export_onnx(chatbot, f"{base_path}.onnx")
        print(f"✓ ONNX: {results['onnx']}")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
    
    # GGUF
    try:
        results['gguf'] = export_gguf(chatbot, f"{base_path}.gguf")
        print(f"✓ GGUF: {results['gguf']}")
    except Exception as e:
        print(f"✗ GGUF export failed: {e}")
    
    # Guile Scheme
    try:
        results['scm'] = export_scheme(chatbot, f"{base_path}.scm", include_weights=include_weights_in_scheme)
        print(f"✓ Scheme: {results['scm']}")
    except Exception as e:
        print(f"✗ Scheme export failed: {e}")
    
    return results


__all__ = [
    "export_json",
    "export_pickle",
    "export_numpy",
    "export_config_json",
    "export_onnx",
    "export_gguf",
    "export_scheme",
    "export_all_formats",
]
