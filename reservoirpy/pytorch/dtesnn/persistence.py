"""
====================================
DTESNN Persistence (:mod:`reservoirpy.pytorch.dtesnn.persistence`)
====================================

Model persistence utilities for saving and loading trained DTESNN chatbots.

This module provides functions to serialize and deserialize the complete state
of a trained DTESNNChatbot, including:
- Configuration parameters
- Vocabulary and embeddings
- DTESNN model weights and state
- Training data and conversation history

Supported formats:
- pickle (.pkl) - Full Python object serialization
- numpy (.npz) - Compressed numpy arrays for weights only
- JSON (.json) - Configuration and metadata only

Examples
--------
>>> from reservoirpy.pytorch.dtesnn.persistence import save_chatbot, load_chatbot
>>> 
>>> # Save a trained chatbot
>>> save_chatbot(chatbot, "my_chatbot.pkl")
>>> 
>>> # Load the chatbot
>>> loaded_chatbot = load_chatbot("my_chatbot.pkl")
>>> response = loaded_chatbot.respond("hello")
"""

# License: MIT License
# Copyright: nanechopy contributors

import os
import json
import pickle
import gzip
from dataclasses import asdict
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np

try:
    from .chatbot import DTESNNChatbot, ChatbotConfig, Vocabulary, STANDARD_VOCABULARY
    from .model import DTESNN, DTESNNConfig
except ImportError:
    from chatbot import DTESNNChatbot, ChatbotConfig, Vocabulary, STANDARD_VOCABULARY
    from model import DTESNN, DTESNNConfig


def _extract_model_state(model: DTESNN) -> Dict[str, Any]:
    """Extract serializable state from DTESNN model.
    
    Parameters
    ----------
    model : DTESNN
        The DTESNN model to extract state from
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all model state
    """
    state = {
        "config": asdict(model.config),
        "input_dim": model.input_dim,
        "output_dim": model._output_dim,
        "seed": model.seed,
        "initialized": model.initialized,
        "plantings": model.plantings,
        "_jsurface_dim": model._jsurface_dim,
        "_reservoir_dim": model._reservoir_dim,
        "_combined_dim": model._combined_dim,
    }
    
    # Extract synchronizer parameters
    if model.synchronizer:
        state["synchronizer"] = {
            "base_order": model.synchronizer.base_order,
            "tree_counts": model.synchronizer.params.tree_counts,
            "total_trees": model.synchronizer.params.total_trees,
        }
    
    # Extract JSurface ESN state
    if model.jsurface and model.jsurface.initialized:
        jsurface_state = {
            "W_in": model.jsurface._W_in.tolist() if hasattr(model.jsurface, '_W_in') and model.jsurface._W_in is not None else None,
            "W": model.jsurface._W.tolist() if hasattr(model.jsurface, '_W') and model.jsurface._W is not None else None,
            "bias": model.jsurface._bias.tolist() if hasattr(model.jsurface, '_bias') and model.jsurface._bias is not None else None,
        }
        state["jsurface"] = jsurface_state
    
    # Extract Membrane Reservoir state
    if model.reservoir and model.reservoir.initialized:
        reservoir_state = {
            "W_in": model.reservoir._W_in.tolist() if hasattr(model.reservoir, '_W_in') and model.reservoir._W_in is not None else None,
            "W": model.reservoir._W.tolist() if hasattr(model.reservoir, '_W') and model.reservoir._W is not None else None,
            "bias": model.reservoir._bias.tolist() if hasattr(model.reservoir, '_bias') and model.reservoir._bias is not None else None,
        }
        state["reservoir"] = reservoir_state
    
    # Extract Ridge Tree state (trained weights)
    if model.ridge_tree:
        ridge_state = {
            "W_out": model.ridge_tree._W_out.tolist() if hasattr(model.ridge_tree, '_W_out') and model.ridge_tree._W_out is not None else None,
            "bias_out": model.ridge_tree._bias_out.tolist() if hasattr(model.ridge_tree, '_bias_out') and model.ridge_tree._bias_out is not None else None,
        }
        state["ridge_tree"] = ridge_state
    
    # Extract current state
    if model.state:
        state["model_state"] = {
            k: v.tolist() if hasattr(v, 'tolist') else v 
            for k, v in model.state.items()
        }
    
    return state


def _restore_model_state(model: DTESNN, state: Dict[str, Any]) -> None:
    """Restore model state from dictionary.
    
    Parameters
    ----------
    model : DTESNN
        The DTESNN model to restore state to
    state : Dict[str, Any]
        Dictionary containing model state
    """
    # Restore basic attributes
    model.input_dim = state.get("input_dim")
    model._output_dim = state.get("output_dim")
    model.initialized = state.get("initialized", False)
    model.plantings = state.get("plantings", {})
    model._jsurface_dim = state.get("_jsurface_dim", 0)
    model._reservoir_dim = state.get("_reservoir_dim", 0)
    model._combined_dim = state.get("_combined_dim", 0)
    
    # Restore JSurface ESN weights
    if "jsurface" in state and model.jsurface:
        js = state["jsurface"]
        if js.get("W_in") is not None:
            model.jsurface._W_in = np.array(js["W_in"])
        if js.get("W") is not None:
            model.jsurface._W = np.array(js["W"])
        if js.get("bias") is not None:
            model.jsurface._bias = np.array(js["bias"])
        model.jsurface.initialized = True
    
    # Restore Membrane Reservoir weights
    if "reservoir" in state and model.reservoir:
        rs = state["reservoir"]
        if rs.get("W_in") is not None:
            model.reservoir._W_in = np.array(rs["W_in"])
        if rs.get("W") is not None:
            model.reservoir._W = np.array(rs["W"])
        if rs.get("bias") is not None:
            model.reservoir._bias = np.array(rs["bias"])
        model.reservoir.initialized = True
    
    # Restore Ridge Tree weights
    if "ridge_tree" in state and model.ridge_tree:
        rt = state["ridge_tree"]
        if rt.get("W_out") is not None:
            model.ridge_tree._W_out = np.array(rt["W_out"])
        if rt.get("bias_out") is not None:
            model.ridge_tree._bias_out = np.array(rt["bias_out"])
    
    # Restore model state
    if "model_state" in state:
        model.state = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in state["model_state"].items()
        }


def save_chatbot(
    chatbot: DTESNNChatbot,
    filepath: Union[str, Path],
    compress: bool = True,
    include_history: bool = True,
) -> str:
    """Save a trained DTESNNChatbot to persistent storage.
    
    Parameters
    ----------
    chatbot : DTESNNChatbot
        The chatbot to save
    filepath : str or Path
        Path to save the chatbot to. Extension determines format:
        - .pkl or .pickle: Full pickle serialization
        - .pkl.gz: Compressed pickle
        - .npz: Numpy arrays only (weights)
    compress : bool, default=True
        Whether to compress the saved file
    include_history : bool, default=True
        Whether to include conversation history
        
    Returns
    -------
    str
        Path to the saved file
        
    Examples
    --------
    >>> chatbot = DTESNNChatbot(ChatbotConfig(base_order=6))
    >>> chatbot.train()
    >>> save_chatbot(chatbot, "chatbot_order6.pkl.gz")
    """
    filepath = Path(filepath)
    
    # Prepare data to save
    data = {
        "version": "1.0",
        "format": "dtesnn_chatbot",
        
        # Configuration
        "chatbot_config": asdict(chatbot.config),
        
        # Vocabulary
        "vocabulary": {
            "words": chatbot.vocab.words,
            "embedding_dim": chatbot.vocab.embedding_dim,
            "embeddings": chatbot.vocab.embeddings.tolist(),
        },
        
        # Training state
        "is_trained": chatbot._is_trained,
        "training_pairs": chatbot._training_pairs,
        
        # Model state
        "model_state": _extract_model_state(chatbot.model),
        
        # RNG state for reproducibility
        "rng_state": chatbot.rng.bit_generator.state,
    }
    
    # Optionally include conversation history
    if include_history:
        data["conversation_history"] = chatbot.conversation_history
    
    # Save based on file extension
    suffix = "".join(filepath.suffixes).lower()
    
    if ".npz" in suffix:
        # Save as numpy arrays (weights only)
        np_data = {}
        
        # Flatten nested dicts for numpy
        if "model_state" in data:
            ms = data["model_state"]
            if "jsurface" in ms:
                for k, v in ms["jsurface"].items():
                    if v is not None:
                        np_data[f"jsurface_{k}"] = np.array(v)
            if "reservoir" in ms:
                for k, v in ms["reservoir"].items():
                    if v is not None:
                        np_data[f"reservoir_{k}"] = np.array(v)
            if "ridge_tree" in ms:
                for k, v in ms["ridge_tree"].items():
                    if v is not None:
                        np_data[f"ridge_tree_{k}"] = np.array(v)
        
        # Save vocabulary embeddings
        np_data["vocab_embeddings"] = np.array(data["vocabulary"]["embeddings"])
        
        # Save config as JSON string
        config_str = json.dumps({
            "chatbot_config": data["chatbot_config"],
            "vocabulary_words": data["vocabulary"]["words"],
            "is_trained": data["is_trained"],
        })
        np_data["config_json"] = np.array([config_str], dtype=object)
        
        np.savez_compressed(filepath, **np_data)
        
    elif compress or ".gz" in suffix:
        # Compressed pickle
        if not str(filepath).endswith('.gz'):
            filepath = Path(str(filepath) + '.gz')
        
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    else:
        # Standard pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return str(filepath)


def load_chatbot(
    filepath: Union[str, Path],
    device: Optional[str] = None,
) -> DTESNNChatbot:
    """Load a saved DTESNNChatbot from persistent storage.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved chatbot file
    device : str, optional
        Device to load the model to
        
    Returns
    -------
    DTESNNChatbot
        The loaded chatbot, ready to use
        
    Examples
    --------
    >>> chatbot = load_chatbot("chatbot_order6.pkl.gz")
    >>> response = chatbot.respond("hello")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Chatbot file not found: {filepath}")
    
    suffix = "".join(filepath.suffixes).lower()
    
    if ".npz" in suffix:
        # Load from numpy format
        np_data = np.load(filepath, allow_pickle=True)
        
        # Parse config
        config_str = str(np_data["config_json"][0])
        config_data = json.loads(config_str)
        
        # Create config
        chatbot_config = ChatbotConfig(**config_data["chatbot_config"])
        
        # Create vocabulary
        vocab = Vocabulary(
            words=config_data["vocabulary_words"],
            embedding_dim=chatbot_config.embedding_dim,
        )
        vocab.embeddings = np_data["vocab_embeddings"]
        
        # Create chatbot
        chatbot = DTESNNChatbot(config=chatbot_config, vocabulary=vocab)
        chatbot._is_trained = config_data["is_trained"]
        
        # Restore model weights
        if chatbot._is_trained:
            # Need to initialize model first
            dummy_X = np.zeros((1, chatbot_config.embedding_dim))
            dummy_y = np.zeros((1, vocab.size))
            chatbot.model.initialize(dummy_X, dummy_y)
            
            # Restore weights
            if "jsurface_W_in" in np_data and chatbot.model.jsurface:
                chatbot.model.jsurface._W_in = np_data["jsurface_W_in"]
            if "jsurface_W" in np_data and chatbot.model.jsurface:
                chatbot.model.jsurface._W = np_data["jsurface_W"]
            if "reservoir_W_in" in np_data and chatbot.model.reservoir:
                chatbot.model.reservoir._W_in = np_data["reservoir_W_in"]
            if "reservoir_W" in np_data and chatbot.model.reservoir:
                chatbot.model.reservoir._W = np_data["reservoir_W"]
            if "ridge_tree_W_out" in np_data and chatbot.model.ridge_tree:
                chatbot.model.ridge_tree._W_out = np_data["ridge_tree_W_out"]
        
        return chatbot
    
    elif ".gz" in suffix:
        # Compressed pickle
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        # Standard pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
    # Validate format
    if data.get("format") != "dtesnn_chatbot":
        raise ValueError(f"Invalid file format: {data.get('format')}")
    
    # Create config
    chatbot_config = ChatbotConfig(**data["chatbot_config"])
    
    # Create vocabulary
    vocab_data = data["vocabulary"]
    vocab = Vocabulary(
        words=vocab_data["words"],
        embedding_dim=vocab_data["embedding_dim"],
    )
    vocab.embeddings = np.array(vocab_data["embeddings"])
    
    # Create chatbot
    chatbot = DTESNNChatbot(config=chatbot_config, vocabulary=vocab)
    
    # Restore training state
    chatbot._is_trained = data["is_trained"]
    chatbot._training_pairs = data.get("training_pairs", [])
    
    # Restore conversation history
    if "conversation_history" in data:
        chatbot.conversation_history = data["conversation_history"]
    
    # Restore RNG state
    if "rng_state" in data:
        chatbot.rng.bit_generator.state = data["rng_state"]
    
    # Restore model state
    if chatbot._is_trained and "model_state" in data:
        model_state = data["model_state"]
        
        # Re-create model components if needed
        if not chatbot.model.initialized:
            dummy_X = np.zeros((1, chatbot_config.embedding_dim))
            dummy_y = np.zeros((1, vocab.size))
            chatbot.model.initialize(dummy_X, dummy_y)
        
        # Restore weights
        _restore_model_state(chatbot.model, model_state)
    
    return chatbot


def get_model_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a saved chatbot without fully loading it.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved chatbot file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with model information
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Chatbot file not found: {filepath}")
    
    suffix = "".join(filepath.suffixes).lower()
    
    if ".npz" in suffix:
        np_data = np.load(filepath, allow_pickle=True)
        config_str = str(np_data["config_json"][0])
        config_data = json.loads(config_str)
        
        return {
            "format": "npz",
            "chatbot_config": config_data["chatbot_config"],
            "is_trained": config_data["is_trained"],
            "vocab_size": len(config_data["vocabulary_words"]),
        }
    
    elif ".gz" in suffix:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    
    return {
        "version": data.get("version"),
        "format": data.get("format"),
        "chatbot_config": data.get("chatbot_config"),
        "is_trained": data.get("is_trained"),
        "vocab_size": len(data.get("vocabulary", {}).get("words", [])),
        "has_history": "conversation_history" in data,
        "history_length": len(data.get("conversation_history", [])),
    }


# Convenience methods to add to DTESNNChatbot class
def _chatbot_save(self, filepath: Union[str, Path], compress: bool = True) -> str:
    """Save this chatbot to a file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to save to
    compress : bool, default=True
        Whether to compress the file
        
    Returns
    -------
    str
        Path to saved file
    """
    return save_chatbot(self, filepath, compress=compress)


@classmethod
def _chatbot_load(cls, filepath: Union[str, Path]) -> "DTESNNChatbot":
    """Load a chatbot from a file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to load from
        
    Returns
    -------
    DTESNNChatbot
        Loaded chatbot
    """
    return load_chatbot(filepath)


# Patch the DTESNNChatbot class with save/load methods
DTESNNChatbot.save = _chatbot_save
DTESNNChatbot.load = _chatbot_load


__all__ = [
    "save_chatbot",
    "load_chatbot",
    "get_model_info",
]
