"""
====================================
DTESNN Chatbot (:mod:`reservoirpy.pytorch.dtesnn.chatbot`)
====================================

A chatbot implementation using DTESNN (Deep Tree Echo State Neural Network)
to explore how different A000081 orders affect response generation.

The chatbot uses:
- Standard vocabulary with embeddings
- DTESNN for sequence processing
- Temperature-based sampling for response generation

This allows us to observe how the tree-structured architecture
influences language generation at different orders.
"""

# License: MIT License
# Copyright: nanechopy contributors

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import re

# Import DTESNN components
try:
    from .model import DTESNN, DTESNNConfig
    from .synchronizer import A000081Synchronizer
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'autognosis'))
    from model import DTESNN, DTESNNConfig
    from synchronizer import A000081Synchronizer


# ============================================================================
# Standard Vocabulary
# ============================================================================

# Core vocabulary for the chatbot - common English words plus special tokens
STANDARD_VOCABULARY = [
    # Special tokens
    "<PAD>", "<UNK>", "<START>", "<END>", "<SEP>",
    
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "this", "that", "these", "those", "who", "what", "which", "whom", "whose",
    
    # Common verbs
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "having", "do", "does", "did", "doing", "done",
    "will", "would", "could", "should", "may", "might", "must", "can", "shall",
    "say", "said", "think", "thought", "know", "knew", "known", "see", "saw", "seen",
    "want", "wanted", "need", "needed", "like", "liked", "love", "loved",
    "go", "went", "gone", "going", "come", "came", "coming",
    "make", "made", "making", "take", "took", "taken", "taking",
    "get", "got", "getting", "give", "gave", "given", "giving",
    "find", "found", "finding", "tell", "told", "telling",
    "ask", "asked", "asking", "use", "used", "using",
    "try", "tried", "trying", "help", "helped", "helping",
    "feel", "felt", "feeling", "become", "became", "becoming",
    "leave", "left", "leaving", "put", "putting",
    "mean", "meant", "meaning", "keep", "kept", "keeping",
    "let", "begin", "began", "begun", "seem", "seemed",
    "show", "showed", "shown", "hear", "heard", "hearing",
    "play", "played", "playing", "run", "ran", "running",
    "move", "moved", "moving", "live", "lived", "living",
    "believe", "believed", "hold", "held", "holding",
    "bring", "brought", "happen", "happened", "write", "wrote", "written",
    "provide", "provided", "sit", "sat", "stand", "stood",
    "lose", "lost", "pay", "paid", "meet", "met",
    "include", "included", "continue", "continued", "set", "learn", "learned",
    "change", "changed", "lead", "led", "understand", "understood",
    "watch", "watched", "follow", "followed", "stop", "stopped",
    "create", "created", "speak", "spoke", "spoken", "read",
    "spend", "spent", "grow", "grew", "grown", "open", "opened",
    "walk", "walked", "win", "won", "offer", "offered",
    "remember", "remembered", "consider", "considered", "appear", "appeared",
    "buy", "bought", "wait", "waited", "serve", "served",
    "die", "died", "send", "sent", "expect", "expected",
    "build", "built", "stay", "stayed", "fall", "fell", "fallen",
    "cut", "reach", "reached", "kill", "killed", "remain", "remained",
    
    # Common nouns
    "time", "year", "people", "way", "day", "man", "woman", "child", "children",
    "world", "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point", "home",
    "water", "room", "mother", "area", "money", "story", "fact", "month", "lot",
    "right", "study", "book", "eye", "job", "word", "business", "issue", "side",
    "kind", "head", "house", "service", "friend", "father", "power", "hour", "game",
    "line", "end", "member", "law", "car", "city", "community", "name", "president",
    "team", "minute", "idea", "body", "information", "back", "parent", "face",
    "others", "level", "office", "door", "health", "person", "art", "war",
    "history", "party", "result", "change", "morning", "reason", "research", "girl",
    "guy", "moment", "air", "teacher", "force", "education", "tree", "echo",
    "network", "neural", "model", "data", "input", "output", "state", "memory",
    "thought", "mind", "brain", "intelligence", "learning", "pattern", "structure",
    "order", "system", "process", "function", "response", "question", "answer",
    
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "human",
    "local", "sure", "free", "better", "true", "whole", "special", "hard",
    "best", "possible", "full", "real", "clear", "simple", "recent", "certain",
    "personal", "open", "red", "difficult", "available", "likely", "short",
    "single", "past", "strong", "happy", "serious", "ready", "deep", "fast",
    "natural", "similar", "central", "nice", "interesting", "beautiful",
    
    # Common adverbs
    "not", "also", "very", "often", "however", "too", "usually", "really",
    "early", "never", "always", "sometimes", "together", "likely", "simply",
    "generally", "instead", "actually", "already", "ever", "well", "now",
    "then", "here", "there", "where", "when", "why", "how", "again",
    "still", "just", "more", "most", "even", "back", "only", "yet",
    
    # Prepositions and conjunctions
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "over", "after", "beneath", "under", "above", "between", "through",
    "during", "before", "without", "against", "within", "along", "following",
    "across", "behind", "beyond", "plus", "except", "around", "among", "per",
    "and", "or", "but", "if", "because", "as", "until", "while", "although",
    "though", "since", "unless", "so", "than", "whether", "once", "both",
    
    # Articles and determiners
    "the", "a", "an", "some", "any", "no", "every", "each", "all", "many",
    "much", "more", "most", "few", "several", "enough", "such",
    
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third", "fourth", "fifth",
    
    # Question words
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    
    # Common expressions
    "yes", "no", "maybe", "please", "thank", "thanks", "sorry", "hello", "hi",
    "goodbye", "bye", "okay", "ok", "well", "oh", "ah", "um", "uh",
    
    # Punctuation (as tokens)
    ".", ",", "!", "?", ":", ";", "'", '"', "-", "(", ")", "[", "]",
]


@dataclass
class Vocabulary:
    """Vocabulary manager for the chatbot.
    
    Handles word-to-index and index-to-word mappings,
    plus simple embedding generation.
    """
    words: List[str] = field(default_factory=list)
    word_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_word: Dict[int, str] = field(default_factory=dict)
    embedding_dim: int = 64
    embeddings: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.words:
            self.words = STANDARD_VOCABULARY.copy()
        self._build_mappings()
        self._generate_embeddings()
    
    def _build_mappings(self):
        """Build word-index mappings."""
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.words)}
    
    def _generate_embeddings(self, seed: int = 42):
        """Generate random embeddings for vocabulary.
        
        In a real system, these would be pre-trained embeddings.
        Here we use random embeddings with some structure.
        """
        rng = np.random.default_rng(seed)
        
        vocab_size = len(self.words)
        self.embeddings = np.zeros((vocab_size, self.embedding_dim))
        
        for idx, word in enumerate(self.words):
            # Base random embedding
            emb = rng.standard_normal(self.embedding_dim) * 0.1
            
            # Add some structure based on word properties
            # Length feature
            emb[0] = len(word) / 10.0
            
            # First character feature
            if word and word[0].isalpha():
                emb[1] = (ord(word[0].lower()) - ord('a')) / 26.0
            
            # Vowel ratio feature
            if word:
                vowels = sum(1 for c in word.lower() if c in 'aeiou')
                emb[2] = vowels / len(word)
            
            # Special token features
            if word.startswith('<') and word.endswith('>'):
                emb[3] = 1.0
            
            self.embeddings[idx] = emb
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        self.embeddings = self.embeddings / norms
    
    @property
    def size(self) -> int:
        return len(self.words)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        # Simple tokenization
        tokens = self._tokenize(text)
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx.get("<UNK>", 1))
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        words = []
        for idx in indices:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if not word.startswith('<'):  # Skip special tokens
                    words.append(word)
        return ' '.join(words)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split
        text = text.lower()
        # Separate punctuation
        text = re.sub(r'([.,!?;:\'\"\-\(\)\[\]])', r' \1 ', text)
        # Split on whitespace
        tokens = text.split()
        return tokens
    
    def get_embedding(self, idx: int) -> np.ndarray:
        """Get embedding for token index."""
        if 0 <= idx < len(self.embeddings):
            return self.embeddings[idx]
        return self.embeddings[self.word_to_idx.get("<UNK>", 1)]
    
    def embed_sequence(self, indices: List[int]) -> np.ndarray:
        """Get embeddings for sequence of indices."""
        return np.array([self.get_embedding(idx) for idx in indices])


@dataclass
class ChatbotConfig:
    """Configuration for DTESNN chatbot."""
    base_order: int = 5
    embedding_dim: int = 64
    units_per_component: int = 32
    max_response_length: int = 50
    temperature: float = 0.8
    top_k: int = 10
    warmup_steps: int = 10
    seed: int = 42


class DTESNNChatbot:
    """A chatbot powered by DTESNN.
    
    Uses the A000081-structured network for sequence processing
    and response generation.
    
    Parameters
    ----------
    config : ChatbotConfig
        Chatbot configuration
    vocabulary : Vocabulary, optional
        Vocabulary to use (creates default if not provided)
        
    Attributes
    ----------
    model : DTESNN
        The DTESNN model
    vocab : Vocabulary
        The vocabulary
    conversation_history : List[str]
        History of conversation turns
        
    Examples
    --------
    >>> chatbot = DTESNNChatbot(ChatbotConfig(base_order=5))
    >>> response = chatbot.respond("Hello, how are you?")
    >>> print(response)
    """
    
    def __init__(
        self,
        config: Optional[ChatbotConfig] = None,
        vocabulary: Optional[Vocabulary] = None,
    ):
        self.config = config or ChatbotConfig()
        self.vocab = vocabulary or Vocabulary(embedding_dim=self.config.embedding_dim)
        
        # Create DTESNN model
        dtesnn_config = DTESNNConfig(
            base_order=self.config.base_order,
            units_per_component=self.config.units_per_component,
            leak_rate=0.3,
            spectral_radius=0.9,
            use_symplectic=True,
            use_hierarchical_ridge=True,
        )
        
        self.model = DTESNN(
            config=dtesnn_config,
            seed=self.config.seed,
        )
        
        # Conversation state
        self.conversation_history: List[str] = []
        self.rng = np.random.default_rng(self.config.seed)
        
        # Training data for response patterns
        self._training_pairs: List[Tuple[str, str]] = []
        self._is_trained = False
        
        # Initialize with default training data
        self._add_default_training_data()
    
    def _add_default_training_data(self):
        """Add default conversational training pairs."""
        default_pairs = [
            # Greetings
            ("hello", "hello there how are you today"),
            ("hi", "hi nice to meet you"),
            ("hey", "hey what can i help you with"),
            ("good morning", "good morning hope you have a great day"),
            ("good evening", "good evening how was your day"),
            
            # How are you
            ("how are you", "i am doing well thank you for asking"),
            ("how do you feel", "i feel good ready to help you"),
            ("are you okay", "yes i am okay thank you"),
            
            # Questions about identity
            ("who are you", "i am a neural network chatbot using tree structure"),
            ("what are you", "i am an echo state network with tree architecture"),
            ("what is your name", "i am the deep tree echo chatbot"),
            
            # Questions about capability
            ("what can you do", "i can have conversations and learn patterns"),
            ("can you help me", "yes i will try my best to help you"),
            ("do you understand", "i process your words through my neural network"),
            
            # Simple responses
            ("yes", "okay good"),
            ("no", "i understand"),
            ("maybe", "that is possible"),
            ("okay", "great let us continue"),
            
            # Farewells
            ("goodbye", "goodbye have a nice day"),
            ("bye", "bye take care"),
            ("see you", "see you later"),
            ("thank you", "you are welcome"),
            ("thanks", "no problem happy to help"),
            
            # Questions
            ("what is intelligence", "intelligence is the ability to learn and adapt"),
            ("what is learning", "learning is changing based on experience"),
            ("what is a tree", "a tree is a hierarchical structure with branches"),
            ("what is echo", "echo is a reflection that comes back to you"),
            ("what is neural", "neural relates to networks of connected nodes"),
            
            # About the system
            ("tell me about yourself", "i am built with tree structured neural networks"),
            ("how do you work", "i use echo state networks with membrane reservoirs"),
            ("what is your structure", "i have trees membranes and differentials"),
            ("explain your architecture", "my architecture follows the a000081 sequence"),
        ]
        
        self._training_pairs.extend(default_pairs)
    
    def train(self, additional_pairs: Optional[List[Tuple[str, str]]] = None):
        """Train the chatbot on conversation pairs.
        
        Parameters
        ----------
        additional_pairs : List[Tuple[str, str]], optional
            Additional (input, response) pairs to train on
        """
        if additional_pairs:
            self._training_pairs.extend(additional_pairs)
        
        if not self._training_pairs:
            return
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for input_text, response_text in self._training_pairs:
            # Encode input
            input_indices = self.vocab.encode(input_text)
            input_emb = self.vocab.embed_sequence(input_indices)
            
            # Average embedding for input
            if len(input_emb) > 0:
                input_vec = np.mean(input_emb, axis=0)
            else:
                input_vec = np.zeros(self.config.embedding_dim)
            
            # Encode response (as target distribution over vocabulary)
            response_indices = self.vocab.encode(response_text)
            response_dist = np.zeros(self.vocab.size)
            for idx in response_indices:
                response_dist[idx] += 1
            if np.sum(response_dist) > 0:
                response_dist = response_dist / np.sum(response_dist)
            
            X_train.append(input_vec)
            y_train.append(response_dist)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Initialize and train model
        self.model.initialize(X_train, y_train)
        self.model.fit(X_train, y_train, warmup=min(self.config.warmup_steps, len(X_train) // 2))
        
        self._is_trained = True
    
    def _encode_input(self, text: str) -> np.ndarray:
        """Encode input text to embedding vector."""
        indices = self.vocab.encode(text)
        if not indices:
            return np.zeros(self.config.embedding_dim)
        
        embeddings = self.vocab.embed_sequence(indices)
        return np.mean(embeddings, axis=0)
    
    def _sample_token(self, logits: np.ndarray) -> int:
        """Sample token from logits with temperature."""
        # Apply temperature
        logits = logits / max(self.config.temperature, 0.01)
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-k sampling
        top_k_indices = np.argsort(probs)[-self.config.top_k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)
        
        # Sample
        sampled_idx = self.rng.choice(top_k_indices, p=top_k_probs)
        return int(sampled_idx)
    
    def _generate_response(self, input_vec: np.ndarray) -> str:
        """Generate response from input vector."""
        if not self._is_trained:
            self.train()
        
        # Reset model state
        self.model.reset()
        
        # Process input through model
        state = self.model._step(self.model.state, input_vec)
        output = state["out"]
        
        if hasattr(output, 'numpy'):
            output = output.numpy()
        output = np.asarray(output).flatten()
        
        # Generate tokens
        response_tokens = []
        
        # Use output as initial logits
        # Project to vocabulary size if needed
        if len(output) != self.vocab.size:
            # Simple projection: repeat/tile to match vocab size
            if len(output) > self.vocab.size:
                logits = output[:self.vocab.size]
            else:
                repeats = (self.vocab.size // len(output)) + 1
                logits = np.tile(output, repeats)[:self.vocab.size]
        else:
            logits = output
        
        # Generate response tokens
        for _ in range(self.config.max_response_length):
            token_idx = self._sample_token(logits)
            
            # Check for end token
            if token_idx == self.vocab.word_to_idx.get("<END>", -1):
                break
            
            # Skip special tokens in output
            token = self.vocab.idx_to_word.get(token_idx, "")
            if token and not token.startswith('<'):
                response_tokens.append(token)
            
            # Update logits based on sampled token (autoregressive-like)
            token_emb = self.vocab.get_embedding(token_idx)
            state = self.model._step(self.model.state, token_emb)
            output = state["out"]
            
            if hasattr(output, 'numpy'):
                output = output.numpy()
            output = np.asarray(output).flatten()
            
            if len(output) != self.vocab.size:
                if len(output) > self.vocab.size:
                    logits = output[:self.vocab.size]
                else:
                    repeats = (self.vocab.size // len(output)) + 1
                    logits = np.tile(output, repeats)[:self.vocab.size]
            else:
                logits = output
            
            # Early stopping if response is long enough
            if len(response_tokens) >= 10 and token in '.!?':
                break
        
        return ' '.join(response_tokens)
    
    def respond(self, user_input: str) -> str:
        """Generate a response to user input.
        
        Parameters
        ----------
        user_input : str
            The user's message
            
        Returns
        -------
        str
            The chatbot's response
        """
        # Add to history
        self.conversation_history.append(f"User: {user_input}")
        
        # Encode input
        input_vec = self._encode_input(user_input)
        
        # Generate response
        response = self._generate_response(input_vec)
        
        # Clean up response
        response = self._clean_response(response)
        
        # Add to history
        self.conversation_history.append(f"Bot: {response}")
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove duplicate words
        words = response.split()
        cleaned = []
        for i, word in enumerate(words):
            if i == 0 or word != words[i-1]:
                cleaned.append(word)
        
        # Capitalize first letter
        response = ' '.join(cleaned)
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure ends with punctuation
        if response and response[-1] not in '.!?':
            response += '.'
        
        return response
    
    def get_model_info(self) -> str:
        """Get information about the model structure."""
        info = [
            f"DTESNN Chatbot (Order {self.config.base_order})",
            "=" * 50,
            f"Vocabulary size: {self.vocab.size}",
            f"Embedding dimension: {self.config.embedding_dim}",
            f"Units per component: {self.config.units_per_component}",
            f"Temperature: {self.config.temperature}",
            f"Top-k: {self.config.top_k}",
            "",
        ]
        
        if self.model.initialized:
            info.extend([
                "Model Structure:",
                f"  J-Surface ESN dim: {self.model._jsurface_dim}",
                f"  Membrane Reservoir dim: {self.model._reservoir_dim}",
                f"  Combined dim: {self.model._combined_dim}",
                f"  Output dim: {self.model.output_dim}",
                "",
                f"Tree counts: {self.model.synchronizer.params.tree_counts}",
                f"Total trees: {len(self.model.synchronizer._all_trees)}",
            ])
        
        return '\n'.join(info)
    
    def reset(self):
        """Reset conversation history and model state."""
        self.conversation_history = []
        if self.model.initialized:
            self.model.reset()


def compare_orders(
    user_input: str,
    orders: List[int] = [3, 4, 5, 6, 7],
    seed: int = 42,
) -> Dict[int, str]:
    """Compare chatbot responses across different A000081 orders.
    
    Parameters
    ----------
    user_input : str
        Input message to test
    orders : List[int]
        List of orders to compare
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    Dict[int, str]
        Mapping from order to response
    """
    results = {}
    
    for order in orders:
        config = ChatbotConfig(
            base_order=order,
            seed=seed,
        )
        chatbot = DTESNNChatbot(config)
        chatbot.train()
        response = chatbot.respond(user_input)
        results[order] = response
    
    return results


def interactive_chat(base_order: int = 5, seed: int = 42):
    """Run interactive chat session.
    
    Parameters
    ----------
    base_order : int
        A000081 base order for the model
    seed : int
        Random seed
    """
    config = ChatbotConfig(base_order=base_order, seed=seed)
    chatbot = DTESNNChatbot(config)
    
    print("Training chatbot...")
    chatbot.train()
    
    print()
    print(chatbot.get_model_info())
    print()
    print("Chat started! Type 'quit' to exit, 'info' for model info.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'info':
                print(chatbot.get_model_info())
                continue
            
            response = chatbot.respond(user_input)
            print(f"Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Demo: compare responses across orders
    print("=" * 70)
    print("DTESNN Chatbot - Order Comparison Demo")
    print("=" * 70)
    
    test_inputs = [
        "hello",
        "what is intelligence",
        "tell me about yourself",
        "how do you work",
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: '{test_input}'")
        print("-" * 50)
        
        results = compare_orders(test_input, orders=[3, 4, 5, 6], seed=42)
        
        for order, response in results.items():
            print(f"Order {order}: {response}")
