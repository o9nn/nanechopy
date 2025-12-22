#!/usr/bin/env python3
"""
Generate vocabulary from DTESNN training dataset.

This script extracts all unique tokens from the training data and creates
a vocabulary file that can be used for training DTESNN chatbot models.

Usage:
    python scripts/generate_vocab.py [--input data/training_dataset.jsonl] [--output data/vocab.json]
"""

import json
import argparse
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str) -> list:
    """
    Simple tokenizer that splits on whitespace and punctuation.
    Returns lowercase tokens.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Split on whitespace and keep punctuation as separate tokens
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    
    return tokens


def extract_text_from_messages(messages: list) -> str:
    """Extract all text content from a messages list."""
    texts = []
    for msg in messages:
        content = msg.get('content', '')
        if content:
            texts.append(content)
    return ' '.join(texts)


def generate_vocab_from_jsonl(input_path: str, min_freq: int = 1) -> dict:
    """
    Generate vocabulary from JSONL training file.
    
    Args:
        input_path: Path to the JSONL training file
        min_freq: Minimum frequency for a token to be included
        
    Returns:
        Dictionary mapping tokens to indices
    """
    token_counts = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                text = extract_text_from_messages(messages)
                tokens = tokenize(text)
                token_counts.update(tokens)
            except json.JSONDecodeError:
                continue
    
    # Filter by minimum frequency and sort by frequency (descending)
    filtered_tokens = [
        token for token, count in token_counts.most_common()
        if count >= min_freq
    ]
    
    # Create vocabulary with special tokens
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
    }
    
    # Add tokens starting from index 4
    for i, token in enumerate(filtered_tokens, start=4):
        vocab[token] = i
    
    return vocab, token_counts


def main():
    parser = argparse.ArgumentParser(description='Generate vocabulary from DTESNN training dataset')
    parser.add_argument('--input', '-i', type=str, default='data/training_dataset.jsonl',
                        help='Input JSONL training file')
    parser.add_argument('--output', '-o', type=str, default='data/vocab.json',
                        help='Output vocabulary JSON file')
    parser.add_argument('--stats', '-s', type=str, default='data/vocab_stats.json',
                        help='Output vocabulary statistics JSON file')
    parser.add_argument('--min-freq', type=int, default=1,
                        help='Minimum frequency for token inclusion')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Generate vocabulary
    print(f"Generating vocabulary from: {args.input}")
    vocab, token_counts = generate_vocab_from_jsonl(args.input, args.min_freq)
    
    # Save vocabulary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f"Vocabulary saved to: {args.output}")
    print(f"Total tokens: {len(vocab)}")
    
    # Save statistics
    stats = {
        'total_tokens': len(vocab),
        'special_tokens': 4,
        'unique_words': len(vocab) - 4,
        'total_occurrences': sum(token_counts.values()),
        'min_frequency': args.min_freq,
        'top_50_tokens': dict(token_counts.most_common(50)),
    }
    
    stats_path = Path(args.stats)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to: {args.stats}")
    print(f"\nTop 20 tokens:")
    for token, count in token_counts.most_common(20):
        print(f"  {token}: {count}")
    
    return 0


if __name__ == '__main__':
    exit(main())
