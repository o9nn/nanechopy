# DTESNN Training Data

This directory contains training data and vocabulary files for the DTESNN chatbot.

## Files

### `training_dataset.jsonl`

The main training dataset in JSONL format. Each line contains a JSON object with the following structure:

```json
{
  "messages": [
    {"role": "user", "content": "User message..."},
    {"role": "assistant", "content": "Assistant response..."}
  ]
}
```

### `vocab.json`

Generated vocabulary mapping tokens to indices. Includes special tokens:
- `<pad>` (0): Padding token
- `<unk>` (1): Unknown token
- `<sos>` (2): Start of sequence
- `<eos>` (3): End of sequence

### `vocab_stats.json`

Statistics about the vocabulary including:
- Total token count
- Token frequency distribution
- Top 50 most common tokens

## Generating Vocabulary

To regenerate the vocabulary from the training dataset:

```bash
python scripts/generate_vocab.py
```

Options:
- `--input`: Input JSONL file (default: `data/training_dataset.jsonl`)
- `--output`: Output vocabulary JSON file (default: `data/vocab.json`)
- `--stats`: Output statistics JSON file (default: `data/vocab_stats.json`)
- `--min-freq`: Minimum token frequency for inclusion (default: 1)

## Adding Training Data

To add new training data:

1. Add new conversation pairs to `training_dataset.jsonl` in the JSONL format
2. Regenerate the vocabulary: `python scripts/generate_vocab.py`
3. Train a new model: `python scripts/train_and_save.py`

Or use the GitHub Actions workflow to train directly from the repository.

## Dataset Statistics

- **Conversations**: 256 training pairs
- **Vocabulary Size**: ~10,000 unique tokens
- **Domain**: Deep Tree Echo system interactions, reservoir computing, neural networks
