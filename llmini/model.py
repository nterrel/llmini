# model.py
from llmini.arch import TinyGPT, ComplexGPT


def get_model(architecture, vocab_size, block_size, device):
    if architecture == "tiny":
        return TinyGPT(vocab_size, block_size=block_size, n_layer=6, n_head=8, n_embd=256, dropout=0.0).to(device)
    elif architecture == "complex":
        return ComplexGPT(vocab_size, block_size=block_size, n_layer=8, n_head=8, n_embd=512, dropout=0.1).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_from_args(args, vocab_size, block_size, device):
    """
    Select and return the appropriate model based on the parsed arguments.

    Args:
        args: Parsed command-line arguments.
        vocab_size: Size of the vocabulary.
        block_size: Maximum sequence length.
        device: Device to load the model on.

    Returns:
        An instance of the selected model.
    """
    if args.model == "tiny":
        return TinyGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=6,
            n_head=8,
            n_embd=256,
            dropout=0.0
        ).to(device)
    elif args.model == "complex":
        return ComplexGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=8,
            n_head=8,
            n_embd=512,
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
