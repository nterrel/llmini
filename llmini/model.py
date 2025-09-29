# model.py
from llmini.arch import TinyGPT, ComplexGPT


def get_model(architecture, vocab_size, block_size, device):
    """
    Select and return the appropriate model based on the architecture.

    Args:
        architecture: The model architecture (e.g., "tiny", "complex").
        vocab_size: Size of the vocabulary.
        block_size: Maximum sequence length.
        device: Device to load the model on.

    Returns:
        An instance of the selected model.
    """
    if architecture == "tiny":
        return TinyGPT(vocab_size, block_size=block_size, n_layer=6, n_head=8, n_embd=256, dropout=0.0).to(device)
    elif architecture == "complex":
        return ComplexGPT(vocab_size, block_size=block_size, n_layer=8, n_head=8, n_embd=512, dropout=0.1).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_model_from_args(args, vocab_size, block_size, device):
    """
    Wrapper function to select and return the appropriate model based on parsed arguments.
    Prompts the user to select a model if not provided in args.

    Args:
        args: Parsed command-line arguments.
        vocab_size: Size of the vocabulary.
        block_size: Maximum sequence length.
        device: Device to load the model on.

    Returns:
        An instance of the selected model.
    """
    if not hasattr(args, 'model') or not args.model:
        args.model = input(
            "Please specify the model architecture (tiny/complex): ").strip().lower()

    return get_model(args.model, vocab_size, block_size, device)
