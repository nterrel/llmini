import argparse
import logging
from llmini.model import get_model


def setup_logger(name, level=logging.INFO):
    """
    Set up a logger with the specified name and logging level.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def parse_arguments():
    """
    Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tiny", "complex"], default="tiny",
                        help="Choose the model architecture: 'tiny' or 'complex'")
    parser.add_argument("--dataset", choices=["tinyshakespeare", "wikitext"], default="tinyshakespeare",
                        help="Choose the dataset: 'tinyshakespeare' or 'wikitext'")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tinygpt_char.pt",
                        help="Path to save the model checkpoint.")
    return parser.parse_args()


def get_model_from_args(args, vocab_size, block_size, device):
    """
    Wrapper function to select and return the appropriate model based on parsed arguments.
    Prompts the user to select a model if not provided in args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        device (str): Device to load the model on (e.g., 'cpu', 'cuda').

    Returns:
        nn.Module: An instance of the selected model.

    Raises:
        ValueError: If the user provides an unrecognized model type.
    """
    if not hasattr(args, 'model') or not args.model:
        args.model = input(
            "Please specify the model architecture (tiny/complex): ").strip().lower()

    return get_model(args.model, vocab_size, block_size, device)
