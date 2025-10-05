import torch
from llmini.utils import get_model_from_args
from llmini.config import BLOCK_SIZE, DEVICE
import argparse
import re


def load_model(checkpoint_path, vocab_size):
    """
    Load the model from the specified checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        vocab_size (int): Size of the vocabulary.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = get_model_from_args(args, vocab_size, BLOCK_SIZE, DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def generate_text(model, stoi, itos, prompt, max_length=100):
    """
    Generate text using the model.

    Args:
        model (torch.nn.Module): The trained model.
        stoi (dict): String-to-index mapping.
        itos (dict): Index-to-string mapping.
        prompt (str): The initial text prompt.
        max_length (int): Maximum length of the generated text.

    Returns:
        str: The generated text.
    """
    # Ensure the model does not print training data during inference
    if hasattr(model, 'debug'):
        model.debug = False

    input_ids = torch.tensor([stoi.get(char, stoi['<unk>'])
                             for char in prompt], dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated = input_ids.tolist()[0]

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            input_ids = torch.tensor(
                [generated[-BLOCK_SIZE:]], dtype=torch.long).to(DEVICE)

    return ''.join([itos.get(idx, '<unk>') for idx in generated])


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate text using ComplexGPT.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument("--prompt", type=str, help="Initial text prompt.")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of the generated text.")
    args = parser.parse_args()

    # Prompt for missing arguments
    if not args.checkpoint:
        args.checkpoint = input(
            "Please enter the path to the model checkpoint: ")
    if not args.prompt:
        args.prompt = input("Please enter the initial text prompt: ")

    return args


def infer_model_architecture(checkpoint_path):
    """
    Infer the model architecture from the checkpoint file name.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        str: The inferred model architecture ('tiny' or 'complex').
    """
    if re.search(r"tiny", checkpoint_path, re.IGNORECASE):
        return "tiny"
    elif re.search(r"complex", checkpoint_path, re.IGNORECASE):
        return "complex"
    return None


if __name__ == "__main__":
    args = get_args()

    # Infer the model architecture from the checkpoint file name
    inferred_arch = infer_model_architecture(args.checkpoint)
    if not inferred_arch:
        inferred_arch = input(
            "Please specify the model architecture (tiny/complex): ")

    print(f"Using model architecture: {inferred_arch}")

    # Load the data loader to get vocabulary mappings
    from llmini.data import WikiTextDataLoader
    data_loader = WikiTextDataLoader(block_size=BLOCK_SIZE, device=DEVICE)
    stoi, itos = data_loader.stoi, data_loader.itos

    # Load the model
    model = load_model(args.checkpoint, data_loader.vocab_size)

    # Generate text
    generated_text = generate_text(
        model, stoi, itos, args.prompt, args.max_length)
    print("Generated Text:")
    print(generated_text)
