import torch
import argparse


def split_checkpoint(full_checkpoint_path, output_path):
    """
    Splits the model weights from a full checkpoint and saves them to a new file.

    Args:
        full_checkpoint_path (str): Path to the full checkpoint file.
        output_path (str): Path to save the split model weights.
    """
    print(f"Loading full checkpoint from {full_checkpoint_path}...")
    checkpoint = torch.load(full_checkpoint_path, map_location="cpu")

    if "model" not in checkpoint:
        raise ValueError("The checkpoint does not contain model weights under the 'model' key.")

    model_weights = checkpoint["model"]
    print(f"Saving model weights to {output_path}...")
    torch.save(model_weights, output_path)
    print("Split checkpoint saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split model weights from a full checkpoint.")
    parser.add_argument("--full-checkpoint", type=str, required=True, help="Path to the full checkpoint file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the split model weights.")

    args = parser.parse_args()
    split_checkpoint(args.full_checkpoint, args.output)
    print("Split checkpoint saved successfully.")

# Example usage:
# python /Users/nickterrel/llmini/scripts/split_pt.py --full-checkpoint /Users/nickterrel/llmini/checkpoints/tinygpt_full.pt --output /Users/nickterrel/llmini/checkpoints/tinygpt_char_small.pt
