import torch
import argparse

def inspect_checkpoint(checkpoint_path):
    """
    Inspect the contents of a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("Checkpoint keys:")
    for key in checkpoint.keys():
        print(f"- {key}")

    if "model" in checkpoint:
        print("\nModel state_dict keys:")
        for key in checkpoint["model"].keys():
            print(f"- {key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the contents of a checkpoint file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")

    args = parser.parse_args()
    inspect_checkpoint(args.checkpoint)