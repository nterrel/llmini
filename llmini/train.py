# train.py
import torch
import math
from tqdm import trange
from llmini.utils import parse_arguments, get_model_from_args
from llmini.config import BLOCK_SIZE, BATCH_SIZE, STEPS, LEARNING_RATE, DEVICE
import os
from llmini.data import CharDataLoader
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load the state into.

    Returns:
        tuple: A tuple (start_step, best_val_loss, no_improve_steps) containing the training state.
    """
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        no_improve_steps = checkpoint.get("no_improve_steps", 0)

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        print(f"Resuming training from step {start_step}")
        return start_step, best_val_loss, no_improve_steps
    return 0, float('inf'), 0


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, best_val_loss, no_improve_steps):
    """
    Save the current training state to a checkpoint file.

    Args:
        checkpoint_path (str): Path to save the checkpoint file.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        scheduler (torch.optim.lr_scheduler): The scheduler to save.
        step (int): The current training step.
        best_val_loss (float): The best validation loss achieved so far.
        no_improve_steps (int): The number of steps without improvement in validation loss.
    """
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_val_loss": best_val_loss,
                "no_improve_steps": no_improve_steps},
               checkpoint_path)
    print(f"Saved checkpoint at step {step}")


def estimate_loss(iters=25, test_mode=False, mdl=None, loader=None, batch_size=None):
    """Estimate the training and validation loss.

    This function is imported directly in tests before ``__main__`` executes, so
    the global ``model`` created in the training run may not yet exist. To make
    it test-friendly we allow passing an explicit ``mdl`` and ``loader``; if they
    are not provided we fall back to globals, and if those are missing we lazily
    construct a minimal tiny model + char loader.

    Args:
        iters (int): Number of mini-batches to average.
        test_mode (bool): Use a smaller batch size to keep tests fast.
        mdl (nn.Module, optional): Model to evaluate.
        loader (object, optional): Data loader exposing ``get_batch(split, bs)``.
        batch_size (int, optional): Override batch size (defaults to config value).

    Returns:
        dict: Average loss for 'train' and 'val'.
    """
    # Resolve model
    global data_loader  # ensure we can reuse/augment the existing global
    if mdl is None:
        if 'model' in globals():  # pragma: no cover - simple branch
            mdl = globals()['model']
        else:
            # Lazy import to avoid circulars
            from llmini.model import get_model
            # Build a minimal tiny model for evaluation context
            vocab_size_local = getattr(data_loader, 'vocab_size', 128)
            mdl = get_model('tiny', vocab_size_local, BLOCK_SIZE, DEVICE)
    if loader is None:
        if 'data_loader' in globals() and data_loader is not None:
            loader = data_loader
        else:
            from llmini.data import CharDataLoader
            loader = CharDataLoader(block_size=BLOCK_SIZE, device=DEVICE)
            data_loader = loader

    bs = batch_size or BATCH_SIZE
    eff_bs = bs if not test_mode else min(8, bs)

    mdl_was_training = mdl.training
    mdl.eval()
    outs = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            los = 0.0
            for _ in range(iters):
                xb, yb = loader.get_batch(split, eff_bs)
                _, loss = mdl(xb, yb)
                los += loss.item()
            outs[split] = los / max(1, iters)
    if mdl_was_training:
        mdl.train()
    return outs


# Initialize data loader
data_loader = CharDataLoader(block_size=BLOCK_SIZE, device=DEVICE)
vocab_size = data_loader.vocab_size


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Initialize the appropriate data loader based on the dataset
    if args.dataset == "wikitext":
        from llmini.data import WikiTextDataLoader
        data_loader = WikiTextDataLoader(
            path="external/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet",
            block_size=BLOCK_SIZE, device=DEVICE)
    else:
        from llmini.data import CharDataLoader
        data_loader = CharDataLoader(block_size=BLOCK_SIZE, device=DEVICE)

    vocab_size = data_loader.vocab_size

    # Initialize the model
    model = get_model_from_args(args, vocab_size, BLOCK_SIZE, DEVICE)

    # Apply Xavier initialization
    for param in model.parameters():
        if param.dim() > 1:
            xavier_uniform_(param)

    # Integrate dropout into the model
    # Ensure the model definition includes dropout layers
    # Example: Add dropout to the model architecture
    model.dropout = Dropout(p=0.1)  # Add dropout to the model instance

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STEPS, eta_min=LEARNING_RATE * 0.1)

    # Early stopping parameters
    patience = 10
    no_improve_steps = 0
    best_val_loss = float('inf')

    # Use the --checkpoint argument for the checkpoint path
    start_step, best_val_loss, no_improve_steps = load_checkpoint(
        args.checkpoint, model, optimizer, scheduler)

    # Dynamic validation frequency based on dataset size
    validation_frequency = max(
        100, len(data_loader.train_data) // (10 * BATCH_SIZE))

    # Ensure 'step' is properly referenced in the checkpoint path logic
    for step in trange(start_step, STEPS):
        xb, yb = data_loader.get_batch("train", BATCH_SIZE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update checkpoint path dynamically
        checkpoint_path = f"checkpoints/complex/complex_model_step_{step + 1}.pt"

        if (step + 1) % validation_frequency == 0:
            losses = estimate_loss(25)
            print(
                f"step {step+1}: train {losses['train']:.3f} | val {losses['val']:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improve_steps = 0
                save_checkpoint(checkpoint_path, model, optimizer,
                                scheduler, step + 1, best_val_loss, no_improve_steps)
            else:
                no_improve_steps += 1

            if no_improve_steps >= patience:
                print("Early stopping triggered. Training terminated.")
                break
