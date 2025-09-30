# train.py
import torch
import math
from tqdm import trange
from llmini.data import CharDataLoader
from llmini.utils import parse_arguments, get_model_from_args
from llmini.config import BLOCK_SIZE, BATCH_SIZE, STEPS, LEARNING_RATE, DEVICE
import os


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


# Initialize data loader
data_loader = CharDataLoader(block_size=BLOCK_SIZE, device=DEVICE)
vocab_size = data_loader.vocab_size


def estimate_loss(iters=25, test_mode=False):
    """
    Estimate the training and validation loss over a number of iterations.

    Args:
        iters (int): Number of iterations to average the loss over.
        test_mode (bool): Whether to use a smaller batch size for testing.

    Returns:
        dict: A dictionary containing the average loss for 'train' and 'val' splits.
    """
    model.eval()
    outs = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            los = 0.0
            for _ in range(iters):
                xb, yb = data_loader.get_batch(
                    split, BATCH_SIZE if not test_mode else 8)
                _, loss = model(xb, yb)
                los += loss.item()
            outs[split] = los / iters
    model.train()
    return outs


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Initialize the model
    model = get_model_from_args(args, vocab_size, BLOCK_SIZE, DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)

    warmup = 200
    min_lr = LEARNING_RATE * 0.1
    patience = 10

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda t: min(
            1.0,
            (t + 1) / warmup
        ) if t < warmup else (
            min_lr / LEARNING_RATE + (1 - min_lr / LEARNING_RATE) * 0.5 * (1 + math.cos(math.pi * (t - warmup) / max(1, STEPS - warmup))))
    )

    checkpoint_path = "checkpoints/tinygpt_char.pt"
    start_step, best_val_loss, no_improve_steps = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler)

    for step in trange(start_step, STEPS):
        xb, yb = data_loader.get_batch("train", BATCH_SIZE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0:
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
