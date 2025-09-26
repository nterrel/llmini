# train.py
import torch
import math
from tqdm import trange
from llmini.data import load_char_data
from llmini.model import TinyGPT
import os

device = "cpu"
block_size = 256  # was 128
batch_size = 64
vocab_size, get_batch, decode, _, _ = load_char_data(
    block_size=block_size, device=device)

steps = 5000  # Reduced from 20000 to speed up training
lr = 3e-4  # keep for now

model = TinyGPT(
    vocab_size,
    block_size=block_size,
    n_layer=6,  # Keeping model complexity
    n_head=8,
    n_embd=256,
    dropout=0.0  # tiny datasets do better with little/no dropout
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

warmup = 200
min_lr = lr * 0.1
patience = 10  # Define patience for early stopping

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda t: min(
        1.0,
        (t + 1) / warmup
    ) if t < warmup else (
        min_lr / lr + (1 - min_lr / lr) * 0.5 * (1 +
                                                 math.cos(math.pi * (t - warmup) / max(1, steps - warmup)))
    )
)

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


def estimate_loss(iters=25, test_mode=False):  # Reduced iterations for faster evaluation
    model.eval()
    outs = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            los = 0.0
            for _ in range(iters):
                # Use smaller batch size in test mode
                xb, yb = get_batch(split, batch_size if not test_mode else 8)
                _, loss = model(xb, yb)
                los += loss.item()
            outs[split] = los / iters
    model.train()
    return outs


if __name__ == "__main__":
    # Check if a checkpoint exists
    checkpoint_path = "checkpoints/tinygpt_char.pt"
    start_step = 0
    best_val_loss = float('inf')
    no_improve_steps = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        no_improve_steps = checkpoint.get("no_improve_steps", 0)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print(
                "Warning: Optimizer state not found in checkpoint. Reinitializing optimizer.")

        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            print(
                "Warning: Scheduler state not found in checkpoint. Reinitializing scheduler.")

        print(f"Resuming training from step {start_step}")

    for step in trange(start_step, steps):
        xb, yb = get_batch("train", batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 500 == 0:  # More frequent loss estimation
            # Reduced iterations for faster feedback
            losses = estimate_loss(25)
            print(
                f"step {step+1}: train {losses['train']:.3f} | val {losses['val']:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

            # Early stopping check
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improve_steps = 0
                # Save the best model
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "step": step + 1,
                            "best_val_loss": best_val_loss,
                            "no_improve_steps": no_improve_steps,
                            "config": {"vocab_size": vocab_size, "block_size": block_size}},
                           checkpoint_path)
                print("Saved tinygpt_char.pt (best model)")
            else:
                no_improve_steps += 1

            if no_improve_steps >= patience:
                print("Early stopping triggered. Training terminated.")
                break

    if no_improve_steps < patience:
        # Save the final model if early stopping was not triggered
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": steps,
                    "best_val_loss": best_val_loss,
                    "no_improve_steps": no_improve_steps,
                    "config": {"vocab_size": vocab_size, "block_size": block_size}},
                   checkpoint_path)
        print("Saved tinygpt_char.pt (final model)")
