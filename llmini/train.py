# train.py
import torch
import math
from tqdm import trange
from llmini.data import load_char_data
from llmini.model import TinyGPT

device = "cpu"
block_size = 256  # was 128
batch_size = 64
vocab_size, get_batch, decode, _, _ = load_char_data(
    block_size=block_size, device=device)

steps = 20000  # was 3000; ~6–7x more learning
lr = 3e-4  # keep for now

model = TinyGPT(
    vocab_size,
    block_size=block_size,
    n_layer=6,  # was 4
    n_head=8,  # was 4
    n_embd=256,  # was 128
    dropout=0.0  # tiny datasets do better with little/no dropout
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

warmup = 200
min_lr = lr * 0.1
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


def estimate_loss(iters=50, test_mode=False):
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
    for step in trange(steps):
        xb, yb = get_batch("train", batch_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 1000 == 0:
            losses = estimate_loss(50)
            print(
                f"step {step+1}: train {losses['train']:.3f} | val {losses['val']:.3f} | lr {scheduler.get_last_lr()[0]:.2e}")

    torch.save({"model": model.state_dict(),
                "config": {"vocab_size": vocab_size, "block_size": block_size}},
               "checkpoints/tinygpt_char.pt")
    print("Saved tinygpt_char.pt")
