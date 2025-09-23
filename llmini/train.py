# train.py
import torch
import math
from tqdm import trange
from llmini.data import load_char_data
from llmini.model import TinyGPT
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cpu"  # GTX 780 isn't helpful here; CPU is fine
block_size = 128
batch_size = 64
vocab_size, get_batch, decode = load_char_data(
    block_size=block_size, device=device)

model = TinyGPT(vocab_size, block_size=block_size, n_layer=4,
                n_head=4, n_embd=128, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=steps)

steps = 3000  # ~a few minutes on a desktop CPU
eval_every = 200


def estimate_loss(iters=50):
    model.eval()
    outs = {}
    with torch.no_grad():
        for split in ["train", "val"]:
            los = 0.0
            for _ in range(iters):
                xb, yb = get_batch(split, batch_size)
                _, loss = model(xb, yb)
                los += loss.item()
            outs[split] = los / iters
    model.train()
    return outs


for step in trange(steps):
    xb, yb = get_batch("train", batch_size)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if (step + 1) % eval_every == 0:
        losses = estimate_loss(20)
        print(
            f"step {step+1}: train {losses['train']:.3f} | val {losses['val']:.3f}")

torch.save({"model": model.state_dict(),
            "config": {"vocab_size": vocab_size, "block_size": block_size}},
           "checkpoints/tinygpt_char.pt")
print("Saved tinygpt_char.pt")
