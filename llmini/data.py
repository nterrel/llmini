# data.py
from pathlib import Path
import numpy as np
import torch

BLOCK_SIZE = 128  # Centralized block size configuration


def load_char_data(path="llmini/data/tinyshakespeare.txt", block_size=BLOCK_SIZE, split=0.9, device="cpu"):
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(xs):
        return "".join([itos[x] for x in xs])

    data = np.array(encode(text), dtype=np.int64)
    n = int(len(data) * split)
    train_data, val_data = data[:n], data[n:]

    def get_batch(split="train", batch_size=64):
        source = train_data if split == "train" else val_data
        ix = np.random.randint(
            0, len(source) - block_size - 1, size=(batch_size,))
        x = np.stack([source[i:i + block_size] for i in ix])
        y = np.stack([source[i + 1:i + block_size + 1] for i in ix])
        return (torch.from_numpy(x).to(device),
                torch.from_numpy(y).to(device))

    vocab_size = len(chars)
    return vocab_size, get_batch, decode
