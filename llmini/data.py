# data.py
from pathlib import Path
import numpy as np
import torch

BLOCK_SIZE = 128  # Centralized block size configuration


class CharDataLoader:
    def __init__(self, path="data/tinyshakespeare.txt", block_size=BLOCK_SIZE, split=0.9, device="cpu"):
        self.block_size = block_size
        self.device = device
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        data = np.array(self.encode(text), dtype=np.int64)
        n = int(len(data) * split)
        self.train_data, self.val_data = data[:n], data[n:]
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, xs):
        return "".join([self.itos.get(x, "?") for x in xs])

    def get_batch(self, split="train", batch_size=64):
        source = self.train_data if split == "train" else self.val_data
        ix = np.random.randint(
            0, len(source) - self.block_size - 1, size=(batch_size,))
        x = np.stack([source[i:i + self.block_size] for i in ix])
        y = np.stack([source[i + 1:i + self.block_size + 1] for i in ix])
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))
