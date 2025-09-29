# data.py
from pathlib import Path
import numpy as np
import torch

BLOCK_SIZE = 128  # Centralized block size configuration


class CharDataLoader:
    """
    A class for loading and batching character-level data for training and evaluation.

    Attributes:
        block_size (int): Maximum sequence length for batching.
        device (str): Device to load the data onto (e.g., 'cpu', 'cuda').
        stoi (dict): Mapping from characters to integer indices.
        itos (dict): Mapping from integer indices to characters.
        train_data (np.ndarray): Training data as a numpy array of integer indices.
        val_data (np.ndarray): Validation data as a numpy array of integer indices.
        vocab_size (int): Size of the vocabulary.
    """

    def __init__(self, path="data/tinyshakespeare.txt", block_size=BLOCK_SIZE, split=0.9, device="cpu"):
        """
        Initialize the CharDataLoader.

        Args:
            path (str): Path to the text file containing the dataset.
            block_size (int): Maximum sequence length for batching.
            split (float): Fraction of data to use for training (remainder is for validation).
            device (str): Device to load the data onto (e.g., 'cpu', 'cuda').
        """
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
        """
        Encode a string into a list of integer indices.

        Args:
            s (str): Input string to encode.

        Returns:
            list: List of integer indices corresponding to the input string.
        """
        return [self.stoi[c] for c in s]

    def decode(self, xs):
        """
        Decode a list of integer indices into a string.

        Args:
            xs (list): List of integer indices to decode.

        Returns:
            str: Decoded string.
        """
        return "".join([self.itos.get(x, "?") for x in xs])

    def get_batch(self, split="train", batch_size=64):
        """
        Generate a batch of input and target sequences.

        Args:
            split (str): Data split to use ('train' or 'val').
            batch_size (int): Number of sequences in the batch.

        Returns:
            tuple: A tuple (x, y) where x is the input tensor and y is the target tensor.
        """
        source = self.train_data if split == "train" else self.val_data
        ix = np.random.randint(
            0, len(source) - self.block_size - 1, size=(batch_size,))
        x = np.stack([source[i:i + self.block_size] for i in ix])
        y = np.stack([source[i + 1:i + self.block_size + 1] for i in ix])
        return (torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device))
