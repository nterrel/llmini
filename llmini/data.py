# data.py
from pathlib import Path
import numpy as np
import torch

# Optional tokenizers dependency: provide graceful fallback for environments
# (like minimal CI) where it isn't installed.
try:  # pragma: no cover - import guard
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    _HAS_TOKENIZERS = True
except Exception:  # broad to catch binary wheels issues
    _HAS_TOKENIZERS = False
    Tokenizer = BPE = BpeTrainer = Whitespace = None  # type: ignore

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
        if not _HAS_TOKENIZERS:
            # Fallback: simple character-level vocab if tokenizers missing
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.vocab_size = len(self.stoi)
            data = np.array([self.stoi[c] for c in text], dtype=np.int64)
        elif _HAS_TOKENIZERS and all(sym is not None for sym in (Tokenizer, BPE, BpeTrainer, Whitespace)):
            # Initialize and train a tiny BPE tokenizer per file
            self.tokenizer = Tokenizer(BPE())  # type: ignore[arg-type]
            self.tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]
            trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])  # type: ignore[operator]
            self.tokenizer.train_from_iterator([text], trainer)
            self.stoi = self.tokenizer.get_vocab()
            self.itos = {idx: token for token, idx in self.stoi.items()}
            self.vocab_size = len(self.stoi)
            data = np.array(self.tokenizer.encode(text).ids, dtype=np.int64)
        else:
            raise RuntimeError("Inconsistent tokenizers state; install 'tokenizers' or ensure optional deps are available.")

        # Split the data into training and validation sets
        n = int(len(data) * split)
        self.train_data, self.val_data = data[:n], data[n:]

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


class WikiTextDataLoader:
    """
    A class for loading and batching tokenized WikiText data for training and evaluation.

    Attributes:
        block_size (int): Maximum sequence length for batching.
        device (str): Device to load the data onto (e.g., 'cpu', 'cuda').
        train_data (list): Numericalized training data.
        val_data (list): Numericalized validation data.
        vocab_size (int): Size of the vocabulary.
        stoi (dict): String-to-index mapping for tokens.
        itos (dict): Index-to-string mapping for tokens.
    """

    def __init__(self, path="external/wikitext/wikitext-2-v1/train-00000-of-00001.parquet", block_size=BLOCK_SIZE, split=0.9, device="cpu"):
        """
        Initialize the WikiTextDataLoader.

        Args:
            path (str): Path to the .parquet file containing the dataset.
            block_size (int): Maximum sequence length for batching.
            split (float): Fraction of data to use for training (remainder is for validation).
            device (str): Device to load the data onto (e.g., 'cpu', 'cuda').
        """
        self.block_size = block_size
        self.device = device

        # Load the dataset lazily to avoid importing pyarrow unless needed
        try:
            from llmini.scripts.load_wikitext import WikiTextDataset  # local import
        except ModuleNotFoundError as e:  # pragma: no cover - environment specific
            raise ImportError(
                "WikiTextDataset requires optional dependency 'pyarrow'. Install with 'pip install pyarrow' or skip using the wikitext dataset in tests."  # noqa: E501
            ) from e
        dataset = WikiTextDataset(path)
        # Tokenize the dataset
        tokens = [token for text in dataset for token in text.split()]

        # Build vocabulary
        unique_tokens = sorted(set(tokens))
        self.stoi = {token: idx for idx, token in enumerate(unique_tokens)}
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        # Numericalize the dataset
        data = [self.stoi[token] for token in tokens]

        # Split the data into training and validation sets
        n = int(len(data) * split)
        self.train_data, self.val_data = data[:n], data[n:]

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
        return (torch.tensor(x, dtype=torch.long).to(self.device),
                torch.tensor(y, dtype=torch.long).to(self.device))


def load_char_data(path="data/tinyshakespeare.txt", block_size=BLOCK_SIZE, split=0.9, device="cpu"):
    """
    Load character-level data and return essential components.

    Args:
        path (str): Path to the text file containing the dataset.
        block_size (int): Maximum sequence length for batching.
        split (float): Fraction of data to use for training (remainder is for validation).
        device (str): Device to load the data onto (e.g., 'cpu', 'cuda').

    Returns:
        tuple: A tuple containing (vocab_size, decode, stoi, itos).
    """
    data_loader = CharDataLoader(
        path=path, block_size=block_size, split=split, device=device)
    return data_loader.vocab_size, data_loader.decode, data_loader.stoi, data_loader.itos


def get_batch(split="train", batch_size=64, path="data/tinyshakespeare.txt", block_size=BLOCK_SIZE, device="cpu"):
    """
    Generate a batch of input and target sequences using CharDataLoader.

    Args:
        split (str): Data split to use ('train' or 'val').
        batch_size (int): Number of sequences in the batch.
        path (str): Path to the text file containing the dataset.
        block_size (int): Maximum sequence length for batching.
        device (str): Device to load the data onto (e.g., 'cpu', 'cuda').

    Returns:
        tuple: A tuple (x, y) where x is the input tensor and y is the target tensor.
    """
    data_loader = CharDataLoader(
        path=path, block_size=block_size, split=0.9, device=device)  # Ensure split is a float
    return data_loader.get_batch(split=split, batch_size=batch_size)
