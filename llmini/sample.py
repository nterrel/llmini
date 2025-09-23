# sample.py
import torch
from llmini.data import load_char_data
from llmini.model import TinyGPT
import torch.nn as nn

# Load vocabulary and mappings
vocab_size, stoi, decode = load_char_data()

# Initialize the model
model = TinyGPT(vocab_size=83, block_size=128, n_layer=4,
                n_head=4, n_embd=128, dropout=0.1)

# Load the checkpoint
ckpt = torch.load("checkpoints/tinygpt_char.pt", map_location="cpu")
state_dict = ckpt["model"]

# Expand tok_emb weights
if "tok_emb.weight" in state_dict:
    old_weights = state_dict["tok_emb.weight"]
    new_weights = torch.nn.functional.pad(
        old_weights, (0, 0, 0, 18))  # Add 18 rows
    state_dict["tok_emb.weight"] = new_weights

# Expand head weights
if "head.weight" in state_dict:
    old_weights = state_dict["head.weight"]
    new_weights = torch.nn.functional.pad(
        old_weights, (0, 0, 0, 18))  # Add 18 rows
    state_dict["head.weight"] = new_weights

# Load the modified state_dict
model.load_state_dict(state_dict)
model.eval()

# prime with a prompt
prompt = "ROMEO:"
ids = torch.tensor([[stoi[c] for c in prompt]])  # Use stoi for encoding
# better: map through your stoi (left as exercise)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=500)[0].tolist()

print(decode(out))  # Use decode for decoding
