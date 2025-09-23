# sample.py
import torch
from data import load_char_data
from model import TinyGPT

ckpt = torch.load("tinygpt_char.pt", map_location="cpu")
vocab_size = ckpt["config"]["vocab_size"]
block_size = ckpt["config"]["block_size"]

_, _, decode = load_char_data(block_size=block_size)  # just to reuse decode
model = TinyGPT(vocab_size, block_size=block_size)
model.load_state_dict(ckpt["model"])
model.eval()

# prime with a prompt
prompt = "ROMEO:"
ids = torch.tensor([[ord(c) for c in prompt]])  # quick+dirty for ASCII prompts
# better: map through your stoi (left as exercise)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=500)[0].tolist()

print("".join(chr(c) for c in out))