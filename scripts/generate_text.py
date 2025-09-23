# generate_text.py
from llmini.model import TinyGPT
import torch


model = TinyGPT.load_from_checkpoint("checkpoints/tinygpt_char.pt")
prompt = "ROMEO:"
ids = torch.tensor([[ord(c) for c in prompt]])
output = model.generate(ids, max_new_tokens=500)
print("Generated Text:", "".join([chr(c) for c in output[0].tolist()]))
