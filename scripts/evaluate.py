# evaluate.py

from llmini.train import estimate_loss
losses = estimate_loss(iters=50)
print(f"Validation Loss: {losses['val']:.3f}")
