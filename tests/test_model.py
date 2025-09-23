# Test for model forward pass in `model.py`

def test_model_forward():
    from llmini.model import TinyGPT
    import torch
    model = TinyGPT(vocab_size=65, block_size=128, n_layer=4,
                    n_head=4, n_embd=128, dropout=0.1)
    x = torch.randint(0, 65, (2, 128))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 128, 65)
