# Test for text generation length output by `sample.py`

def test_generate_length():
    from llmini.model import TinyGPT
    import torch
    model = TinyGPT(vocab_size=65, block_size=128, n_layer=4,
                    n_head=4, n_embd=128, dropout=0.1)
    prompt = torch.tensor([[1, 2, 3]])
    output = model.generate(prompt, max_new_tokens=10)
    assert output.size(1) == 13  # 3 initial tokens + 10 new tokens
