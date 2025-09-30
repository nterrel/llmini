# Test for data loading in `data.py`

def test_load_char_data():
    from llmini.data import load_char_data
    vocab_size, decode, stoi, itos = load_char_data()
    assert vocab_size == len(stoi) == len(itos)
    assert decode([stoi['a']]) == 'a'
