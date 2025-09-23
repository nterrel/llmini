# LLMini

LLMini is a lightweight implementation of a transformer-based language model inspired by GPT architectures. It is designed for educational purposes and small-scale experiments, making it ideal for understanding the inner workings of language models.

## Features

- Transformer-based architecture with causal self-attention.
- Configurable hyperparameters for layers, heads, and embedding dimensions.
- Sampling techniques including temperature and top-k sampling.
- Pretrained model checkpoint for quick experimentation.
- Modular design for easy customization and extension.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/nterrel/llmini.git
   cd llmini
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train the model on the Tiny Shakespeare dataset:

```bash
python scripts/train_model.py
```

### Text Generation

Generate text using the pretrained model:

```bash
python scripts/generate_text.py
```

### Evaluation

Evaluate the model's performance:

```bash
python scripts/evaluate.py
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite it as follows:

```bibtex
@software{terrel2025llmini,
  author = {Nick Terrel},
  title = {LLMini: A Tiny LLM Implementation},
  year = {2025},
  version = {1.0.0},
  doi = {10.1234/llmini.2025},
  url = {https://github.com/nterrel/llmini}
}
```

## Credits

- Parts of the model architecture were inspired by [minGPT](https://github.com/karpathy/minGPT).
- Dataset: [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
