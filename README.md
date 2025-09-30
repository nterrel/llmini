# LLMini

LLMini is a lightweight implementation of a transformer-based language model inspired by GPT architectures. It is designed for educational purposes and small-scale experiments, making it ideal for understanding the inner workings of language models.

## Features

- Transformer-based architecture with causal self-attention.
- Configurable hyperparameters for layers, heads, and embedding dimensions.
- Sampling techniques including temperature and top-k sampling.
- Pretrained model checkpoint for quick experimentation.
- Modular design for easy customization and extension.
- Early stopping and checkpointing during training.
- Script to split model weights from full checkpoints.

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

   Alternatively, use the Conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate llmini
   ```

## Usage

### Training

Train the model on the Tiny Shakespeare dataset:

```bash
python llmini/train.py
```

To train a specific model, use the `--model` flag:

```bash
python llmini/train.py --model tiny
```

### Text Generation

Generate text using the pretrained model:

```bash
python llmini/sample.py
```

To enable debugging logs:

```bash
python llmini/sample.py --debug
```

To generate text with a specific model:

```bash
python llmini/sample.py --model tiny
```

### Splitting Checkpoints

Split the model weights from a full checkpoint:

```bash
python llmini/scripts/split_pt.py --full-checkpoint checkpoints/tinygpt_full.pt --output checkpoints/tinygpt_char_small.pt
```

### Evaluation

Evaluate the model's performance:

```bash
python scripts/evaluate.py
```

### Configuration

Modify `llmini/config.py` to adjust parameters like `BLOCK_SIZE`, `BATCH_SIZE`, and `LEARNING_RATE`.

## Project Structure

```
llmini/
├── arch.py        # Model architectures
├── config.py      # Centralized configuration
├── data.py        # Data loading and preprocessing
├── layers.py      # Reusable building blocks for models
├── model.py       # Model initialization and utilities
├── sample.py      # Text generation script
├── train.py       # Training script
├── utils.py       # Shared utilities
```

## Testing

Run the test suite using:

```bash
pytest tests/
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
