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

1. Clone the repository with submodules:

   ```bash
   git clone --recurse-submodules <repo-url>
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

3. Initialize external dependencies:

   The `setup_external.py` script automatically initializes external dependencies, such as downloading datasets or setting up submodules, during installation.

## Usage

### Training

Train the model on the Tiny Shakespeare dataset:

```bash
python llmini/train.py
```

#### Training Options

- **Model Selection**: Specify the model architecture using the `--model` flag. For example:

  ```bash
  python llmini/train.py --model tiny
  ```

- **Dataset Selection**: Choose the dataset to train on using the `--dataset` flag. Supported datasets include `tinyshakespeare` and `wikitext`. For example:

  ```bash
  python llmini/train.py --dataset wikitext
  ```

- **Checkpoint Path**: Resume training from a specific checkpoint using the `--checkpoint` flag. For example:

  ```bash
  python llmini/train.py --checkpoint checkpoints/tinygpt_char.pt
  ```

These options can be combined to customize the training process. For instance:

```bash
python llmini/train.py --model tiny --dataset wikitext --checkpoint checkpoints/tinygpt_char.pt
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
llmini/
├── arch.py        # Model architectures
├── config.py      # Centralized configuration
├── data.py        # Data loading and preprocessing
├── layers.py      # Reusable building blocks for models
├── model.py       # Model initialization and utilities
├── sample.py      # Text generation script
├── train.py       # Training script
├── utils.py       # Shared utilities
```bash
python scripts/evaluate.py
```

### Configuration

Modify `llmini/config.py` to adjust parameters like `BLOCK_SIZE`, `BATCH_SIZE`, and `LEARNING_RATE`.

## Project Structure

```md
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
