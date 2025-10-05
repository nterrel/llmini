# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of LLMini with core features including training, text generation, and evaluation.
- Modular design for easy customization.
- Support for Tiny Shakespeare dataset.

### Changed

- Updated `README.md` to clarify installation instructions and remove references to unused dependencies.
- Removed `torchdata` and `torchtext` from `requirements.txt` and `environment.yaml`.

## [1.0.0] - 2025-09-23

### Updates

- Transformer-based architecture with causal self-attention.
- Configurable hyperparameters for layers, heads, and embedding dimensions.
- Sampling techniques including temperature and top-k sampling.
- Pretrained model checkpoint for quick experimentation.
- Early stopping and checkpointing during training.
- Script to split model weights from full checkpoints.

[Unreleased]: https://github.com/nterrel/llmini/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/nterrel/llmini/releases/tag/v1.0.0
