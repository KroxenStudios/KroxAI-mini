# Installation Guide for KroxAI-Mini

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation Methods

### 1. Install from PyPI (Recommended)

Once the package is published to PyPI, you can install it with:

```bash
pip install kroxai-mini
```

### 2. Install from Source

For development or to get the latest unreleased features:

```bash
git clone https://github.com/KroxenStudios/KroxAI-Mini.git
cd KroxAI-Mini
pip install -e .
```

The `-e` flag installs the package in "editable" mode, meaning changes to the source code will be immediately reflected without reinstalling.

### 3. Install with Optional Dependencies

KroxAI-Mini has several optional dependency groups:

#### Server Dependencies (FastAPI, uvicorn, BM25)
```bash
pip install kroxai-mini[server]
```

#### PyTorch Support
```bash
pip install kroxai-mini[torch]
```

#### HuggingFace Transformers Support
```bash
pip install kroxai-mini[transformers]
```

#### All Optional Dependencies
```bash
pip install kroxai-mini[all]
```

## Package Structure

After installation, the package provides:

- **Core library**: `kroxai_mini` module with core AI components
  - `SimpleTokenizer`: Lightweight tokenizer
  - `TransformerLM`: Transformer language model
  - `KroxAI`: Main EAM chatbot (requires PyTorch)
  - `data_utils`: Data loading and processing utilities

- **Example Scripts**: Located in the `examples/` directory
  - `chat.py`: Interactive chat interface
  - `server.py`: FastAPI server for REST API access
  - `train.py`: Basic training script
  - `torch_train.py`: PyTorch-based training
  - `dpo_simple.py`: DPO (Direct Preference Optimization) training
  - `eval_harness.py`: Evaluation harness

## Verification

Verify the installation:

```python
import kroxai_mini
print(kroxai_mini.__all__)
# Should print: ['SimpleTokenizer', 'data_utils', 'KroxAI', 'TransformerLM']
```

## Running Examples

After installation, you can run the example scripts:

```bash
# Interactive chat (requires numpy)
python examples/chat.py

# Start the server (requires fastapi, uvicorn)
python examples/server.py

# Train a model (requires numpy)
python examples/train.py data.json
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you've installed the required dependencies:

```bash
# Basic dependencies
pip install numpy

# For PyTorch models
pip install torch

# For server
pip install fastapi uvicorn rank_bm25 requests

# For HuggingFace tokenizers
pip install transformers
```

### Package Not Found

If Python can't find the package, ensure:
1. You've activated the correct virtual environment (if using one)
2. The package is installed: `pip list | grep kroxai`
3. You're running Python from the same environment where you installed the package

## Development Setup

For contributing to KroxAI-Mini:

```bash
# Clone the repository
git clone https://github.com/KroxenStudios/KroxAI-Mini.git
cd KroxAI-Mini

# Install in editable mode with all dependencies
pip install -e .[all]

# Run examples
python examples/chat.py
```

## Uninstallation

To remove KroxAI-Mini:

```bash
pip uninstall kroxai-mini
```
