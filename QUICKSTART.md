# Quick Start Guide

Get started with KroxAI-Mini in minutes!

## 1. Installation

```bash
pip install kroxai-mini
```

Or install from source:
```bash
git clone https://github.com/KroxenStudios/KroxAI-Mini.git
cd KroxAI-Mini
pip install -e .
```

## 2. First Steps

### Use the Simple Tokenizer

```python
from kroxai_mini import SimpleTokenizer

tk = SimpleTokenizer()
tokens = tk.encode("Hello, AI world!", add_bos=True)
text = tk.decode(tokens)
print(text)  # Output: Hello, AI world!
```

### Build a Simple Model (requires numpy)

```python
from kroxai_mini import SimpleTokenizer, TransformerLM
import numpy as np

# Create tokenizer and model
tk = SimpleTokenizer()
model = TransformerLM(
    vocab_size=tk.vocab_size,
    dim=128,
    n_layers=2,
    n_heads=4,
    ff_hidden=256,
    max_len=128
)

# Generate some text
prompt = "Q: What is machine learning?\nA: "
tokens = tk.encode(prompt, add_bos=True)
x = np.array([tokens], dtype=np.int64)

# Generate continuation
output = model.generate(x, max_new_tokens=50, temperature=0.8)
response = tk.decode(output[0, len(tokens):].tolist())
print(response)
```

## 3. Run Examples

### Interactive Chat

```bash
python examples/chat.py
```

Options:
```bash
python examples/chat.py --temperature 0.8 --top-p 0.9 --preset small
```

### Start the API Server

First install server dependencies:
```bash
pip install kroxai-mini[server]
```

Then run:
```bash
python examples/server.py
```

The server will be available at http://localhost:5000

Test it:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

### Train a Model

Install PyTorch first:
```bash
pip install kroxai-mini[torch]
```

Then train:
```bash
python examples/torch_train.py training_data.json --epochs 10 --batch-size 32
```

## 4. Core Concepts

### Evidence-Augmented Models (EAM)

KroxAI-Mini implements the EAM paradigm:

- **Evidence-based**: Answers are grounded in retrievable evidence
- **Transparent**: Each response can be traced to sources
- **Modular**: Easy to add custom tools and retrieval systems
- **Lightweight**: Runs on modest hardware

### Components

1. **SimpleTokenizer**: Basic byte-level tokenizer (no dependencies)
2. **TransformerLM**: Small transformer model (requires numpy)
3. **KroxAI**: Full EAM system with RAG (requires PyTorch)
4. **Server**: REST API for production use (requires FastAPI)

## 5. Next Steps

- Read the full [INSTALL.md](INSTALL.md) for detailed installation options
- Check out the [README.md](README.md) for more features
- Explore the `examples/` directory for advanced usage
- Review the source code in `src/kroxai_mini/` to understand the implementation

## 6. Common Issues

**Import Error: No module named 'numpy'**
```bash
pip install numpy
```

**Import Error: No module named 'torch'**
```bash
pip install torch
```

**Server won't start**
```bash
pip install kroxai-mini[server]
```

## 7. Development

Want to contribute? Set up a development environment:

```bash
git clone https://github.com/KroxenStudios/KroxAI-Mini.git
cd KroxAI-Mini
pip install -e .[all]
python test_install.py  # Verify installation
```

## 8. Getting Help

- **Issues**: https://github.com/KroxenStudios/KroxAI-Mini/issues
- **Discussions**: https://github.com/KroxenStudios/KroxAI-Mini/discussions
- **Examples**: Check the `examples/` directory

Happy coding with KroxAI-Mini! 🚀
