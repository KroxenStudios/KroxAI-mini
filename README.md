# KroxAI-mini
**KroxAI mini** is an open source AI project. Released on 1st November 2025.

**KroxAI mini**
* Untrained simple EAM (Evidence‑Augmented Model) Chatbot with RAG features (e.g for implementing live databases)
* Trained (small) version available too.
* Alternative to LocalAI/HF Transformers and AI APIs (e.g Claude, OpenAI, ...).


**Features** (may change as develoment proceeds)
* Adjustable server (add different tools for output generation)
* RAG
* Simple Q/A answering.
* Two tokenizers available: HuggingFace(if installed), Simple (work in progress, fallback if no HF tokenizer is installed)


# Introducing EAM

**Evidence‑Augmented Models (EAMs)** are a new class of AI systems that go beyond standard Large Language Models (LLMs).  
Instead of generating answers purely from statistical patterns, an EAM is designed to:

- **Ground every answer in evidence** (citations, database fragments, or retrieved documents).  
- **Provide transparency** through audit logs, coverage scores, and conflict detection.  
- **Act as an agent** by planning and executing tool‑based actions (e.g. retrieval, parsing, calculation).  
- **Stay lightweight and reproducible**, so it can run locally on modest hardware.  

Formally, while an LLM maps a query *Q* to an answer *A* an EAM maps a query *Q* and evidence *E* to both an answer *A* and a step‑by‑step protocol *S*.

_In short:_  
> **EAM = “Answers with evidence.”**  

KroxAI mini is the first open-source prototype of this approach — a compact EAM that demonstrates how evidence‑grounded reasoning can be combined with modular tool use.


## Installation

### From PyPI (once published)
```bash
pip install kroxai-mini
```

### From source
```bash
git clone https://github.com/KroxenStudios/KroxAI-Mini.git
cd KroxAI-Mini
pip install -e .
```

### Optional dependencies

Install with server support (FastAPI, uvicorn, BM25):
```bash
pip install kroxai-mini[server]
```

Install with PyTorch support:
```bash
pip install kroxai-mini[torch]
```

Install with HuggingFace Transformers support:
```bash
pip install kroxai-mini[transformers]
```

Install all optional dependencies:
```bash
pip install kroxai-mini[all]
```

## Usage

### Basic usage - Tokenizer (no dependencies)
```python
from kroxai_mini import SimpleTokenizer

# Initialize tokenizer
tk = SimpleTokenizer()

# Encode text
text = "Hello, world!"
tokens = tk.encode(text, add_bos=True, add_eos=True)
print(f"Tokens: {tokens}")

# Decode tokens
decoded = tk.decode(tokens)
print(f"Decoded: {decoded}")
```

### Advanced usage - Full model (requires numpy)
```python
from kroxai_mini import SimpleTokenizer, TransformerLM
import numpy as np

# Initialize tokenizer and model
tk = SimpleTokenizer()
model = TransformerLM(
    vocab_size=tk.vocab_size, 
    dim=128, 
    n_layers=2, 
    n_heads=4, 
    ff_hidden=256, 
    max_len=128
)

# Generate text
prompt = "Q: What is AI?\nA: "
ids = tk.encode(prompt, add_bos=True)
x = np.array([ids], dtype=np.int64)
y = model.generate(x, max_new_tokens=64)
response = tk.decode(y[0, len(ids):].tolist())
print(response)
```

### Running the server
```bash
python examples/server.py
```

### Training a model
```bash
python examples/torch_train.py data.json
```

### Interactive chat
```bash
python examples/chat.py
```

