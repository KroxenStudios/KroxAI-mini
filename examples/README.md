# KroxAI-Mini Examples

This directory contains example scripts demonstrating various features of KroxAI-Mini.

## Prerequisites

First, install KroxAI-Mini:
```bash
pip install kroxai-mini
```

Or for local development:
```bash
cd ..
pip install -e .
```

## Examples

### 1. Interactive Chat (`chat.py`)

A simple command-line chat interface using the numpy-based transformer model.

**Requirements**: numpy

```bash
python chat.py
```

**Options**:
- `--temperature`: Sampling temperature (default: 1.0)
- `--top-k`: Top-k sampling (default: None)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--preset`: Model size preset (tiny/small/base/large)

**Example**:
```bash
python chat.py --temperature 0.8 --preset small
```

### 2. API Server (`server.py`)

FastAPI-based REST API server with RAG capabilities.

**Requirements**: Install server extras
```bash
pip install kroxai-mini[server]
```

**Run**:
```bash
python server.py
```

The server will be available at http://localhost:5000

**Endpoints**:
- `GET /` - Web UI
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
- `POST /rag/add` - Add documents to RAG store
- `GET /rag/list` - List RAG documents
- `DELETE /rag/delete/{doc_id}` - Delete RAG document

**Example request**:
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "What is machine learning?"}'
```

### 3. Training (`train.py`)

Basic training script using numpy (placeholder for gradient descent).

**Requirements**: numpy

```bash
python train.py training_data.json
```

**Options**:
- `--batch-size`: Batch size (default: 8)
- `--dim`: Model dimension (default: 128)
- `--layers`: Number of layers (default: 2)
- `--heads`: Number of attention heads (default: 4)
- `--ff`: Feed-forward hidden size (default: 256)
- `--max-len`: Maximum sequence length (default: 128)

### 4. PyTorch Training (`torch_train.py`)

Full training script using PyTorch with gradient descent.

**Requirements**: Install torch extras
```bash
pip install kroxai-mini[torch]
```

**Run**:
```bash
python torch_train.py training_data.json
```

**Options**:
- `--epochs`: Number of epochs (default: 10)
- `--batch-size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-3)
- `--preset`: Model preset (tiny/small/base/large)
- `--device`: Device (cpu/cuda/mps)
- `--checkpoint-dir`: Directory to save checkpoints
- And many more...

**Example**:
```bash
python torch_train.py data.json --epochs 20 --batch-size 32 --preset base
```

### 5. DPO Training (`dpo_simple.py`)

Direct Preference Optimization training for preference learning.

**Requirements**: Install torch extras
```bash
pip install kroxai-mini[torch]
```

**Run**:
```bash
python dpo_simple.py preferences.jsonl --epochs 5
```

Expected JSONL format:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 6. Evaluation Harness (`eval_harness.py`)

Evaluate the model using a test dataset.

**Requirements**: requests (optional)

```bash
python eval_harness.py prompts.jsonl
```

**Options**:
- `--server-url`: Server URL (default: http://127.0.0.1:5000)
- `--api-key`: API key (or set KROXAI_API_KEY)
- `--output`: Output JSONL file
- `--timeout`: Request timeout in seconds

### 7. Invoice Send (`invoice_send.py`)

Example integration script (requires additional setup).

## Data Format

### Training Data (JSON)

For `train.py` and `torch_train.py`:

```json
[
  {
    "question": "What is AI?",
    "answer": "AI is artificial intelligence..."
  },
  {
    "question": "How does ML work?",
    "answer": "Machine learning works by..."
  }
]
```

### Preference Data (JSONL)

For `dpo_simple.py`:

```json
{"prompt": "Explain quantum computing", "chosen": "Good explanation...", "rejected": "Bad explanation..."}
{"prompt": "What is Python?", "chosen": "Python is...", "rejected": "Python..."}
```

### Evaluation Prompts (JSONL)

For `eval_harness.py`:

```json
{"prompt": "What is the capital of France?"}
{"prompt": "Explain machine learning"}
```

## Environment Variables

### Server Configuration

- `KROXAI_API_KEY`: API key for authentication
- `KROXAI_CORS_ORIGINS`: CORS origins (comma-separated or *)
- `KROXAI_MEMORY_TURNS`: Number of conversation turns to remember (default: 6)
- `KROXAI_MAX_CONTEXT_CHARS`: Max context characters (default: 8000)
- `KROXAI_RAG`: Enable RAG (default: 1)
- `KROXAI_RAG_PATH`: Path to RAG store file
- `KROXAI_CE_RERANK`: Enable cross-encoder reranking (default: 0)
- `KROXAI_CE_MODEL`: Cross-encoder model name
- `KROXAI_TEMPLATES_DIR`: Templates directory
- `KROXAI_STATIC_DIR`: Static files directory

## Tips

1. **Start Simple**: Begin with `chat.py` to test basic functionality
2. **Use Presets**: The `--preset` option provides pre-configured model sizes
3. **Monitor Resources**: Training can be memory-intensive; adjust batch size accordingly
4. **Save Checkpoints**: Use `--checkpoint-dir` in torch_train.py to save progress
5. **Test Locally**: Test with small datasets before scaling up

## Getting Help

- Main README: ../README.md
- Installation Guide: ../INSTALL.md
- Quick Start: ../QUICKSTART.md
- Issues: https://github.com/KroxenStudios/KroxAI-Mini/issues
