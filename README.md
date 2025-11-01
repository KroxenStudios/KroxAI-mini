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




