<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LiteLLM-Multi--Provider-orange?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI%20%7C%20Gemini%20%7C%20Claude-Supported-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Output-CSV%20%7C%20JSONL%20%7C%20TXT-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">🗂 dscurator</h1>

<p align="center">
  <b>Domain-specific synthetic dataset generation — from a single keyword to thousands of training examples.<br/>
  BFS trees, Markov chains, graph expansion, concurrent generation, and quality filtering. All in one CLI.</b>
</p>

<p align="center">
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-dataset-types">Dataset Types</a> •
  <a href="#-providers">Providers</a>
</p>

---

## 🔁 How It Works

```
         Input: Domain keyword (e.g., "medical imaging")
                          │
                          ▼
         ┌────────────────────────────────┐
         │     Prompt Expansion Engine    │
         │                               │
         │  ┌──────────┐  ┌───────────┐  │
         │  │ BFS Tree │  │ Markov    │  │
         │  │          │  │ Chain     │  │
         │  └──────────┘  └───────────┘  │
         │  ┌──────────┐  ┌───────────┐  │
         │  │ Random   │  │ Graph     │  │
         │  │ Walk     │  │ Expansion │  │
         │  └──────────┘  └───────────┘  │
         └────────────────────────────────┘
                          │
                          ▼ (hundreds of diverse prompts)
         ┌────────────────────────────────┐
         │    Concurrent LLM Generation   │
         │    (via LiteLLM — any model)   │
         └────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────┐
         │      Quality Filtering         │
         │  - Length checks               │
         │  - Diversity scoring           │
         │  - Embedding similarity        │
         └────────────────────────────────┘
                          │
                          ▼
         📦 Dataset: CSV / JSONL / TXT
            (Q&A, corpus, CoT, agent trajectories)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Algorithmic Prompt Expansion** | BFS trees, random walks, Markov chains, hierarchical clustering, graph traversal |
| 🔌 **Multi-Provider Support** | OpenAI, Anthropic (Claude), Google (Gemini), Cohere, Groq, Ollama — unified via LiteLLM |
| 📊 **Multiple Dataset Types** | Q&A pairs, text corpora, chain-of-thought, agent trajectories |
| 📁 **Flexible Output Formats** | CSV, JSONL, TXT — ready for fine-tuning pipelines |
| ⚡ **Concurrent Generation** | Parallel workers for fast, large-scale dataset creation |
| 🔍 **Quality Filtering** | Length checks, diversity scoring, embedding-based deduplication |
| 💾 **Resumable Runs** | Save state and continue interrupted generation jobs |
| 📝 **YAML Templates** | Fully customizable prompts and system messages |

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/PeakScripter/dscurator.git
cd dscurator

# Install
pip install -r requirements.txt

# Set your API key (any supported provider)
export OPENAI_API_KEY=your_key_here
# or GOOGLE_API_KEY, ANTHROPIC_API_KEY, etc.

# Generate a Q&A dataset for "medical imaging"
python main.py --domain "medical imaging" --type qa --output dataset.jsonl

# Generate with Gemini, 500 examples, concurrent
python main.py --domain "astronomy" --type corpus --provider gemini --count 500 --workers 8
```

---

## 📦 Dataset Types

| Type | Description | Use Case |
|------|-------------|----------|
| `qa` | Question-answer pairs | Instruction fine-tuning |
| `corpus` | Domain text passages | Language model pre-training |
| `cot` | Chain-of-thought reasoning | Reasoning fine-tuning |
| `agent` | Tool-use / agent trajectories | Agent fine-tuning (ReAct, etc.) |

---

## 🔌 Providers

dscurator uses **LiteLLM** as a unified gateway — switch providers with a single flag:

```bash
--provider openai      # GPT-4o, GPT-3.5, etc.
--provider gemini      # Gemini 1.5 Pro / Flash
--provider claude      # Claude 3.5 Sonnet / Haiku
--provider groq        # Llama, Mixtral (fast inference)
--provider ollama      # Local models (no API key needed)
```

---

## Prompt Expansion Algorithms

```
BFS Tree         — breadth-first expansion of topic subtopics
Random Walk      — stochastic exploration of the topic space
Markov Chain     — probabilistic next-topic generation
Graph Expansion  — knowledge-graph-style traversal
Hierarchical     — cluster-then-expand for diverse coverage
```

---

## 🗂 Output Structure

```
output/
├── dataset.jsonl        # Generated examples (JSONL)
├── dataset.csv          # Same data in CSV format
├── prompts_used.txt     # All expanded prompts (for inspection)
└── run_state.json       # Checkpoint for resumable runs
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM Gateway:** LiteLLM (multi-provider)
- **Graph Algorithms:** NetworkX
- **Embeddings:** sentence-transformers (for quality filtering)
- **Concurrency:** asyncio + ThreadPoolExecutor

---

## 📄 License

MIT License — see LICENSE for details.
