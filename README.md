# DS Curator – Domain-Specific Synthetic Data Generation Toolkit

DS Curator generates synthetic datasets from a short domain idea (e.g., "tourism").
It expands prompts via algorithmic trees/graphs and queries your choice of LLMs to produce datasets in CSV/JSONL/TXT. It supports Q&A pairs, text corpora, chain-of-thought, and agent-like trajectories. It can resume runs from saved state.

## Features
- **Algorithmic prompt expansion**: BFS trees, random walks, hierarchical clustering, Markov chains, graph exploration
- **Multi-provider**: OpenAI, Anthropic (Claude), Google (Gemini), Cohere, Groq, Ollama, via LiteLLM
- **Dataset types**: corpus, Q&A, chain-of-thought, agent trajectories
- **Output formats**: CSV, JSONL, TXT
- **YAML templates**: Customizable prompts and system messages
- **State management**: Resume interrupted runs
- **Concurrent generation**: Parallel workers for faster generation
- **Quality filtering**: Length checks, diversity scoring, embedding-based similarity

## Quickstart
```bash
# Create virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set API keys
$env:OPENAI_API_KEY = "your-key-here"
$env:ANTHROPIC_API_KEY = "your-key-here"  # optional

# Generate 100 tourism Q&A pairs
python -m dscurator run --domain "tourism" --dataset qa --model "gpt-4o-mini" --format csv --limit 100

# Use concurrent generation (4 workers) with quality filtering
python -m dscurator run --domain "tourism" --dataset qa --algorithm clustering --num_workers 4 --enforce_diversity --min_quality_score 0.3
```

## Algorithms

Choose expansion strategies:
- **bfs** (default): Breadth-first tree expansion
- **walk**: Random walk through related topics
- **clustering**: Hierarchical clustering approach
- **markov**: Markov chain generation
- **graph**: Graph traversal with neighbor exploration

## Templates

Export and customize templates:
```bash
python -m dscurator init-template --type qa --out template.qa.yaml
```

Use example templates in `examples/`:
```bash
python -m dscurator run --template examples/tourism_qa.yaml --model "gpt-4o-mini" --format jsonl
```

## Resume a run
```bash
python -m dscurator run --resume RUN_2025-10-31_12-00-00
```

## Advanced Options

**Concurrent generation**: Speed up with parallel workers
```bash
python -m dscurator run --domain "finance" --num_workers 8 --algorithm markov
```

**Quality filtering**: Use embedding-based diversity checking
```bash
python -m dscurator run --domain "science" --enforce_diversity --min_quality_score 0.5
```

**Resume interrupted runs**: State is saved in `runs/<run_id>/`
```bash
python -m dscurator run --resume RUN_2025-01-15_14-30-00
```

## Providers

Powered by `litellm`; supports OpenAI, Anthropic, Google, Cohere, Groq, Ollama, and many more.

Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.
For Ollama/local models: `LITELLM_API_BASE=http://localhost:11434`

## Examples

See `examples/` for domain-specific templates:
- Tourism Q&A pairs
- Medical chain-of-thought reasoning
- Tech agent trajectories

## Notes

- Agent trajectories prompt the model for JSON step arrays
- CoT datasets include explicit reasoning fields
- Quality filtering uses sentence-transformers embeddings (auto-downloads on first use)
- Concurrent workers help with high-volume generation, but watch rate limits


