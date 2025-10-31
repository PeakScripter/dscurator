from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print

from .generator import synthesize


app = typer.Typer(add_completion=False, help="Domain-Specific Synthetic Data Generator")


@app.command()
def run(
	model: str = typer.Option("gpt-4o-mini", help="Model name (LiteLLM compatible)"),
	domain: Optional[str] = typer.Option(None, help="Short domain seed, e.g., 'tourism'"),
	dataset: str = typer.Option("qa", help="corpus|qa|cot|agent"),
	format: str = typer.Option("csv", help="csv|jsonl|txt"),
	limit: int = typer.Option(100, help="Max number of items to generate"),
	max_depth: int = typer.Option(2, help="Tree expansion depth"),
	branching: int = typer.Option(5, help="Children per node during expansion"),
	algorithm: str = typer.Option("bfs", help="Algorithm: bfs|walk|clustering|markov|graph"),
	walk_steps: int = typer.Option(15, help="Number of steps for walk algorithm"),
	num_workers: int = typer.Option(1, help="Concurrent workers (1=sequential)"),
	enforce_diversity: bool = typer.Option(False, help="Filter similar records"),
	min_quality_score: float = typer.Option(0.0, help="Minimum quality score (0.0-1.0)"),
	resume: Optional[str] = typer.Option(None, help="Run id to resume"),
	output: Optional[str] = typer.Option(None, help="Output directory (defaults to outputs/<run>)"),
	template: Optional[str] = typer.Option(None, help="YAML template path for customization"),
):
	"""Generate a dataset from a seed or YAML template."""
	run_id = synthesize(
		model=model,
		domain=domain,
		dataset=dataset,
		format=format,
		limit=limit,
		max_depth=max_depth,
		branching=branching,
		resume=resume,
		output=output,
		template_yaml=template,
		algorithm=algorithm,
		walk_steps=walk_steps,
		num_workers=num_workers,
		enforce_diversity=enforce_diversity,
		min_quality_score=min_quality_score,
	)
	print(f"[bold green]Run complete:[/bold green] {run_id}")


@app.command("init-template")
def init_template(
	type: str = typer.Option("qa", help="Template type: corpus|qa|cot|agent"),
	out: str = typer.Option("template.yaml", help="Destination path"),
):
	"""Write a starter YAML template you can customize."""
	templates = {
		"corpus": {
			"type": "corpus",
			"seed": "tourism",
			"prompt_template": "Write a paragraph about {topic} (path: {path}).",
			"system": "You are a domain expert writer.",
		},
		"qa": {
			"type": "qa",
			"seed": "tourism",
			"prompt_template": "Generate Q&A for {topic} (path: {path}).",
			"system": "You are a careful teacher; keep answers precise.",
		},
		"cot": {
			"type": "cot",
			"seed": "tourism",
			"prompt_template": "Create a reasoning problem for {topic}.",
			"system": "Explain reasoning steps clearly before the answer.",
		},
		"agent": {
			"type": "agent",
			"seed": "plan a 3-day trip in Kyoto",
			"prompt_template": "Simulate agent steps for {topic}.",
			"system": "Return clean JSON with steps.",
		},
	}
	data = templates.get(type, templates["qa"])  # default qa
	Path(out).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
	print(f"Template written to {out}")


def main() -> None:
	app()


if __name__ == "__main__":
	main()


