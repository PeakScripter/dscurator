from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from rich.progress import track

from .dataset_types import make_agent_prompt, make_corpus_prompt, make_cot_prompt, make_qa_prompt
from .providers import LLMProvider
from .prompt_algorithms import (
	TopicNode,
	expand_topics_bfs,
	graph_walk_prompts,
	hierarchical_clustering_prompts,
	markov_chain_prompts,
	prompts_from_tree,
	random_walk_prompts,
)
from .quality import QualityFilter, score_record
from .state import RunState, new_run_id
from .writers import write_records


def load_template(path: Optional[str]) -> Dict[str, object]:
	if not path:
		return {}
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def strip_markdown_json(text: str) -> str:
	"""Strip markdown code blocks from JSON responses."""
	# Remove markdown code blocks (```json ... ```)
	text = re.sub(r'```json\s*\n', '', text)
	text = re.sub(r'```\s*\n?$', '', text)
	# Remove leading/trailing whitespace
	text = text.strip()
	return text


def synthesize(
	*,
	model: str,
	domain: Optional[str] = None,
	template_yaml: Optional[str] = None,
	dataset: str = "qa",
	format: str = "csv",
	limit: int = 100,
	max_depth: int = 2,
	branching: int = 5,
	resume: Optional[str] = None,
	output: Optional[str] = None,
	walk_steps: int = 0,
	algorithm: str = "bfs",
	num_workers: int = 1,
	enforce_diversity: bool = False,
	min_quality_score: float = 0.0,
) -> str:
	"""Main orchestration to create a dataset and return the run id.

	- If `resume` is provided, continue that run.
	- Otherwise, start a new run using `domain` and/or `template_yaml`.
	- algorithm: bfs|walk|clustering|markov|graph
	- num_workers: >1 for concurrent generation
	"""

	if resume:
		state = RunState.load(resume)
		config = state.config
		records: List[Dict[str, object]] = state.records
	else:
		config = {
			"model": model,
			"domain": domain,
			"dataset": dataset,
			"format": format,
			"limit": limit,
			"max_depth": max_depth,
			"branching": branching,
			"walk_steps": walk_steps,
			"algorithm": algorithm,
			"num_workers": num_workers,
		}
		records = []
		run_id = new_run_id()
		state = RunState(run_id=run_id, config=config, records=records)

	provider = LLMProvider(model=config["model"])  # type: ignore[index]

	# Build prompts list
	prompts: List[str] = []
	user_template = load_template(template_yaml)
	if resume:
		prompts = state.config.get("prompts", [])  # type: ignore[assignment]
	else:
		if config.get("domain"):
			seed: str = str(config["domain"])  # type: ignore[index]
			system_hint = user_template.get("system") if user_template else None
			alg = config.get("algorithm", "bfs")
			
			if alg == "bfs":
				topics = expand_topics_bfs(
					seed,
					provider,
					max_depth=int(config["max_depth"]),
					branching=int(config["branching"]),
					system=system_hint,  # type: ignore[arg-type]
				)
				template_prompt = (user_template.get("prompt_template") if user_template else None) or "Create data about {topic} (path: {path})."
				prompts = prompts_from_tree(topics, template_prompt)
			elif alg == "walk":
				prompts = random_walk_prompts(seed, provider, steps=int(config.get("walk_steps", 15)), system=system_hint)  # type: ignore[arg-type]
			elif alg == "clustering":
				prompts = hierarchical_clustering_prompts(seed, provider, num_clusters=3, per_cluster=10, system=system_hint)  # type: ignore[arg-type]
			elif alg == "markov":
				prompts = markov_chain_prompts(seed, provider, num_states=20, transitions=3, system=system_hint)  # type: ignore[arg-type]
			elif alg == "graph":
				prompts = graph_walk_prompts(seed, provider, iterations=15, system=system_hint)  # type: ignore[arg-type]
			else:
				# Default to bfs
				topics = expand_topics_bfs(seed, provider, max_depth=int(config["max_depth"]), branching=int(config["branching"]), system=system_hint)  # type: ignore[arg-type]
				template_prompt = (user_template.get("prompt_template") if user_template else None) or "Create data about {topic} (path: {path})."
				prompts = prompts_from_tree(topics, template_prompt)
		elif user_template and user_template.get("seed"):
			prompts = [str(user_template["seed"])]

	state.config["prompts"] = prompts
	state.save()

	# Choose builder for dataset type
	def build_record(prompt: str) -> Dict[str, object]:
		if config["dataset"] == "corpus":
			text = provider.generate(make_corpus_prompt(prompt), system=user_template.get("system") if user_template else None)
			return {"topic": prompt, "text": text}
		if config["dataset"] == "qa":
			resp = provider.generate(make_qa_prompt(prompt), system=user_template.get("system") if user_template else None)
			try:
				# Try stripping markdown first
				cleaned = strip_markdown_json(resp)
				obj = json.loads(cleaned)
			except Exception:
				obj = {"question": prompt, "answer": resp}
			obj["topic"] = prompt
			return obj
		if config["dataset"] == "cot":
			resp = provider.generate(make_cot_prompt(prompt), system=user_template.get("system") if user_template else None)
			try:
				cleaned = strip_markdown_json(resp)
				obj = json.loads(cleaned)
			except Exception:
				obj = {"question": prompt, "reasoning": resp, "final_answer": ""}
			obj["topic"] = prompt
			return obj
		if config["dataset"] == "agent":
			resp = provider.generate(make_agent_prompt(prompt), system=user_template.get("system") if user_template else None)
			try:
				cleaned = strip_markdown_json(resp)
				steps = json.loads(cleaned)
			except Exception:
				steps = [{"thought": "", "action": "", "observation": resp}]
			return {"topic": prompt, "trajectory": steps}
		raise ValueError(f"Unknown dataset type: {config['dataset']}")

	# Generate
	start = state.progress_index
	end = min(len(prompts), int(config["limit"]))
	num_workers = int(config.get("num_workers", 1))
	quality_filter = QualityFilter()
	
	if num_workers <= 1:
		# Sequential generation
		for i in track(range(start, end), description="Generating"):
			rec = build_record(prompts[i])
			score = score_record(rec)
			if score >= config.get("min_quality_score", 0.0):
				records.append(rec)
			state.records = records
			state.progress_index = i + 1
			state.save()
	else:
		# Concurrent generation
		remaining = [(idx, p) for idx, p in enumerate(prompts[start:end])]
		completed = 0
		
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			futures = {executor.submit(build_record, p): (start + idx, p) for idx, p in remaining}
			
			for future in track(as_completed(futures), total=len(futures), description="Generating"):
				idx, _ = futures[future]
				try:
					rec = future.result()
					score = score_record(rec)
					if score >= config.get("min_quality_score", 0.0):
						records.append(rec)
					state.records = records
					state.progress_index = max(state.progress_index, idx + 1)
					completed += 1
					# Save periodically
					if completed % 5 == 0:
						state.save()
				except Exception as e:
					print(f"Error generating record {idx}: {e}")
		state.save()
	
	# Apply quality filter if enabled
	if config.get("enforce_diversity", False):
		records = quality_filter.filter_records(records, enforce_diversity=True)

	# Write output
	out_dir = Path("outputs") / state.run_id
	if output:
		out_dir = Path(output)
	out_dir.mkdir(parents=True, exist_ok=True)
	write_records(records, out_dir / f"dataset.{config['format']}", fmt=str(config["format"]))
	state.save()
	return state.run_id


