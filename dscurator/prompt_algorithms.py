from __future__ import annotations

import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from .providers import LLMProvider


@dataclass
class TopicNode:
	name: str
	depth: int
	parent: Optional[str] = None


def expand_topics_bfs(
	seed: str,
	provider: LLMProvider,
	max_depth: int = 2,
	branching: int = 5,
	system: Optional[str] = None,
) -> List[TopicNode]:
	"""Expand a seed domain into a topic tree via BFS with the LLM.

	The LLM is asked to return concise subtopic names. We keep a simple graph in memory.
	"""
	queue: Deque[TopicNode] = deque([TopicNode(seed, depth=0, parent=None)])
	result: List[TopicNode] = []

	while queue:
		node = queue.popleft()
		result.append(node)
		if node.depth >= max_depth:
			continue

		prompt = (
			f"You are expanding the domain '{node.name}'. "
			f"List {branching} specific subtopics as a bullet list; one short phrase per line."
		)
		text = provider.generate(prompt=prompt, system=system, temperature=0.4)
		subtopics = [line.strip("- • \t ") for line in text.splitlines() if line.strip()]
		for sub in subtopics[:branching]:
			queue.append(TopicNode(sub, depth=node.depth + 1, parent=node.name))

	return result


def random_walk_prompts(
	seed: str,
	provider: LLMProvider,
	steps: int = 15,
	system: Optional[str] = None,
) -> List[str]:
	"""Generate a chain of related prompts via a lightweight random walk.

	At each step we ask the model for the next related, more specific angle.
	"""
	current = seed
	walk: List[str] = [current]
	for _ in range(steps - 1):
		prompt = (
			f"Current topic: {current}\n"
			"Suggest a next, more specific angle as a short phrase only."
		)
		current = provider.generate(prompt=prompt, system=system, temperature=0.7).splitlines()[0].strip()
		walk.append(current)
	return walk


def prompts_from_tree(topics: List[TopicNode], template_prompt: str) -> List[str]:
	"""Render prompts from a topic tree using a string template with {topic} and {path}."""
	paths: Dict[str, List[str]] = {}
	for node in topics:
		path: List[str] = [node.name]
		parent = node.parent
		while parent is not None:
			path.append(parent)
			parent = next((n.parent for n in topics if n.name == parent), None)
		paths[node.name] = list(reversed(path))

	return [template_prompt.format(topic=n.name, path=", ".join(paths[n.name])) for n in topics]


def hierarchical_clustering_prompts(
	seed: str,
	provider: LLMProvider,
	num_clusters: int = 3,
	per_cluster: int = 10,
	system: Optional[str] = None,
) -> List[str]:
	"""Use LLM-guided hierarchical clustering to generate diverse prompts.
	
	First asks for high-level themes, then drills into each theme.
	"""
	prompt = (
		f"Domain: {seed}\n"
		f"List {num_clusters} high-level themes as bullets."
	)
	text = provider.generate(prompt=prompt, system=system, temperature=0.5)
	themes = [line.strip("- • \t ") for line in text.splitlines() if line.strip()][:num_clusters]
	
	result: List[str] = []
	for theme in themes:
		prompt = (
			f"Theme: {theme}\n"
			f"List {per_cluster} specific subtopics as bullets."
		)
		text = provider.generate(prompt=prompt, system=system, temperature=0.6)
		subs = [line.strip("- • \t ") for line in text.splitlines() if line.strip()][:per_cluster]
		result.extend(subs)
	return result


def markov_chain_prompts(
	seed: str,
	provider: LLMProvider,
	num_states: int = 20,
	transitions: int = 3,
	system: Optional[str] = None,
) -> List[str]:
	"""Generate prompts via a Markov-like chain where each state influences the next."""
	states: List[str] = [seed]
	
	prompt = (
		f"Seed topic: {seed}\n"
		"List 3 closely related topics as bullets."
	)
	text = provider.generate(prompt=prompt, system=system, temperature=0.6)
	next_states = [line.strip("- • \t ") for line in text.splitlines() if line.strip()][:transitions]
	states.extend(next_states)
	
	for _ in range(num_states - len(states)):
		current = random.choice(states[-transitions:])
		prompt = (
			f"Given topic: {current}\n"
			"Suggest 1 related, more specific angle as one short phrase only."
		)
		new = provider.generate(prompt=prompt, system=system, temperature=0.7).splitlines()[0].strip()
		states.append(new)
	
	return states


def graph_walk_prompts(
	seed: str,
	provider: LLMProvider,
	iterations: int = 15,
	system: Optional[str] = None,
) -> List[str]:
	"""Walk through a conceptual graph by asking for neighbors at each node."""
	visited: Set[str] = {seed}
	result: List[str] = [seed]
	current = seed
	
	for _ in range(iterations - 1):
		prompt = (
			f"Topic: {current}\n"
			"List 3 related topics (one per line) that we haven't covered yet."
		)
		text = provider.generate(prompt=prompt, system=system, temperature=0.7)
		candidates = [line.strip("- • \t ") for line in text.splitlines() if line.strip()]
		candidates = [c for c in candidates if c not in visited]
		
		if candidates:
			current = random.choice(candidates)
			visited.add(current)
			result.append(current)
		else:
			# Fallback: jump to a less explored area
			prompt = (
				f"Given topic: {current}\n"
				"Suggest a tangential but interesting angle as one phrase."
			)
			current = provider.generate(prompt=prompt, system=system, temperature=0.8).splitlines()[0].strip()
			result.append(current)
	
	return result
