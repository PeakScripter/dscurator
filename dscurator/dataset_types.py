from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def make_corpus_prompt(topic: str, style: str = "neutral", length: str = "short") -> str:
	return (
		f"Write a {length} informational paragraph about '{topic}' in a {style} style."
	)


def make_qa_prompt(topic: str) -> str:
	return (
		f"Generate a challenging question and a correct, concise answer about: {topic}.\n"
		"Respond as JSON with fields 'question' and 'answer'."
	)


def make_cot_prompt(topic: str) -> str:
	return (
		f"Create a reasoning problem for: {topic}.\n"
		"Return JSON with 'question', 'reasoning' (step-by-step), and 'final_answer'."
	)


def make_agent_prompt(topic: str) -> str:
	return (
		"Simulate an agent solving a user goal.\n"
		f"User goal: {topic}.\n"
		"Output a JSON array of steps; each step has 'thought', 'action', 'observation'."
	)


