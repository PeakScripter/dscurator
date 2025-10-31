from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import json

try:
	from sentence_transformers import SentenceTransformer
	import numpy as np
	has_embeddings = True
except ImportError:
	has_embeddings = False


class QualityFilter:
	"""Filters and scores generated data for quality and diversity."""
	
	def __init__(self, min_length: int = 10, max_embedding_distance: float = 0.85) -> None:
		self.min_length = min_length
		self.max_embedding_distance = max_embedding_distance
		self.encoder: Optional[SentenceTransformer] = None
		if has_embeddings:
			try:
				self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
			except Exception:
				pass  # Gracefully degrade if model can't load
	
	def check_length(self, record: Dict[str, Any]) -> bool:
		"""Check if a record has minimum content."""
		text = json.dumps(record, ensure_ascii=False)
		return len(text) >= self.min_length
	
	def check_diversity(self, new_record: Dict[str, Any], existing: List[Dict[str, Any]], threshold: float = 0.85) -> bool:
		"""Check if new_record is diverse enough from existing records.
		
		Uses embedding similarity if available, otherwise falls back to simple checks.
		"""
		if not existing or not self.encoder:
			return True
		
		try:
			new_text = json.dumps(new_record, ensure_ascii=False)
			existing_texts = [json.dumps(r, ensure_ascii=False) for r in existing]
			
			new_emb = self.encoder.encode([new_text])[0]
			existing_embs = self.encoder.encode(existing_texts)
			
			similarities = np.dot(existing_embs, new_emb) / (np.linalg.norm(existing_embs, axis=1) * np.linalg.norm(new_emb))
			max_sim = float(np.max(similarities))
			return max_sim < threshold
		except Exception:
			return True  # On error, allow the record
	
	def filter_records(self, records: List[Dict[str, Any]], enforce_diversity: bool = False) -> List[Dict[str, Any]]:
		"""Filter records for quality and optionally enforce diversity."""
		filtered: List[Dict[str, Any]] = []
		
		for rec in records:
			if not self.check_length(rec):
				continue
			if enforce_diversity and not self.check_diversity(rec, filtered):
				continue
			filtered.append(rec)
		
		return filtered


def score_record(record: Dict[str, Any]) -> float:
	"""Simple quality score between 0 and 1.
	
	Currently based on:
	- Has required fields
	- Not too short
	- Not redundant
	"""
	score = 0.0
	text = json.dumps(record, ensure_ascii=False).lower()
	
	# Length bonus
	score += min(len(text) / 500.0, 0.4)
	
	# Field completeness bonus
	has_fields = len([k for k, v in record.items() if v and v != ""])
	score += min(has_fields / 5.0, 0.3)
	
	# Diversity bonus (simple heuristics)
	if len(set(text.split())) > 10:
		score += 0.3
	
	return min(score, 1.0)

