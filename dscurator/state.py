from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


RUNS_DIR = Path("runs")


@dataclass
class RunState:
	run_id: str
	config: Dict[str, Any]
	records: List[Dict[str, Any]]
	progress_index: int = 0

	@property
	def root(self) -> Path:
		return RUNS_DIR / self.run_id

	def save(self) -> None:
		self.root.mkdir(parents=True, exist_ok=True)
		with (self.root / "state.json").open("w", encoding="utf-8") as f:
			json.dump(asdict(self), f, ensure_ascii=False, indent=2)

	@staticmethod
	def load(run_id: str) -> "RunState":
		with ((RUNS_DIR / run_id) / "state.json").open("r", encoding="utf-8") as f:
			data = json.load(f)
		return RunState(**data)


def new_run_id(prefix: str = "RUN") -> str:
	return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


