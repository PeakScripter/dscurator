from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def write_records(records: List[Dict[str, object]], out_path: Path, fmt: str) -> None:
	fmt = fmt.lower()
	out_path.parent.mkdir(parents=True, exist_ok=True)
	if fmt == "csv":
		pd.DataFrame.from_records(records).to_csv(out_path, index=False)
	elif fmt in {"jsonl", "jsonlines"}:
		with out_path.open("w", encoding="utf-8") as f:
			for r in records:
				f.write(json.dumps(r, ensure_ascii=False) + "\n")
	elif fmt == "txt":
		with out_path.open("w", encoding="utf-8") as f:
			for r in records:
				f.write(str(r) + "\n")
	else:
		raise ValueError(f"Unsupported format: {fmt}")


