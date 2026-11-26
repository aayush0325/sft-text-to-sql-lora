"""Build a larger, deterministic SQLite execution eval set from the HF test split.

Usage: uv run python scripts/make_eval_set.py --n 2000 --seed 42

The original 32 hand-curated rows in data/eval_set.jsonl are always included.
Additional rows come from gretelai/synthetic_text_to_sql/test, deduplicated and
filtered to those whose reference SQL executes against their own schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sft.data import DATA_FILES, load_hf_dataset
from sft.eval import execute_sql


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["sql_prompt"], row["sql_context"], row["sql"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/eval_set_2000.jsonl"))
    args = parser.parse_args()

    seed_rows = [
        json.loads(line)
        for line in Path("data/eval_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    keys = {row_key(row) for row in seed_rows}

    test = load_hf_dataset("test")
    test = test.shuffle(seed=args.seed)
    selected = list(seed_rows)
    for row in test:
        if len(selected) >= args.n:
            break
        key = row_key(row)
        if key in keys:
            continue
        keys.add(key)
        if execute_sql(row["sql"], row["sql_context"]) is None:
            continue
        selected.append({field: row[field] for field in ("id", "sql_prompt", "sql_context", "sql")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} rows to {args.out} ({len(seed_rows)} seed rows included)")


if __name__ == "__main__":
    main()
