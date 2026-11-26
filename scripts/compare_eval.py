"""Paired comparison of SQLite execution eval results.

Reads per-row predictions written by sft.eval.evaluate_model(details_path=...)
and produces accuracy confidence intervals plus paired McNemar tests and
bootstrap confidence intervals for differences against the best model.

Usage: uv run python scripts/compare_eval.py --dir results/runs/large_eval
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2 * tail)


def bootstrap_diff(a: list[bool], b: list[bool], reps: int = 10000) -> tuple[float, float]:
    rng = random.Random(0)
    n = len(a)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in idx) / n
        db = sum(b[i] for i in idx) / n
        diffs.append(da - db)
    diffs.sort()
    return (diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("results/runs/large_eval"))
    parser.add_argument("--best", default="mlp_r32")
    args = parser.parse_args()

    summary = json.loads((args.dir / "sqlite_eval_2000.json").read_text(encoding="utf-8"))
    details: dict[str, list[bool]] = {}
    for path in sorted((args.dir / "eval_details_2000").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        details[path.stem] = [bool(row["correct"]) for row in payload["details"]]

    names = list(summary.keys())
    if not all(name in details for name in names):
        raise SystemExit("missing per-row details for some models")
    lengths = {len(v) for v in details.values()}
    if len(lengths) != 1:
        raise SystemExit(f"per-row details have inconsistent lengths: {lengths}")
    n = lengths.pop()

    best = args.best
    best_rows = details[best]
    report: dict[str, Any] = {
        "rows": n,
        "best": best,
        "models": {},
    }
    for name in names:
        rows = details[name]
        acc = sum(rows) / n
        lo, hi = wilson(sum(rows), n)
        entry: dict[str, Any] = {
            "execution_accuracy": summary[name]["execution_accuracy"],
            "valid_sql_rate": summary[name]["valid_sql_rate"],
            "wilson_95ci": [lo, hi],
        }
        if name != best:
            b = sum(1 for x, y in zip(best_rows, rows, strict=True) if x and not y)
            c = sum(1 for x, y in zip(best_rows, rows, strict=True) if y and not x)
            diff_lo, diff_hi = bootstrap_diff(best_rows, rows)
            entry["vs_best"] = {
                "best_correct_other_wrong": b,
                "best_wrong_other_correct": c,
                "mcnemar_exact_p": mcnemar_exact(b, c),
                "accuracy_difference": sum(best_rows) / n - acc,
                "bootstrap_95ci": [diff_lo, diff_hi],
            }
        report["models"][name] = entry

    (args.dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Paired eval comparison ({n} rows, best = {best})",
        "",
        "| model | exec accuracy | Wilson 95% CI | valid SQL | McNemar p vs best | Δ accuracy (bootstrap 95% CI) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        entry = report["models"][name]
        ci = entry["wilson_95ci"]
        vs = entry.get("vs_best", {})
        p = f"{vs['mcnemar_exact_p']:.4g}" if vs else "–"
        diff = (
            f"{vs['accuracy_difference']:+.4f} [{vs['bootstrap_95ci'][0]:+.4f}, {vs['bootstrap_95ci'][1]:+.4f}]"
            if vs
            else "–"
        )
        lines.append(
            f"| {name} | {entry['execution_accuracy']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{entry['valid_sql_rate']:.4f} | {p} | {diff} |"
        )
    (args.dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.dir / 'comparison.json'} and {args.dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
