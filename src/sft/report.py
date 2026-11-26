"""Build comparison tables and loss curves for a LoRA sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import TrainConfig, load_lora_specs


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(train_config: str | Path, lora_config: str | Path) -> dict[str, str]:
    cfg = TrainConfig.from_yaml(train_config)
    specs = load_lora_specs(lora_config)
    root = Path(cfg.output_dir)

    train_results = {item["name"]: item for item in _read_json(root / "results.json", [])}
    eval_results = _read_json(root / "sqlite_eval.json", {})

    header = [
        "model",
        "r",
        "alpha",
        "target_modules",
        "trainable_params",
        "final_train_loss",
        "eval_loss",
        "execution_accuracy",
        "valid_sql_rate",
    ]
    rows: list[list[str]] = []
    base_eval = eval_results.get("base", {})
    rows.append(
        ["base", "-", "-", "-", "-", "-", "-",
         _fmt(base_eval.get("execution_accuracy")), _fmt(base_eval.get("valid_sql_rate"))]
    )

    for spec in specs:
        result = train_results.get(spec.name, {})
        evaluation = eval_results.get(spec.name, {})
        history = _read_json(root / spec.name / "history.json", [])
        final_loss = history[-1]["loss"] if history else None
        rows.append(
            [
                spec.name,
                str(spec.r),
                str(spec.lora_alpha),
                "+".join(spec.target_modules),
                _fmt(result.get("trainable_params")),
                _fmt(final_loss),
                _fmt(result.get("eval_loss")),
                _fmt(evaluation.get("execution_accuracy")),
                _fmt(evaluation.get("valid_sql_rate")),
            ]
        )

    csv_path = root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)

    markdown = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    markdown += ["| " + " | ".join(row) + " |" for row in rows]
    md_path = root / "results.md"
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 5))
    for spec in specs:
        history = _read_json(root / spec.name / "history.json", [])
        if history:
            axis.plot(
                [point["step"] for point in history],
                [point["loss"] for point in history],
                marker="o",
                label=spec.name,
            )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training loss")
    axis.set_title("LoRA training loss curves")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    curve_path = root / "loss_curves.png"
    figure.savefig(curve_path, dpi=150)
    plt.close(figure)

    return {"csv": str(csv_path), "markdown": str(md_path), "loss_curves": str(curve_path)}
