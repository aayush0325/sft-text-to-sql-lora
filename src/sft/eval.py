"""SQLite execution-based evaluation of the base model and LoRA adapters.

Generated SQL and the reference SQL are both run against an in-memory SQLite
database built from the row's `sql_context`. A prediction is correct when it
executes and returns the same rows as the reference query.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build_prompt_messages
from .train import TrainConfig, load_lora_specs


def load_eval_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value.strip()
    return value


def execute_sql(sql: str, context: str) -> list[tuple[Any, ...]] | None:
    """Run `sql` after applying the CREATE/INSERT statements in `context`."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(context)
        cursor = connection.execute(sql)
        return [tuple(_normalize_cell(cell) for cell in row) for row in cursor.fetchall()]
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def clean_generation(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    if ";" in text:
        text = text.split(";", 1)[0] + ";"
    return text.strip()


@torch.no_grad()
def evaluate_model(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    system_prompt: str,
    max_new_tokens: int,
    batch_size: int,
    details_path: str | Path | None = None,
) -> dict[str, Any]:
    gold_results = [execute_sql(row["sql"], row["sql_context"]) for row in rows]
    usable = [(row, gold) for row, gold in zip(rows, gold_results, strict=True) if gold is not None]
    if not usable:
        return {"execution_accuracy": 0.0, "valid_sql_rate": 0.0, "rows": 0}

    prompts = [
        tokenizer.apply_chat_template(
            build_prompt_messages(row, system_prompt), tokenize=False, add_generation_prompt=True
        )
        for row, _ in usable
    ]
    valid = correct = 0
    details: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for (row, gold), completion in zip(
            usable[start : start + batch_size], completions, strict=True
        ):
            generated = clean_generation(completion)
            result = execute_sql(generated, row["sql_context"])
            is_valid = result is not None
            is_correct = int(is_valid and result == gold)
            valid += int(is_valid)
            correct += is_correct
            details.append(
                {
                    "id": row.get("id"),
                    "valid": is_valid,
                    "correct": bool(is_correct),
                    "generated_sql": generated,
                }
            )

    total = len(usable)
    if details_path is not None:
        Path(details_path).write_text(
            json.dumps({"rows": total, "details": details}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return {
        "execution_accuracy": correct / total,
        "valid_sql_rate": valid / total,
        "rows": total,
    }


def run_eval(
    train_config: str | Path,
    lora_config: str | Path,
    *,
    dataset_path: str | Path | None = None,
    output_path: str | Path | None = None,
    details_dir: str | Path | None = None,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    cfg = TrainConfig.from_yaml(train_config)
    specs = load_lora_specs(lora_config)
    rows = load_eval_rows(dataset_path if dataset_path is not None else cfg.eval_dataset_path)
    max_new_tokens = cfg.eval_max_new_tokens if max_new_tokens is None else max_new_tokens
    batch_size = cfg.eval_batch_size if batch_size is None else batch_size
    print(f"SQLite eval on {len(rows)} examples")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cfg.dtype, torch.float32 if device == "cpu" else torch.float16)

    if details_dir is not None:
        Path(details_dir).mkdir(parents=True, exist_ok=True)

    def details_path(name: str) -> str | Path | None:
        if details_dir is None:
            return None
        return Path(details_dir) / f"{name}.json"

    def load_base() -> Any:
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=dtype if device == "cuda" else torch.float32,
            trust_remote_code=cfg.trust_remote_code,
        ).to(device)

    results: dict[str, Any] = {}
    print("\n=== base model ===")
    model = load_base()
    results["base"] = evaluate_model(
        model, tokenizer, rows, cfg.system_prompt, max_new_tokens, batch_size, details_path("base")
    )
    print(json.dumps(results["base"], indent=2))
    del model
    torch.cuda.empty_cache()

    for spec in specs:
        adapter_dir = Path(cfg.output_dir) / spec.name / "adapter"
        if not adapter_dir.exists():
            print(f"\n=== {spec.name}: no adapter at {adapter_dir}, skipping ===")
            results[spec.name] = {"error": "adapter not found"}
            continue
        print(f"\n=== {spec.name} ===")
        base = load_base()
        adapted = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
        results[spec.name] = evaluate_model(
            adapted, tokenizer, rows, cfg.system_prompt, max_new_tokens, batch_size, details_path(spec.name)
        )
        print(json.dumps(results[spec.name], indent=2))
        del adapted, base
        torch.cuda.empty_cache()

    output = Path(output_path) if output_path is not None else Path(cfg.output_dir) / "sqlite_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate SQL outputs against SQLite")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    args = parser.parse_args(argv)
    run_eval(args.train_config, args.lora_config)
