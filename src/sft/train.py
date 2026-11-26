"""Simple YAML-driven LoRA fine-tuning sweep for text-to-SQL.

One run per LoRA config in `configs/lora.yaml`. Each run reports eval loss on
the test split; the lowest-loss adapter is copied to `<output_dir>/best_adapter`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from .data import build_prompt_messages, load_hf_dataset


@dataclass(frozen=True)
class TrainConfig:
    model_id: str
    dtype: str
    trust_remote_code: bool
    data_dir: str | None
    cache_dir: str | None
    train_split: str
    eval_split: str
    train_limit: int | None
    eval_limit: int
    num_proc: int
    system_prompt: str
    max_length: int
    output_dir: str
    seed: int
    num_train_epochs: float
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    bf16: bool
    fp16: bool
    tf32: bool
    attn_implementation: str
    padding_free: bool
    dataloader_num_workers: int
    optim: str
    logging_steps: int
    report_to: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    warmup_ratio: float
    scheduler: str
    max_grad_norm: float
    eval_dataset_path: str
    eval_max_new_tokens: int
    eval_batch_size: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        raw = _read_yaml(path)
        model, data, prompt = _section(raw, "model"), _section(raw, "data"), _section(raw, "prompt")
        train, optim = _section(raw, "training"), _section(raw, "optimizer")
        eval_section = _section(raw, "eval")

        warmup_ratio = float(_required(optim, "warmup_ratio"))
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        betas = _required(optim, "betas")
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("optimizer.betas must be a two-element list")

        return cls(
            model_id=str(_required(model, "id")),
            dtype=str(model.get("dtype", "bfloat16")),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            data_dir=None if data.get("data_dir") is None else str(data["data_dir"]),
            cache_dir=None if data.get("cache_dir") is None else str(data["cache_dir"]),
            train_split=str(data.get("train_split", "train")),
            eval_split=str(data.get("eval_split", "test")),
            train_limit=None if data.get("train_limit") is None else int(data["train_limit"]),
            eval_limit=int(data.get("eval_limit", 500)),
            num_proc=int(data.get("num_proc", 1)),
            system_prompt=str(_required(prompt, "system")),
            max_length=int(prompt.get("max_length", 1024)),
            output_dir=str(_required(train, "output_dir")),
            seed=int(train.get("seed", 42)),
            num_train_epochs=float(train.get("num_train_epochs", 1.0)),
            max_steps=int(train.get("max_steps", -1)),
            per_device_train_batch_size=int(train.get("per_device_train_batch_size", 4)),
            per_device_eval_batch_size=int(train.get("per_device_eval_batch_size", 8)),
            gradient_accumulation_steps=int(train.get("gradient_accumulation_steps", 4)),
            gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
            bf16=bool(train.get("bf16", True)),
            fp16=bool(train.get("fp16", False)),
            tf32=bool(train.get("tf32", False)),
            attn_implementation=str(train.get("attn_implementation", "sdpa")),
            padding_free=bool(train.get("padding_free", False)),
            dataloader_num_workers=int(train.get("dataloader_num_workers", 0)),
            optim=str(train.get("optim", "adamw_torch_fused")),
            logging_steps=int(train.get("logging_steps", 20)),
            report_to=str(train.get("report_to", "none")),
            learning_rate=float(_required(optim, "learning_rate")),
            weight_decay=float(_required(optim, "weight_decay")),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(_required(optim, "eps")),
            warmup_ratio=warmup_ratio,
            scheduler=str(optim.get("scheduler", "cosine")),
            max_grad_norm=float(_required(optim, "max_grad_norm")),
            eval_dataset_path=str(_required(eval_section, "dataset_path")),
            eval_max_new_tokens=int(_required(eval_section, "max_new_tokens")),
            eval_batch_size=int(_required(eval_section, "batch_size")),
        )


@dataclass(frozen=True)
class LoRASpec:
    name: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    use_rslora: bool = False
    use_dora: bool = False

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.target_modules),
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )


class LossLogger(TrainerCallback):
    """Append training loss points to a JSON file for plotting later."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.points: list[dict[str, float]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs or "loss" not in logs:
            return
        self.points.append(
            {
                "step": float(state.global_step),
                "loss": float(logs["loss"]),
                "learning_rate": float(logs.get("learning_rate", 0.0)),
            }
        )
        self.path.write_text(json.dumps(self.points, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RunResult:
    name: str
    eval_loss: float
    trainable_params: int
    adapter_dir: str


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = raw.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"Missing or invalid '{name}' section in config")
    return section


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_lora_specs(path: str | Path) -> list[LoRASpec]:
    entries = _read_yaml(path).get("loras")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty 'loras' list")

    specs = []
    for entry in entries:
        targets = _required(entry, "target_modules")
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"LoRA {entry.get('name')!r}: target_modules must be a non-empty list")
        specs.append(
            LoRASpec(
                name=str(_required(entry, "name")),
                r=int(_required(entry, "r")),
                lora_alpha=int(_required(entry, "lora_alpha")),
                lora_dropout=float(_required(entry, "lora_dropout")),
                target_modules=[str(module) for module in targets],
                use_rslora=bool(entry.get("use_rslora", False)),
                use_dora=bool(entry.get("use_dora", False)),
            )
        )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("LoRA run names must be unique")
    return specs


def prepare_datasets(cfg: TrainConfig) -> tuple[Dataset, Dataset]:
    train_raw = load_hf_dataset(cfg.train_split, cache_dir=cfg.cache_dir, data_dir=cfg.data_dir)
    eval_raw = load_hf_dataset(cfg.eval_split, cache_dir=cfg.cache_dir, data_dir=cfg.data_dir)
    if not isinstance(train_raw, Dataset) or not isinstance(eval_raw, Dataset):
        raise TypeError("Training requires non-streaming datasets")

    train_raw = train_raw.shuffle(seed=cfg.seed)
    if cfg.train_limit is not None:
        train_raw = train_raw.select(range(min(cfg.train_limit, len(train_raw))))
    eval_raw = eval_raw.shuffle(seed=cfg.seed).select(range(min(cfg.eval_limit, len(eval_raw))))

    def format_example(example: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "prompt": build_prompt_messages(example, cfg.system_prompt),
            "completion": [{"role": "assistant", "content": str(example["sql"])}],
        }

    train = train_raw.map(
        format_example,
        num_proc=cfg.num_proc,
        remove_columns=train_raw.column_names,
        desc="Formatting train examples",
    )
    eval_set = eval_raw.map(
        format_example,
        num_proc=cfg.num_proc,
        remove_columns=eval_raw.column_names,
        desc="Formatting eval examples",
    )
    return train, eval_set


def step_counts(cfg: TrainConfig, num_train_examples: int) -> tuple[int, int]:
    world_size = max(int(os.getenv("WORLD_SIZE", "1")), 1)
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(num_train_examples / effective_batch)
    total = cfg.max_steps if cfg.max_steps > 0 else math.ceil(cfg.num_train_epochs * steps_per_epoch)
    warmup = min(total, round(cfg.warmup_ratio * total))
    return total, warmup


def build_sft_config(cfg: TrainConfig, run_dir: Path, warmup_steps: int) -> SFTConfig:
    return SFTConfig(
        output_dir=str(run_dir),
        seed=cfg.seed,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        eval_strategy="no",
        report_to=cfg.report_to,
        max_length=cfg.max_length,
        packing=False,
        completion_only_loss=True,
        assistant_only_loss=False,
        dataset_num_proc=cfg.num_proc,
        dataloader_num_workers=cfg.dataloader_num_workers,
        trust_remote_code=cfg.trust_remote_code,
        # SDPA dispatches to FlashAttention kernels on Ada/Ampere GPUs. No
        # separate flash-attn build is needed, and BF16 is native on the L4.
        model_init_kwargs={"dtype": cfg.dtype, "attn_implementation": cfg.attn_implementation},
        # packing-free batches: no padding tokens are fed to the model.
        padding_free=cfg.padding_free,
        # Keep prompt/completion columns so the collator can mask the prompt.
        remove_unused_columns=False,
        # Fused AdamW with linear warmup followed by cosine decay.
        optim=cfg.optim,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.betas[0],
        adam_beta2=cfg.betas[1],
        adam_epsilon=cfg.eps,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.scheduler,
        warmup_steps=warmup_steps,
    )


def run_one(
    spec: LoRASpec,
    cfg: TrainConfig,
    tokenizer: Any,
    train: Dataset,
    eval_set: Dataset,
    warmup_steps: int,
) -> RunResult:
    run_dir = Path(cfg.output_dir) / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== LoRA {spec.name!r}: r={spec.r} alpha={spec.lora_alpha} targets={spec.target_modules}")

    trainer = SFTTrainer(
        model=cfg.model_id,
        args=build_sft_config(cfg, run_dir, warmup_steps),
        train_dataset=train,
        eval_dataset=eval_set,
        processing_class=tokenizer,
        peft_config=spec.to_peft_config(),
        callbacks=[LossLogger(run_dir / "history.json")],
    )
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    trainer.train()
    eval_loss = float(trainer.evaluate()["eval_loss"])

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return RunResult(
        name=spec.name,
        eval_loss=eval_loss,
        trainable_params=trainable_params,
        adapter_dir=str(adapter_dir),
    )


def run_sweep(train_config: str | Path, lora_config: str | Path) -> list[RunResult]:
    cfg = TrainConfig.from_yaml(train_config)
    specs = load_lora_specs(lora_config)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train, eval_set = prepare_datasets(cfg)
    _, warmup_steps = step_counts(cfg, len(train))
    print(f"Running {len(specs)} LoRA configs on {len(train)} examples | warmup_steps={warmup_steps}")

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.json"

    # Restart-idempotent: a finished config is never trained twice. This matters
    # on Modal, where a lost connection or preempted container can end a run
    # after several hours of training.
    saved = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    by_name = {
        item["name"]: RunResult(**item)
        for item in saved
        if isinstance(item, dict) and item.get("name") in {spec.name for spec in specs}
    }

    results: list[RunResult] = []
    for spec in specs:
        adapter_dir = output_root / spec.name / "adapter"
        if spec.name in by_name and (adapter_dir / "adapter_model.safetensors").exists():
            print(f"\n=== LoRA {spec.name!r}: already trained, reusing saved adapter ===")
            result = by_name[spec.name]
        else:
            result = run_one(spec, cfg, tokenizer, train, eval_set, warmup_steps)
        by_name[spec.name] = result
        results = [by_name[item.name] for item in specs if item.name in by_name]
        results_path.write_text(
            json.dumps([asdict(item) for item in results], indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(asdict(result), indent=2))

    best = min(results, key=lambda item: item.eval_loss)
    best_dir = output_root / "best_adapter"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best.adapter_dir, best_dir)
    (output_root / "best.json").write_text(
        json.dumps(asdict(best) | {"adapter_dir": str(best_dir)}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nBest LoRA: {best.name} (eval_loss={best.eval_loss:.4f})")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.2-1B-Instruct with a LoRA sweep")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--lora-config", default="configs/lora.yaml")
    args = parser.parse_args(argv)
    run_sweep(args.train_config, args.lora_config)
