# sft

LoRA fine-tuning sweep for text-to-SQL on `gretelai/synthetic_text_to_sql`, with
SQLite execution-based evaluation and loss curves. The completed five-config
sweep ran on a Modal L4 (24 GB, native BF16) using SDPA/FlashAttention kernels,
padding-free batches, fused AdamW, and no activation checkpointing.

## Results

**[RESULTS.md](RESULTS.md)** has the full comparison table, metric explanations,
artifact inventory and reproduction steps. In the rigorous 2,000-row paired
eval, `mlp_r32` was nominally best (78.3% SQL execution accuracy vs 53.9% for
the base model) but statistically tied with the other larger adapters; see the
[paired comparison table](results/runs/large_eval/comparison.md) for Wilson CIs,
McNemar tests and bootstrap differences.

Training loss curves: [results/runs/loss_curves.png](results/runs/loss_curves.png)
(generated from `results/runs/*/history.json`).

Reports, curves and per-row eval predictions are tracked in `results/runs/`;
only the adapter binaries are gitignored. The best adapter (`mlp_r32`) is
published on Hugging Face as
[`aayush0325/llama-3.2-1b-text2sql-lora-r32`](https://huggingface.co/aayush0325/llama-3.2-1b-text2sql-lora-r32).

## Layout

- `src/sft/data.py` — `TextToSQLSample` dataclass and dataset loader (HF Hub or local parquet).
- `src/sft/train.py` — YAML-driven LoRA sweep; AdamW with linear warmup + cosine decay.
- `src/sft/eval.py` — executes generated and reference SQL on in-memory SQLite.
- `src/sft/report.py` — comparison tables (`results.csv`, `results.md`) and `loss_curves.png`.
- `configs/train.yaml`, `configs/lora.yaml` — all sweep settings (no CLI flags needed).
- `modal_app.py` — data prep (CPU), a 2-step smoke test, and the train + eval + report function (L4).
- `data/eval_set.jsonl` — 32 hand-curated rows from the original SQLite eval.
- `data/eval_set_2000.jsonl` — deterministic 2,000-row paired eval set (`scripts/make_eval_set.py`).
- `scripts/compare_eval.py` — Wilson CIs, paired McNemar tests and bootstrap differences.

## Reproduce

Settings are YAML-driven. Deploy the app, optionally smoke-test, then spawn the
full sweep as a fire-and-forget call so local restarts cannot kill it:

```bash
modal deploy modal_app.py
modal run modal_app.py::smoke_test
python - <<'PY'
import modal
fn = modal.Function.from_name("sft-text-to-sql-l4", "train_eval_report")
call = fn.spawn()
print(call.object_id, call.get_dashboard_url())
PY
```

Useful commands while it runs:

```bash
modal app list
modal app logs sft-text-to-sql-l4 -f      # follow logs (Ctrl-C only stops following)
modal app stats sft-text-to-sql-l4 --since 1h
```

When it finishes, the function commits `/cache/runs` to the Volume; download
with `mkdir -p results && modal volume get <volume-name> /runs results`.

The sweep is restart-idempotent: any config with a saved adapter is skipped.
