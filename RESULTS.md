# Results — text-to-SQL LoRA sweep

## Experiment

- **Dataset:** `gretelai/synthetic_text_to_sql` (100k train / 5,851 test rows).
- **Base model:** `unsloth/Llama-3.2-1B-Instruct` (public mirror, no gated license).
- **Hardware:** one Modal **L4** (24 GB, native BF16), ~$0.80/hr.
- **Training:** one epoch, effective batch size 16, AdamW (fused, β = 0.9/0.95,
  wd 0.01), 3% linear warmup then cosine decay, lr 2e-4, max length 1024.
- **Speed settings:** BF16, SDPA/FlashAttention kernels, padding-free batches,
  no activation checkpointing, 8 train / 16 eval per device.
- **Five LoRA configs** from `configs/lora.yaml`, trained sequentially on the
  same data and optimizer schedule.

## Comparison table

| model | r | alpha | target modules | trainable params | final train loss | eval loss | execution accuracy | valid SQL rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | – | – | – | – | – | – | 0.5000 | 0.8438 |
| attn_r8 | 8 | 16 | q,k,v,o | 1,703,936 | 0.2188 | 0.2181 | 0.6875 | 0.9062 |
| attn_mlp_r16 | 16 | 32 | q,k,v,o + gate,up,down | 11,272,192 | 0.1944 | 0.1919 | 0.7188 | 0.9375 |
| **mlp_r32** | **32** | **64** | gate,up,down | 15,728,640 | 0.1927 | 0.1890 | **0.8438** | **0.9688** |
| attn_r64_rslora | 64 | 32 | q,k,v,o | 13,631,488 | 0.1941 | 0.1944 | 0.7812 | 0.9062 |
| all_linear_r16_dora | 16 | 32 | q,k,v,o + gate,up,down | 11,649,024 | 0.1940 | 0.1915 | 0.7812 | 0.9375 |

**Best config: `mlp_r32`** — 84.4% execution accuracy (27 / 32) versus 50%
for the base model, and the lowest eval loss (0.1890). Its adapter is also
copied to `results/runs/best_adapter` and published to Hugging Face as
[`aayush0325/llama-3.2-1b-text2sql-lora-r32`](https://huggingface.co/aayush0325/llama-3.2-1b-text2sql-lora-r32).

## What the metrics mean

- **Final train loss** — next-token cross-entropy on the completion (SQL) tokens
  of the training split at the last optimizer step. Lower is better.
- **Eval loss** — the same loss computed on a held-out slice of the test split
  (500 rows), with labels masked on the prompt. Lower is better.
- **Valid SQL rate** — fraction of generated completions that parse and execute
  successfully against the in-memory SQLite database. The rest are syntax or
  schema errors.
- **Execution accuracy** — fraction of generated SQL queries that both execute
  **and** return exactly the same rows as the reference query. This is the
  metric that matters for text-to-SQL: a low loss does not guarantee correct
  SQL, and a syntactically valid query can still return the wrong rows.

The SQLite eval uses the 32 hand-curated rows in `data/eval_set.jsonl`. With
only 32 rows, one query changes accuracy by ~3.1 points, so treat small gaps
as noise; the base → tuned gap (50% → 84%) is well beyond that.

## Loss curves and tables

- [Loss curves](results/runs/loss_curves.png) — training loss vs optimizer step for all five configs.
- [Comparison table (Markdown)](results/runs/results.md)
- [Comparison table (CSV)](results/runs/results.csv)
- [SQLite execution eval](results/runs/sqlite_eval.json)
- [Raw training results](results/runs/results.json)
- [Best adapter metadata](results/runs/best.json)

## Local artifact inventory

All artifacts were downloaded from the Modal Volume into `results/runs/`:

```
results/runs/
├── attn_r8/                    {adapter/, history.json}
├── attn_mlp_r16/               {adapter/, history.json}
├── mlp_r32/                    {adapter/, history.json}
├── attn_r64_rslora/            {adapter/, history.json}
├── all_linear_r16_dora/        {adapter/, history.json}
├── best_adapter/               copy of the mlp_r32 adapter
├── best.json, results.json, results.csv, results.md
├── sqlite_eval.json
└── loss_curves.png
```

Each `adapter/` contains `adapter_model.safetensors`, `adapter_config.json`
and `README.md`. `history.json` holds per-step loss and learning-rate points.
`results/` is gitignored — the binaries stay on disk but are not tracked.

The Modal apps and the `sft-text-to-sql-cache` Volume were deleted after the
download; these files are the only remaining copy of the artifacts.

## How to reproduce

Settings live in `configs/train.yaml` and `configs/lora.yaml`; the app and its
`@app.function` definitions live in `modal_app.py`. To rerun from scratch:

```bash
modal deploy modal_app.py                          # deploy the app once
modal run modal_app.py::smoke_test                 # optional 2-step check
python - <<'PY'
import modal
fn = modal.Function.from_name("sft-text-to-sql-l4", "train_eval_report")
call = fn.spawn()                                  # fire-and-forget, survives local restarts
print(call.object_id, call.get_dashboard_url())
PY
```

Then follow logs with `modal app logs sft-text-to-sql-l4 -f`. When the function
finishes it commits `/cache/runs` to the Volume; download with:

```bash
mkdir -p results && modal volume get <volume> /runs results
```

The sweep is restart-idempotent: a config with a saved adapter is skipped on a
rerun. Change `GPU` in `modal_app.py` and `dtype`/flags in `configs/train.yaml`
to try different hardware or precision.
