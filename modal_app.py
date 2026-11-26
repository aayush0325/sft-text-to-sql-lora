"""Modal app: LoRA sweep + SQLite eval + loss curves on one cheap GPU.

Run with: modal run modal_app.py
"""

from __future__ import annotations

import modal

# L4 (Ada, 24 GB) is the cheapest Modal GPU with BF16 and fast SDPA/Flash kernels.
GPU = "L4"
TRAIN_CONFIG = "/root/configs/train.yaml"
LORA_CONFIG = "/root/configs/lora.yaml"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync()
    .workdir("/root")
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir("src/sft", "/root/sft")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("data", "/root/data")
)

volume = modal.Volume.from_name("sft-text-to-sql-cache", create_if_missing=True)
app = modal.App("sft-text-to-sql-l4")


@app.function(image=image, volumes={"/cache": volume}, timeout=1800)
def prepare_data() -> str:
    """Download the dataset once into the persistent Modal volume."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="gretelai/synthetic_text_to_sql",
        repo_type="dataset",
        local_dir="/cache/data",
    )
    volume.commit()
    return path


# One function runs the whole sweep: ~3h per config x 5 configs, plus eval.
# The old 3600s timeout killed the job after the first hour.
@app.function(image=image, gpu=GPU, cpu=8, volumes={"/cache": volume}, timeout=600)
def smoke_test() -> str:
    """Two-step run on a tiny slice to validate the fast-training settings."""
    import yaml
    from pathlib import Path

    from sft.train import run_sweep

    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text())
    cfg["data"].update(train_limit=8, eval_limit=4)
    cfg["prompt"]["max_length"] = 512
    cfg["training"].update(
        output_dir="/tmp/smoke", max_steps=2, dataloader_num_workers=0,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
    )
    lora = yaml.safe_load(Path(LORA_CONFIG).read_text())
    lora["loras"] = lora["loras"][:1]
    train_yaml, lora_yaml = Path("/tmp/smoke-train.yaml"), Path("/tmp/smoke-lora.yaml")
    train_yaml.write_text(yaml.safe_dump(cfg))
    lora_yaml.write_text(yaml.safe_dump(lora))
    return str(run_sweep(train_yaml, lora_yaml)[0])


@app.function(image=image, gpu=GPU, cpu=8, volumes={"/cache": volume}, timeout=86400)
def train_eval_report() -> dict[str, object]:
    """Train every LoRA config, evaluate on SQLite, and build the report."""
    from sft.eval import run_eval
    from sft.report import build_report
    from sft.train import run_sweep

    train_results = run_sweep(TRAIN_CONFIG, LORA_CONFIG)
    eval_results = run_eval(TRAIN_CONFIG, LORA_CONFIG)
    report_paths = build_report(TRAIN_CONFIG, LORA_CONFIG)
    volume.commit()

    return {
        "train_results": train_results,
        "eval_results": eval_results,
        "report_paths": report_paths,
    }


@app.function(image=image, gpu=GPU, cpu=8, volumes={"/cache": volume}, timeout=86400)
def eval_only() -> dict[str, object]:
    """Paired SQLite eval of the base model + all adapters on 2,000 test rows."""
    import yaml
    from pathlib import Path

    from sft.eval import run_eval

    cfg = yaml.safe_load(Path(TRAIN_CONFIG).read_text())
    cfg["eval"].update(
        dataset_path="/root/data/eval_set_2000.jsonl",
        max_new_tokens=128,
        batch_size=16,
    )
    train_yaml = Path("/tmp/eval_2000.yaml")
    train_yaml.write_text(yaml.safe_dump(cfg))
    results = run_eval(
        train_yaml,
        LORA_CONFIG,
        output_path="/cache/runs/sqlite_eval_2000.json",
        details_dir="/cache/runs/eval_details_2000",
    )
    volume.commit()
    return {"results": results, "dataset": "/root/data/eval_set_2000.jsonl"}


@app.local_entrypoint()
def main() -> None:
    print("Dataset:", prepare_data.remote())
    print("Results:", train_eval_report.remote())
