"""Loader for `gretelai/synthetic_text_to_sql`.

The dataset is fetched from the Hugging Face Hub by default, so nothing needs
to be baked into a container image. On Modal, either point `HF_HOME` at a
mounted Volume (the `hf` CLI and `datasets` then share one cache), or
pre-download with `hf download` and pass the resulting directory as
`data_dir`. `HF_TOKEN` is read from the environment (e.g. a Modal Secret).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

from datasets import Dataset, IterableDataset, load_dataset

DATASET_ID = "gretelai/synthetic_text_to_sql"

DATA_FILES = {
    "train": "synthetic_text_to_sql_train.snappy.parquet",
    "test": "synthetic_text_to_sql_test.snappy.parquet",
}

Split = Literal["train", "test"]


@dataclass(frozen=True, slots=True)
class TextToSQLSample:
    """One text-to-SQL example, with field names matching the HF dataset."""

    id: int
    domain: str
    domain_description: str
    sql_complexity: str
    sql_complexity_description: str
    sql_task_type: str
    sql_task_type_description: str
    sql_prompt: str
    sql_context: str
    sql: str
    sql_explanation: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> TextToSQLSample:
        """Build a sample from a Hugging Face row or plain dict."""
        return cls(**{field.name: row[field.name] for field in fields(cls)})

    def to_dict(self) -> dict[str, Any]:
        """Return the sample as a plain dict (e.g. to feed back to a Dataset)."""
        return asdict(self)


def build_prompt_messages(row: Mapping[str, Any], system_prompt: str) -> list[dict[str, str]]:
    """Build the same prompt the trainer sees, for generation/eval."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"### Schema (SQLite)\n{row['sql_context']}\n\n"
                f"### Question\n{row['sql_prompt']}"
            ),
        },
    ]


def load_hf_dataset(
    split: Split = "train",
    *,
    cache_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    token: str | None = None,
    revision: str | None = None,
    streaming: bool = False,
) -> Dataset | IterableDataset:
    """Load a split for use with a trainer.

    Args:
        split: `"train"` (100k rows) or `"test"` (~5.8k rows).
        cache_dir: Arrow cache path. When unset, `datasets` uses
            `HF_DATASETS_CACHE`/`HF_HOME`.
        data_dir: Directory containing the parquet files, e.g. the target of
            `hf download gretelai/synthetic_text_to_sql --repo-type dataset
            --local-dir /cache/data`. Skips Hub access entirely; useful with a
            pre-populated Modal Volume. `token` and `revision` are ignored.
        token: Hugging Face token; defaults to the `HF_TOKEN` env var. The
            dataset is public, so this is normally unnecessary.
        revision: Optional commit/tag to pin the dataset version.
        streaming: Yield rows without materializing the split; useful for
            quick smoke tests on ephemeral machines.
    """
    resolved_cache = str(cache_dir) if cache_dir is not None else None
    if data_dir is not None:
        directory = Path(data_dir).expanduser()
        return load_dataset(
            "parquet",
            data_files={name: str(directory / filename) for name, filename in DATA_FILES.items()},
            split=split,
            cache_dir=resolved_cache,
            streaming=streaming,
        )
    return load_dataset(
        DATASET_ID,
        split=split,
        cache_dir=resolved_cache,
        token=token if token is not None else os.getenv("HF_TOKEN"),
        revision=revision,
        streaming=streaming,
    )


def load_samples(
    split: Split = "train",
    **kwargs: Any,
) -> Iterator[TextToSQLSample]:
    """Yield `TextToSQLSample` objects from a split.

    Accepts the same keyword arguments as `load_hf_dataset`. Iterating lazily
    keeps memory use low when training on a GPU host with limited CPU RAM.
    """
    for row in load_hf_dataset(split, **kwargs):
        yield TextToSQLSample.from_row(row)
