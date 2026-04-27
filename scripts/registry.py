"""
Dataset registry for ASR benchmark.

Manages which datasets need to be benchmarked by checking a HuggingFace
registry repo. Users add datasets to the registry, and the benchmark
notebook detects new (un-benchmarked) datasets automatically.

Registry format (datasets.json on HF Hub):
[
    {"id": "thanhnew2001/VietSuperSpeech", "split": "validation", "added": "2025-01-01"},
    {"id": "user/NewDataset", "split": "test", "added": "2025-04-27"}
]

Results tracking (benchmarked.json on HF Hub):
{
    "thanhnew2001/VietSuperSpeech:validation": "20250427T120000Z",
    ...
}
"""

import json
import logging
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download, upload_file

logger = logging.getLogger(__name__)


def get_registered_datasets(registry_repo: str) -> list[dict]:
    """Fetch the list of datasets from the registry repo."""
    try:
        path = hf_hub_download(
            repo_id=registry_repo,
            filename="datasets.json",
            repo_type="dataset",
        )
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load registry from {registry_repo}: {e}")
        return []


def get_benchmarked_datasets(results_repo: str) -> dict:
    """Fetch which datasets have already been benchmarked."""
    try:
        path = hf_hub_download(
            repo_id=results_repo,
            filename="benchmarked.json",
            repo_type="dataset",
        )
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def find_new_datasets(
    registry_repo: str,
    results_repo: str,
    fallback_datasets: list[dict] | None = None,
) -> list[dict]:
    """Find datasets that are registered but not yet benchmarked."""
    registered = get_registered_datasets(registry_repo)
    if not registered and fallback_datasets:
        registered = fallback_datasets

    benchmarked = get_benchmarked_datasets(results_repo)

    new_datasets = []
    for ds in registered:
        key = f"{ds['id']}:{ds.get('split', 'validation')}"
        if key not in benchmarked:
            new_datasets.append(ds)

    return new_datasets


def mark_dataset_benchmarked(
    results_repo: str,
    dataset_id: str,
    split: str,
    timestamp: str | None = None,
):
    """Mark a dataset as benchmarked in the results repo."""
    benchmarked = get_benchmarked_datasets(results_repo)
    key = f"{dataset_id}:{split}"
    benchmarked[key] = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    content = json.dumps(benchmarked, indent=2).encode("utf-8")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=content,
        path_in_repo="benchmarked.json",
        repo_id=results_repo,
        repo_type="dataset",
        commit_message=f"Mark {key} as benchmarked",
    )


def init_registry(registry_repo: str, initial_datasets: list[dict] | None = None):
    """Initialize a registry repo with a datasets.json file."""
    api = HfApi()
    api.create_repo(repo_id=registry_repo, repo_type="dataset", exist_ok=True)

    datasets = initial_datasets or []
    content = json.dumps(datasets, indent=2, ensure_ascii=False).encode("utf-8")
    api.upload_file(
        path_or_fileobj=content,
        path_in_repo="datasets.json",
        repo_id=registry_repo,
        repo_type="dataset",
        commit_message="Initialize dataset registry",
    )

    readme = f"""---
language: [vi]
tags: [asr, benchmark, registry]
pretty_name: "ASR Benchmark Dataset Registry"
---

# ASR Benchmark Dataset Registry

Add your datasets here to be automatically benchmarked.

## How to add a dataset

Edit `datasets.json` and add an entry:

```json
{{
    "id": "your-username/your-dataset",
    "split": "test",
    "added": "{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
}}
```

The Kaggle benchmark notebook runs weekly and will automatically detect
and benchmark any new datasets added here.

## Current datasets

```json
{json.dumps(datasets, indent=2, ensure_ascii=False)}
```
""".encode("utf-8")

    api.upload_file(
        path_or_fileobj=readme,
        path_in_repo="README.md",
        repo_id=registry_repo,
        repo_type="dataset",
        commit_message="Add README",
    )
    logger.info(f"Registry initialized at https://huggingface.co/datasets/{registry_repo}")
