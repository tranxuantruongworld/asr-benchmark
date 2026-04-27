"""
Push benchmark results to HuggingFace Hub.

Creates/updates a dataset repo with leaderboard and per-sample results.
Designed to be called automatically from the Kaggle notebook.
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Push ASR benchmark results to HuggingFace")
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--hf-repo-id", type=str, required=True)
    parser.add_argument("--private", action="store_true", default=False)
    return parser.parse_args()


def find_latest_results():
    files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No results in {RESULTS_DIR}")
    return files[0]


def push_results(results: dict, repo_id: str, private: bool = False):
    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

    metadata = results["metadata"]
    leaderboard = results["leaderboard"]

    # Leaderboard split
    lb_keys = ["model_id", "category", "size_params", "dataset_id",
               "wer", "cer", "rtf", "inference_time_s",
               "avg_time_per_sample_s", "cost_T4_usd", "cost_free_tier"]
    lb_ds = Dataset.from_dict({k: [e.get(k) for e in leaderboard] for k in lb_keys})

    # Per-sample split
    all_samples = []
    for detail in results.get("detailed_results", []):
        if "error" in detail:
            continue
        for s in detail.get("per_sample_results", []):
            all_samples.append({
                "model_id": detail["model_id"],
                "dataset_id": detail["dataset_id"],
                **s,
            })

    sample_keys = ["model_id", "dataset_id", "index", "reference", "prediction", "wer", "cer", "duration_s"]
    if all_samples:
        samples_ds = Dataset.from_dict({k: [s.get(k) for s in all_samples] for k in sample_keys})
    else:
        samples_ds = Dataset.from_dict({k: [] for k in sample_keys})

    DatasetDict({"leaderboard": lb_ds, "per_sample": samples_ds}).push_to_hub(repo_id, private=private)
    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")

    # Upload raw JSON
    results_json = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=results_json,
        path_in_repo="latest_benchmark.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update benchmark results",
    )

    # Generate README
    lb_table = "| Rank | Model | Category | WER | CER | RTF | Cost (T4) |\n"
    lb_table += "|------|-------|----------|-----|-----|-----|-----------|\n"
    for i, e in enumerate(leaderboard, 1):
        rtf_str = f"{e['rtf']:.4f}" if e.get("rtf") else "N/A"
        lb_table += f"| {i} | `{e['model_id']}` | {e['category']} | {e['wer']:.4f} | {e['cer']:.4f} | {rtf_str} | ${e.get('cost_T4_usd', 0):.6f} |\n"

    # Collect unique datasets
    ds_list = list({e["dataset_id"] for e in leaderboard if e.get("dataset_id")})

    readme = f"""---
language: [vi]
tags: [asr, benchmark, whisper, vietnamese, phowhisper]
pretty_name: "ASR Benchmark Results"
---

# ASR Benchmark Results

Comparing ASR models on Vietnamese speech datasets.

**Last updated**: {metadata['timestamp']}
**Device**: {metadata['device']}

## Leaderboard

{lb_table}

## Datasets Benchmarked

{chr(10).join(f'- `{d}`' for d in ds_list)}

## Usage

```python
from datasets import load_dataset

# Load leaderboard
lb = load_dataset("{repo_id}", split="leaderboard")
print(lb.to_pandas().sort_values("wer"))

# Load per-sample results
samples = load_dataset("{repo_id}", split="per_sample")
```
"""
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update README",
    )
    logger.info("README updated")


def main():
    args = parse_args()
    path = Path(args.results_file) if args.results_file else find_latest_results()
    logger.info(f"Loading {path}")
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    push_results(results, args.hf_repo_id, args.private)


if __name__ == "__main__":
    main()
