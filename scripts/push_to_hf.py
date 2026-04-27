"""
Push multi-model benchmark results to HuggingFace Hub.

Creates a dataset repository with leaderboard and per-model metrics
viewable via HuggingFace's Dataset Viewer or a dashboard.
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Push ASR benchmark results to HuggingFace")
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to benchmark results JSON. Default: latest in results/.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/asr-benchmark-results')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def find_latest_results():
    results_files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
    if not results_files:
        raise FileNotFoundError(f"No results files found in {RESULTS_DIR}")
    return results_files[0]


def push_results(results: dict, repo_id: str, private: bool):
    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    logger.info(f"Repository: https://huggingface.co/datasets/{repo_id}")

    metadata = results["metadata"]
    leaderboard = results["leaderboard"]

    # 1. Push leaderboard
    lb_data = {k: [entry.get(k) for entry in leaderboard] for k in [
        "model_id", "category", "size_params", "dataset_id",
        "wer", "cer", "rtf", "inference_time_s",
        "avg_time_per_sample_s", "cost_T4_usd", "cost_free_tier",
    ]}
    lb_ds = Dataset.from_dict(lb_data)

    # 2. Push per-sample results (all models combined)
    all_samples = []
    for detail in results.get("detailed_results", []):
        if "error" in detail:
            continue
        model_id = detail["model_id"]
        dataset_id = detail["dataset_id"]
        for sample in detail.get("per_sample_results", []):
            all_samples.append({
                "model_id": model_id,
                "dataset_id": dataset_id,
                **sample,
            })

    if all_samples:
        sample_data = {k: [s.get(k) for s in all_samples] for k in [
            "model_id", "dataset_id", "index", "reference",
            "prediction", "wer", "cer", "duration_s",
        ]}
        samples_ds = Dataset.from_dict(sample_data)
    else:
        samples_ds = Dataset.from_dict({
            "model_id": [], "dataset_id": [], "index": [],
            "reference": [], "prediction": [], "wer": [], "cer": [], "duration_s": [],
        })

    ds_dict = DatasetDict({"leaderboard": lb_ds, "per_sample": samples_ds})
    ds_dict.push_to_hub(repo_id, private=private)
    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")

    # 3. Upload raw JSON
    results_file = find_latest_results()
    api.upload_file(
        path_or_fileobj=str(results_file),
        path_in_repo="latest_benchmark.json",
        repo_id=repo_id,
        repo_type="dataset",
    )

    # 4. Generate README
    lb_table = "| Rank | Model | Category | WER | CER | RTF | Cost (T4) |\n"
    lb_table += "|------|-------|----------|-----|-----|-----|-----------|\n"
    for i, entry in enumerate(leaderboard, 1):
        rtf_str = f"{entry['rtf']:.4f}" if entry.get('rtf') else "N/A"
        lb_table += (
            f"| {i} | `{entry['model_id']}` | {entry['category']} | "
            f"{entry['wer']:.4f} | {entry['cer']:.4f} | {rtf_str} | "
            f"${entry.get('cost_T4_usd', 0):.6f} |\n"
        )

    readme = f"""---
language:
- vi
tags:
- asr
- benchmark
- whisper
- vietnamese
- phowhisper
pretty_name: "ASR Multi-Model Benchmark Results"
---

# ASR Multi-Model Benchmark Results

Comparing {len(leaderboard)} ASR models on Vietnamese speech datasets.

## Leaderboard (sorted by WER)

{lb_table}

## Splits

- **`leaderboard`**: Model comparison table (1 row per model)
- **`per_sample`**: Per-sample predictions for all models

## Usage

```python
from datasets import load_dataset

# Load leaderboard
lb = load_dataset("{repo_id}", split="leaderboard")
print(lb.to_pandas().sort_values("wer"))

# Load per-sample results for a specific model
samples = load_dataset("{repo_id}", split="per_sample")
df = samples.to_pandas()
model_df = df[df["model_id"] == "vinai/PhoWhisper-large"]
```

## Metadata

- **Timestamp**: {metadata['timestamp']}
- **Device**: {metadata['device']}
- **Models tested**: {metadata['num_models']}
- **Datasets**: {metadata['num_datasets']}
"""

    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("README.md uploaded")


def main():
    args = parse_args()
    results_path = Path(args.results_file) if args.results_file else find_latest_results()
    logger.info(f"Loading results from {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    push_results(results, args.hf_repo_id, args.private)
    logger.info("Done!")


if __name__ == "__main__":
    main()
