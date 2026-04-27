"""
Push benchmark results to HuggingFace Hub.

Creates a dataset repository with benchmark metrics that can be
viewed via HuggingFace's Dataset Viewer or consumed by a dashboard.
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
        help="Path to the benchmark results JSON file. If omitted, uses the latest in results/.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID to push results to (e.g., 'username/asr-benchmark-results')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the HuggingFace repo private",
    )
    return parser.parse_args()


def find_latest_results():
    """Find the latest benchmark results file."""
    results_files = sorted(RESULTS_DIR.glob("openai_*.json"), reverse=True)
    if not results_files:
        raise FileNotFoundError(f"No results files found in {RESULTS_DIR}")
    return results_files[0]


def load_results(results_file: Path) -> dict:
    """Load benchmark results from a JSON file."""
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def push_metrics_dataset(results: dict, repo_id: str, private: bool):
    """Push benchmark metrics as a HuggingFace dataset."""
    api = HfApi()

    # Create or get the repo
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    logger.info(f"Repository ready: https://huggingface.co/datasets/{repo_id}")

    metadata = results["metadata"]
    metrics = results["metrics"]

    # 1. Push summary metrics as a dataset
    summary_data = {
        "model_id": [metadata["model_id"]],
        "dataset_id": [metadata["dataset_id"]],
        "split": [metadata["split"]],
        "language": [metadata["language"]],
        "timestamp": [metadata["timestamp"]],
        "device": [metadata["device"]],
        "wer": [metrics["overall_wer"]],
        "cer": [metrics["overall_cer"]],
        "num_samples": [metrics["num_samples"]],
        "total_audio_duration_s": [metrics["total_audio_duration_s"]],
        "total_inference_time_s": [metrics["total_inference_time_s"]],
        "real_time_factor": [metrics.get("real_time_factor")],
        "batch_size": [metrics["batch_size"]],
    }

    summary_features = Features({
        "model_id": Value("string"),
        "dataset_id": Value("string"),
        "split": Value("string"),
        "language": Value("string"),
        "timestamp": Value("string"),
        "device": Value("string"),
        "wer": Value("float64"),
        "cer": Value("float64"),
        "num_samples": Value("int64"),
        "total_audio_duration_s": Value("float64"),
        "total_inference_time_s": Value("float64"),
        "real_time_factor": Value("float64"),
        "batch_size": Value("int64"),
    })

    summary_ds = Dataset.from_dict(summary_data, features=summary_features)

    # 2. Push per-sample results
    per_sample = results.get("per_sample_results", [])
    if per_sample:
        sample_data = {
            "index": [s["index"] for s in per_sample],
            "reference": [s["reference"] for s in per_sample],
            "prediction": [s["prediction"] for s in per_sample],
            "wer": [s["wer"] for s in per_sample],
            "cer": [s["cer"] for s in per_sample],
            "duration_s": [s["duration_s"] for s in per_sample],
        }

        sample_features = Features({
            "index": Value("int64"),
            "reference": Value("string"),
            "prediction": Value("string"),
            "wer": Value("float64"),
            "cer": Value("float64"),
            "duration_s": Value("float64"),
        })

        samples_ds = Dataset.from_dict(sample_data, features=sample_features)
    else:
        samples_ds = Dataset.from_dict({
            "index": [], "reference": [], "prediction": [],
            "wer": [], "cer": [], "duration_s": [],
        })

    ds_dict = DatasetDict({
        "summary": summary_ds,
        "per_sample": samples_ds,
    })

    ds_dict.push_to_hub(repo_id, private=private)
    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")

    # 3. Upload the raw JSON results file as well
    results_file_path = RESULTS_DIR / "latest_summary.json"
    if results_file_path.exists():
        api.upload_file(
            path_or_fileobj=str(results_file_path),
            path_in_repo="latest_summary.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Uploaded latest_summary.json to repo")

    # 4. Create a README / dataset card
    readme_content = f"""---
language:
- vi
tags:
- asr
- benchmark
- whisper
- vietnamese
pretty_name: "ASR Benchmark Results"
---

# ASR Benchmark Results

## Model: `{metadata['model_id']}`
## Dataset: `{metadata['dataset_id']}` (split: `{metadata['split']}`)

### Metrics

| Metric | Value |
|--------|-------|
| **WER** | {metrics['overall_wer']:.4f} ({metrics['overall_wer']*100:.2f}%) |
| **CER** | {metrics['overall_cer']:.4f} ({metrics['overall_cer']*100:.2f}%) |
| Samples | {metrics['num_samples']} |
| Audio Duration | {metrics['total_audio_duration_s']:.1f}s |
| Inference Time | {metrics['total_inference_time_s']:.1f}s |
| Real-Time Factor | {metrics.get('real_time_factor', 'N/A')} |
| Device | {metadata['device']} |
| Batch Size | {metrics['batch_size']} |

### Splits

- **`summary`**: Aggregate metrics (1 row per benchmark run)
- **`per_sample`**: Per-sample predictions, references, WER, and CER

### Usage

```python
from datasets import load_dataset

# Load summary metrics
summary = load_dataset("{repo_id}", split="summary")
print(summary[0])

# Load per-sample results
samples = load_dataset("{repo_id}", split="per_sample")
print(samples[0])
```

### Timestamp
{metadata['timestamp']}
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info("README.md uploaded")


def main():
    args = parse_args()

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = find_latest_results()

    logger.info(f"Loading results from {results_path}")
    results = load_results(results_path)

    logger.info(f"Pushing to HuggingFace: {args.hf_repo_id}")
    push_metrics_dataset(results, args.hf_repo_id, args.private)

    logger.info("Done!")


if __name__ == "__main__":
    main()
