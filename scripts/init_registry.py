"""
Initialize the HuggingFace registry and results repos.

Run this once to set up the automated benchmark flow:
    python scripts/init_registry.py --registry-repo your-name/asr-registry --results-repo your-name/asr-results

This creates:
  1. Registry repo (datasets.json) — where you add new datasets
  2. Results repo (benchmarked.json) — where benchmark results are stored
"""

import argparse
import json
import logging
from datetime import datetime, timezone

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Initialize benchmark registry")
    parser.add_argument("--registry-repo", type=str, required=True,
                        help="HF repo for dataset registry (e.g. username/asr-benchmark-registry)")
    parser.add_argument("--results-repo", type=str, required=True,
                        help="HF repo for results (e.g. username/asr-benchmark-results)")
    args = parser.parse_args()

    api = HfApi()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Create registry repo
    create_repo(repo_id=args.registry_repo, repo_type="dataset", exist_ok=True)
    logger.info(f"Registry repo: https://huggingface.co/datasets/{args.registry_repo}")

    initial_datasets = [
        {"id": "thanhnew2001/VietSuperSpeech", "split": "validation", "added": today},
    ]
    api.upload_file(
        path_or_fileobj=json.dumps(initial_datasets, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo="datasets.json",
        repo_id=args.registry_repo, repo_type="dataset",
        commit_message="Initialize dataset registry",
    )

    registry_readme = f"""---
language: [vi]
tags: [asr, benchmark, registry]
pretty_name: ASR Benchmark Dataset Registry
---

# ASR Benchmark Dataset Registry

Add your datasets here to be automatically benchmarked by the Kaggle notebook.

## How to add a dataset

1. Upload your dataset to HuggingFace (must have `audio` and `text` columns)
2. Edit `datasets.json` in this repo and add an entry:

```json
{{
    "id": "your-username/your-dataset",
    "split": "test",
    "added": "{today}"
}}
```

3. The Kaggle notebook runs weekly and will auto-detect new datasets.

## Current datasets

See `datasets.json` for the full list.
""".encode("utf-8")

    api.upload_file(
        path_or_fileobj=registry_readme,
        path_in_repo="README.md",
        repo_id=args.registry_repo, repo_type="dataset",
        commit_message="Add README",
    )

    # 2. Create results repo
    create_repo(repo_id=args.results_repo, repo_type="dataset", exist_ok=True)
    logger.info(f"Results repo: https://huggingface.co/datasets/{args.results_repo}")

    api.upload_file(
        path_or_fileobj=json.dumps({}, indent=2).encode("utf-8"),
        path_in_repo="benchmarked.json",
        repo_id=args.results_repo, repo_type="dataset",
        commit_message="Initialize benchmarked tracker",
    )

    results_readme = f"""---
language: [vi]
tags: [asr, benchmark, whisper, vietnamese]
pretty_name: ASR Benchmark Results
---

# ASR Benchmark Results

Benchmark results will appear here after the first Kaggle run.

## Usage

```python
from datasets import load_dataset

lb = load_dataset("{args.results_repo}", split="leaderboard")
print(lb.to_pandas().sort_values("wer"))
```
""".encode("utf-8")

    api.upload_file(
        path_or_fileobj=results_readme,
        path_in_repo="README.md",
        repo_id=args.results_repo, repo_type="dataset",
        commit_message="Add README",
    )

    logger.info("\nSetup complete!")
    logger.info(f"\nRegistry: https://huggingface.co/datasets/{args.registry_repo}")
    logger.info(f"Results:  https://huggingface.co/datasets/{args.results_repo}")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Upload benchmark_kaggle.ipynb to Kaggle")
    logger.info(f"2. Add Kaggle secrets: HF_TOKEN, HF_REGISTRY_REPO={args.registry_repo}, HF_RESULTS_REPO={args.results_repo}")
    logger.info(f"3. Schedule weekly runs")
    logger.info(f"4. To add datasets: edit datasets.json in {args.registry_repo}")


if __name__ == "__main__":
    main()
