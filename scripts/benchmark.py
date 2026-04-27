"""
Multi-model ASR benchmark script.

Evaluates multiple ASR models on Vietnamese speech datasets,
computes WER/CER/RTF and estimates cost. Supports loading config
from config.yaml for easy model/dataset management.
"""

import argparse
import gc
import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import requests
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_url
from jiwer import cer, wer
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model ASR benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Specific model IDs to benchmark (overrides config). Space-separated.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Specific dataset IDs to benchmark (overrides config). Space-separated.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (overrides config). None = all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for results.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    return " ".join(text.strip().lower().split())


def download_audio(audio_path: str, dataset_id: str) -> np.ndarray:
    """Download and decode an audio file from HuggingFace Hub."""
    url = hf_hub_url(repo_id=dataset_id, filename=audio_path, repo_type="dataset")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    audio_array, _ = librosa.load(io.BytesIO(response.content), sr=TARGET_SR, mono=True)
    return audio_array


def load_asr_pipeline(model_id: str, language: str = "vietnamese"):
    """Load ASR pipeline for a given model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading model {model_id} on {device} ({torch_dtype})")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe, model


def unload_model(pipe, model):
    """Free GPU memory after benchmarking a model."""
    del pipe
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_cost(inference_time_s: float, gpu_pricing: dict) -> dict:
    """Estimate cloud GPU cost based on inference time."""
    costs = {}
    for gpu_name, price_per_hour in gpu_pricing.items():
        cost = (inference_time_s / 3600) * price_per_hour
        costs[gpu_name] = round(cost, 6)
    return costs


def benchmark_single_model(
    model_config: dict,
    dataset,
    dataset_id: str,
    batch_size: int,
    max_samples: int | None,
    gpu_pricing: dict,
) -> dict:
    """Benchmark a single model on a single dataset."""
    model_id = model_config["id"]
    language = model_config.get("language", "vietnamese")

    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmarking: {model_id}")
    logger.info(f"{'='*60}")

    # Load model
    load_start = time.time()
    pipe, model = load_asr_pipeline(model_id, language)
    model_load_time = time.time() - load_start

    # Determine number of samples
    num_samples = len(dataset)
    if max_samples:
        num_samples = min(num_samples, max_samples)

    references = []
    predictions = []
    per_sample_results = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    logger.info(f"Running on {num_samples} samples (batch_size={batch_size})...")

    for i in tqdm(range(0, num_samples, batch_size), desc=model_id.split("/")[-1]):
        batch_end = min(i + batch_size, num_samples)
        batch = dataset.select(list(range(i, batch_end)))

        audio_arrays = []
        ref_texts = []
        durations = []

        for idx, sample in enumerate(batch):
            try:
                audio_array = download_audio(sample["audio"], dataset_id)
                audio_arrays.append(audio_array)
                ref_texts.append(sample["text"])
                durations.append(sample.get("duration", 0.0))
            except Exception as e:
                logger.warning(f"Skipping sample {i + idx}: {e}")

        if not audio_arrays:
            continue

        start_time = time.time()
        results = pipe(
            audio_arrays,
            batch_size=len(audio_arrays),
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        for j, (result, ref_text, dur) in enumerate(zip(results, ref_texts, durations)):
            ref_norm = normalize_text(ref_text)
            pred_norm = normalize_text(result["text"])
            sample_wer = wer(ref_norm, pred_norm) if ref_norm else 0.0
            sample_cer = cer(ref_norm, pred_norm) if ref_norm else 0.0
            total_audio_duration += dur
            references.append(ref_norm)
            predictions.append(pred_norm)
            per_sample_results.append({
                "index": i + j,
                "reference": ref_norm,
                "prediction": pred_norm,
                "wer": round(sample_wer, 4),
                "cer": round(sample_cer, 4),
                "duration_s": round(dur, 2),
            })

    # Compute aggregate metrics
    overall_wer = wer(references, predictions) if references else 1.0
    overall_cer = cer(references, predictions) if references else 1.0
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else None

    # Estimate costs
    costs = estimate_cost(total_inference_time, gpu_pricing)

    result = {
        "model_id": model_id,
        "category": model_config.get("category", "unknown"),
        "size_params": model_config.get("size_params", "unknown"),
        "notes": model_config.get("notes", ""),
        "dataset_id": dataset_id,
        "metrics": {
            "wer": round(overall_wer, 4),
            "cer": round(overall_cer, 4),
            "num_samples": len(references),
            "total_audio_duration_s": round(total_audio_duration, 2),
            "total_inference_time_s": round(total_inference_time, 2),
            "model_load_time_s": round(model_load_time, 2),
            "real_time_factor": round(rtf, 4) if rtf else None,
            "avg_time_per_sample_s": round(total_inference_time / len(references), 4) if references else None,
        },
        "cost_estimate_usd": costs,
        "per_sample_results": per_sample_results,
    }

    # Log summary
    logger.info(f"  WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    logger.info(f"  CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    logger.info(f"  Inference: {total_inference_time:.1f}s | RTF: {rtf:.4f}" if rtf else "")
    logger.info(f"  Cost (T4): ${costs.get('T4', 0):.6f}")

    # Free GPU memory
    unload_model(pipe, model)

    return result


def run_full_benchmark(config: dict, args):
    """Run benchmark across all models and datasets."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models to benchmark
    if args.models:
        model_configs = [{"id": m, "language": "vietnamese", "category": "custom"} for m in args.models]
    else:
        model_configs = config["models"]

    # Determine datasets to benchmark
    if args.datasets:
        dataset_configs = [{"id": d, "split": "validation"} for d in args.datasets]
    else:
        dataset_configs = config["datasets"]

    batch_size = args.batch_size or config["benchmark"]["default_batch_size"]
    max_samples = args.max_samples if args.max_samples is not None else config["benchmark"]["default_max_samples"]
    gpu_pricing = config.get("gpu_pricing", {"T4": 0.35, "free_tier": 0.00})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = []
    leaderboard = []

    for ds_config in dataset_configs:
        dataset_id = ds_config["id"]
        split = ds_config.get("split", "validation")

        logger.info(f"\nLoading dataset: {dataset_id} ({split})...")
        dataset = load_dataset(dataset_id, split=split)
        logger.info(f"Dataset loaded: {len(dataset)} samples")

        for model_config in model_configs:
            try:
                result = benchmark_single_model(
                    model_config=model_config,
                    dataset=dataset,
                    dataset_id=dataset_id,
                    batch_size=batch_size,
                    max_samples=max_samples,
                    gpu_pricing=gpu_pricing,
                )
                all_results.append(result)

                # Add to leaderboard
                leaderboard.append({
                    "model_id": result["model_id"],
                    "category": result["category"],
                    "size_params": result["size_params"],
                    "dataset_id": result["dataset_id"],
                    "wer": result["metrics"]["wer"],
                    "cer": result["metrics"]["cer"],
                    "rtf": result["metrics"]["real_time_factor"],
                    "inference_time_s": result["metrics"]["total_inference_time_s"],
                    "avg_time_per_sample_s": result["metrics"]["avg_time_per_sample_s"],
                    "cost_T4_usd": result["cost_estimate_usd"].get("T4", 0),
                    "cost_free_tier": result["cost_estimate_usd"].get("free_tier", 0),
                })

            except Exception as e:
                logger.error(f"Failed to benchmark {model_config['id']}: {e}")
                all_results.append({
                    "model_id": model_config["id"],
                    "dataset_id": dataset_id,
                    "error": str(e),
                })

    # Sort leaderboard by WER (best first)
    leaderboard.sort(key=lambda x: x.get("wer", 999))

    # Save full results
    full_output = {
        "metadata": {
            "timestamp": timestamp,
            "device": device,
            "torch_version": torch.__version__,
            "num_models": len(model_configs),
            "num_datasets": len(dataset_configs),
            "max_samples": max_samples,
            "batch_size": batch_size,
        },
        "leaderboard": leaderboard,
        "detailed_results": all_results,
    }

    output_path = output_dir / f"benchmark_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    logger.info(f"\nFull results saved to {output_path}")

    # Save latest leaderboard for dashboard
    leaderboard_path = output_dir / "latest_leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": full_output["metadata"], "leaderboard": leaderboard}, f, ensure_ascii=False, indent=2)
    logger.info(f"Leaderboard saved to {leaderboard_path}")

    # Print leaderboard
    logger.info("\n" + "=" * 80)
    logger.info("LEADERBOARD (sorted by WER, best first)")
    logger.info("=" * 80)
    logger.info(f"{'Rank':<5} {'Model':<40} {'WER':>8} {'CER':>8} {'RTF':>8} {'Cost($)':>10}")
    logger.info("-" * 80)
    for i, entry in enumerate(leaderboard, 1):
        rtf_str = f"{entry['rtf']:.4f}" if entry.get('rtf') else "N/A"
        logger.info(
            f"{i:<5} {entry['model_id']:<40} {entry['wer']:>8.4f} {entry['cer']:>8.4f} "
            f"{rtf_str:>8} {entry['cost_T4_usd']:>10.6f}"
        )
    logger.info("=" * 80)

    return full_output


def main():
    args = parse_args()
    config = load_config(args.config)

    logger.info("=" * 60)
    logger.info("ASR Multi-Model Benchmark")
    logger.info("=" * 60)

    run_full_benchmark(config, args)


if __name__ == "__main__":
    main()
