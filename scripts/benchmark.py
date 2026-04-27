"""
ASR benchmark script — evaluates models on Vietnamese speech datasets.

Supports multi-model and multi-dataset benchmarking with automatic
new-dataset detection via a HuggingFace registry.

Usage:
    # Run on all datasets in config
    python scripts/benchmark.py

    # Run on specific datasets
    python scripts/benchmark.py --datasets thanhnew2001/VietSuperSpeech

    # Quick test
    python scripts/benchmark.py --max-samples 5
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="ASR multi-model benchmark")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Specific dataset IDs (overrides config)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def download_audio(audio_path: str, dataset_id: str) -> np.ndarray:
    url = hf_hub_url(repo_id=dataset_id, filename=audio_path, repo_type="dataset")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    arr, _ = librosa.load(io.BytesIO(resp.content), sr=TARGET_SR, mono=True)
    return arr


def load_asr_pipeline(model_id: str, language: str = "vietnamese"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Loading {model_id} on {device} ({dtype})")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                    torch_dtype=dtype, device=device)
    return pipe, model


def unload_model(pipe, model):
    del pipe, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_cost(inference_time_s: float, gpu_pricing: dict) -> dict:
    return {gpu: round((inference_time_s / 3600) * price, 6) for gpu, price in gpu_pricing.items()}


def benchmark_model(model_cfg, dataset, dataset_id, batch_size, max_samples, gpu_pricing):
    model_id = model_cfg["id"]
    language = model_cfg.get("language", "vietnamese")

    logger.info(f"\n{'='*60}\nBenchmarking: {model_id}\n{'='*60}")

    load_start = time.time()
    pipe, model = load_asr_pipeline(model_id, language)
    model_load_time = time.time() - load_start

    num = min(len(dataset), max_samples) if max_samples else len(dataset)
    refs, preds, samples = [], [], []
    total_audio, total_infer = 0.0, 0.0

    for i in tqdm(range(0, num, batch_size), desc=model_id.split("/")[-1]):
        batch = dataset.select(list(range(i, min(i + batch_size, num))))
        arrs, texts, durs = [], [], []
        for sample in batch:
            try:
                arrs.append(download_audio(sample["audio"], dataset_id))
                texts.append(sample["text"])
                durs.append(sample.get("duration", 0.0))
            except Exception as e:
                logger.warning(f"Skip sample: {e}")
        if not arrs:
            continue

        t0 = time.time()
        results = pipe(arrs, batch_size=len(arrs),
                       generate_kwargs={"language": language, "task": "transcribe"})
        infer_t = time.time() - t0
        total_infer += infer_t

        for j, (r, ref, dur) in enumerate(zip(results, texts, durs)):
            rn, pn = normalize_text(ref), normalize_text(r["text"])
            sw = wer(rn, pn) if rn else 0.0
            sc = cer(rn, pn) if rn else 0.0
            total_audio += dur
            refs.append(rn)
            preds.append(pn)
            samples.append({"index": i + j, "reference": rn, "prediction": pn,
                            "wer": round(sw, 4), "cer": round(sc, 4), "duration_s": round(dur, 2)})

    ow = wer(refs, preds) if refs else 1.0
    oc = cer(refs, preds) if refs else 1.0
    rtf = total_infer / total_audio if total_audio > 0 else None
    costs = estimate_cost(total_infer, gpu_pricing)

    logger.info(f"  WER: {ow:.4f} | CER: {oc:.4f} | RTF: {rtf:.4f if rtf else 'N/A'} | Cost(T4): ${costs.get('T4', 0):.6f}")
    unload_model(pipe, model)

    return {
        "model_id": model_id,
        "category": model_cfg.get("category", ""),
        "size_params": model_cfg.get("size_params", ""),
        "notes": model_cfg.get("notes", ""),
        "dataset_id": dataset_id,
        "metrics": {
            "wer": round(ow, 4), "cer": round(oc, 4), "num_samples": len(refs),
            "total_audio_duration_s": round(total_audio, 2),
            "total_inference_time_s": round(total_infer, 2),
            "model_load_time_s": round(model_load_time, 2),
            "real_time_factor": round(rtf, 4) if rtf else None,
            "avg_time_per_sample_s": round(total_infer / len(refs), 4) if refs else None,
        },
        "cost_estimate_usd": costs,
        "per_sample_results": samples,
    }


def run_benchmark(config, args):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_configs = config["models"]
    if args.datasets:
        dataset_configs = [{"id": d, "split": "validation"} for d in args.datasets]
    else:
        dataset_configs = config["datasets"]

    batch_size = args.batch_size or config["benchmark"]["default_batch_size"]
    max_samples = args.max_samples if args.max_samples is not None else config["benchmark"]["default_max_samples"]
    gpu_pricing = config.get("gpu_pricing", {"T4": 0.35, "free_tier": 0.00})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results, leaderboard = [], []

    for ds_cfg in dataset_configs:
        dataset_id = ds_cfg["id"]
        split = ds_cfg.get("split", "validation")
        logger.info(f"\nLoading dataset: {dataset_id} ({split})...")
        dataset = load_dataset(dataset_id, split=split)
        logger.info(f"Dataset loaded: {len(dataset)} samples")

        for model_cfg in model_configs:
            try:
                result = benchmark_model(model_cfg, dataset, dataset_id,
                                         batch_size, max_samples, gpu_pricing)
                all_results.append(result)
                leaderboard.append({
                    "model_id": result["model_id"], "category": result["category"],
                    "size_params": result["size_params"], "dataset_id": result["dataset_id"],
                    "wer": result["metrics"]["wer"], "cer": result["metrics"]["cer"],
                    "rtf": result["metrics"]["real_time_factor"],
                    "inference_time_s": result["metrics"]["total_inference_time_s"],
                    "avg_time_per_sample_s": result["metrics"]["avg_time_per_sample_s"],
                    "cost_T4_usd": result["cost_estimate_usd"].get("T4", 0),
                    "cost_free_tier": result["cost_estimate_usd"].get("free_tier", 0),
                })
            except Exception as e:
                logger.error(f"Failed: {model_cfg['id']} - {e}")
                all_results.append({"model_id": model_cfg["id"], "dataset_id": dataset_id, "error": str(e)})

    leaderboard.sort(key=lambda x: x.get("wer", 999))

    output = {
        "metadata": {
            "timestamp": timestamp, "device": device,
            "torch_version": torch.__version__,
            "num_models": len(model_configs), "num_datasets": len(dataset_configs),
            "max_samples": max_samples, "batch_size": batch_size,
        },
        "leaderboard": leaderboard,
        "detailed_results": all_results,
    }

    path = output_dir / f"benchmark_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to {path}")

    # Print leaderboard
    logger.info("\n" + "=" * 80)
    logger.info("LEADERBOARD (sorted by WER)")
    logger.info("=" * 80)
    for i, e in enumerate(leaderboard, 1):
        rtf_str = f"{e['rtf']:.4f}" if e.get("rtf") else "N/A"
        logger.info(f"{i}. {e['model_id']:<40} WER={e['wer']:.4f}  CER={e['cer']:.4f}  RTF={rtf_str}")
    logger.info("=" * 80)

    return output


def main():
    args = parse_args()
    config = load_config(args.config)
    run_benchmark(config, args)


if __name__ == "__main__":
    main()
