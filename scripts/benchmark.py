"""
Benchmark script for ASR models on Vietnamese speech datasets.

Evaluates openai/whisper-large-v3-turbo on thanhnew2001/VietSuperSpeech
and computes WER (Word Error Rate) and CER (Character Error Rate).
"""

import argparse
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ASR model on Vietnamese speech")
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="thanhnew2001/VietSuperSpeech",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate (None = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vietnamese",
        help="Language hint for Whisper",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSON file path (default: results/<model>_<dataset>_<timestamp>.json)",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    text = text.strip()
    text = text.lower()
    text = " ".join(text.split())
    return text


def download_audio(audio_path: str, dataset_id: str) -> np.ndarray:
    """Download and decode an audio file from HuggingFace Hub."""
    url = hf_hub_url(repo_id=dataset_id, filename=audio_path, repo_type="dataset")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    audio_bytes = io.BytesIO(response.content)
    audio_array, sr = librosa.load(audio_bytes, sr=TARGET_SR, mono=True)
    return audio_array


def load_asr_pipeline(model_id: str):
    """Load the ASR pipeline with the given model."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Loading model {model_id} on {device} with dtype {torch_dtype}")

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
    return pipe


def run_benchmark(pipe, dataset, args):
    """Run inference on the dataset and compute metrics."""
    references = []
    predictions = []
    per_sample_results = []

    total_audio_duration = 0.0
    total_inference_time = 0.0

    num_samples = len(dataset)
    if args.max_samples:
        num_samples = min(num_samples, args.max_samples)

    logger.info(f"Running benchmark on {num_samples} samples...")

    for i in tqdm(range(0, num_samples, args.batch_size), desc="Benchmarking"):
        batch_end = min(i + args.batch_size, num_samples)
        batch_indices = list(range(i, batch_end))
        batch = dataset.select(batch_indices)

        # Download and decode audio from HuggingFace Hub
        audio_arrays = []
        ref_texts = []
        durations = []
        skip_indices = []

        for idx, sample in enumerate(batch):
            try:
                audio_array = download_audio(sample["audio"], args.dataset_id)
                audio_arrays.append(audio_array)
                ref_texts.append(sample["text"])
                durations.append(sample.get("duration", 0.0))
            except Exception as e:
                logger.warning(f"Skipping sample {i + idx}: {e}")
                skip_indices.append(idx)

        if not audio_arrays:
            continue

        start_time = time.time()
        results = pipe(
            audio_arrays,
            batch_size=len(audio_arrays),
            generate_kwargs={"language": args.language, "task": "transcribe"},
        )
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        for j, (result, ref_text, dur) in enumerate(zip(results, ref_texts, durations)):
            pred_text = result["text"]
            ref_norm = normalize_text(ref_text)
            pred_norm = normalize_text(pred_text)

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
    overall_wer = wer(references, predictions)
    overall_cer = cer(references, predictions)

    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else None

    metrics = {
        "overall_wer": round(overall_wer, 4),
        "overall_cer": round(overall_cer, 4),
        "num_samples": num_samples,
        "total_audio_duration_s": round(total_audio_duration, 2),
        "total_inference_time_s": round(total_inference_time, 2),
        "real_time_factor": round(rtf, 4) if rtf else None,
        "batch_size": args.batch_size,
    }

    return metrics, per_sample_results


def save_results(metrics, per_sample_results, args):
    """Save benchmark results to a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_short = args.model_id.replace("/", "_")
    dataset_short = args.dataset_id.replace("/", "_")

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = RESULTS_DIR / f"{model_short}_{dataset_short}_{timestamp}.json"

    result_payload = {
        "metadata": {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "split": args.split,
            "language": args.language,
            "timestamp": timestamp,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
            "max_samples": args.max_samples,
        },
        "metrics": metrics,
        "per_sample_results": per_sample_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Also save a summary file (for quick access / dashboard)
    summary_path = RESULTS_DIR / "latest_summary.json"
    summary = {
        "metadata": result_payload["metadata"],
        "metrics": metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return output_path


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"ASR Benchmark: {args.model_id}")
    logger.info(f"Dataset:       {args.dataset_id} ({args.split})")
    logger.info(f"Max samples:   {args.max_samples or 'all'}")
    logger.info("=" * 60)

    # Load dataset metadata
    logger.info("Loading dataset...")
    dataset = load_dataset(args.dataset_id, split=args.split)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Load model
    pipe = load_asr_pipeline(args.model_id)

    # Run benchmark
    metrics, per_sample_results = run_benchmark(pipe, dataset, args)

    # Print summary
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"WER:  {metrics['overall_wer']:.4f} ({metrics['overall_wer']*100:.2f}%)")
    logger.info(f"CER:  {metrics['overall_cer']:.4f} ({metrics['overall_cer']*100:.2f}%)")
    logger.info(f"Samples evaluated: {metrics['num_samples']}")
    logger.info(f"Total audio:       {metrics['total_audio_duration_s']:.1f}s")
    logger.info(f"Inference time:    {metrics['total_inference_time_s']:.1f}s")
    if metrics["real_time_factor"]:
        logger.info(f"Real-time factor:  {metrics['real_time_factor']:.4f}")
    logger.info("=" * 60)

    # Save results
    output_path = save_results(metrics, per_sample_results, args)
    logger.info(f"Full results: {output_path}")


if __name__ == "__main__":
    main()
