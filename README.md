# ASR Benchmark: Whisper Large V3 Turbo on VietSuperSpeech

Benchmark pipeline for evaluating ASR models on Vietnamese speech datasets. Computes **WER** (Word Error Rate) and **CER** (Character Error Rate), stores results to HuggingFace, and provides a Streamlit dashboard for visualization.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dashboard]"
```

### 2. Run Benchmark

```bash
# Full validation set (6,750 samples) - requires GPU for reasonable speed
python scripts/benchmark.py

# Quick test with a subset
python scripts/benchmark.py --max-samples 50

# Custom model/dataset
python scripts/benchmark.py \
    --model-id openai/whisper-large-v3-turbo \
    --dataset-id thanhnew2001/VietSuperSpeech \
    --split validation \
    --batch-size 16 \
    --language vietnamese
```

### 3. Push Results to HuggingFace

```bash
# Login first
huggingface-cli login

# Push results
python scripts/push_to_hf.py --hf-repo-id your-username/asr-benchmark-results
```

### 4. View Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard can load results from:
- A **HuggingFace dataset repo** (enter the repo ID in the sidebar)
- A **local JSON file** (upload via the sidebar)

## Project Structure

```
asr-benchmark/
├── scripts/
│   ├── benchmark.py        # Main benchmark script
│   └── push_to_hf.py       # Push results to HuggingFace Hub
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── results/                # Benchmark results (generated)
├── pyproject.toml
└── README.md
```

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate - measures word-level transcription accuracy |
| **CER** | Character Error Rate - measures character-level transcription accuracy |
| **RTF** | Real-Time Factor - ratio of inference time to audio duration |

## Model

- **[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)**: Pruned version of Whisper Large V3 with 4 decoder layers (vs 32), offering ~3-4x faster inference with minor quality loss.

## Dataset

- **[thanhnew2001/VietSuperSpeech](https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech)**: Vietnamese speech dataset with ~67.4k samples (60.7k train, 6.75k validation).

## Dashboard Alternatives

If you prefer other frameworks, here are supported alternatives:

| Framework | Best For | Integration |
|-----------|----------|-------------|
| **Streamlit** | Quick local dashboards | Included (`dashboard/app.py`) |
| **Gradio** | HuggingFace Spaces | Use `gr.Dataframe` + `gr.Plot` |
| **Weights & Biases** | Experiment tracking | `wandb.log(metrics)` |
| **MLflow** | ML lifecycle management | `mlflow.log_metrics(metrics)` |
| **HuggingFace Dataset Viewer** | Zero-config viewing | Automatic with pushed datasets |

## Requirements

- Python 3.10+
- GPU recommended for full benchmark (CPU works but is very slow)
- HuggingFace account (for pushing results)
