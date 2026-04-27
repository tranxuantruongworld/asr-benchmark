# ASR Benchmark: Whisper Large V3 Turbo on VietSuperSpeech

Benchmark pipeline for evaluating ASR models on Vietnamese speech datasets. Computes **WER** (Word Error Rate) and **CER** (Character Error Rate), stores results to HuggingFace, and provides a Streamlit dashboard for visualization.

## Run on Cloud (Recommended)

### Google Colab (Free GPU)

The easiest way to run the benchmark - no local setup needed:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tranxuantruongworld/asr-benchmark/blob/init-setup/notebooks/benchmark_colab.ipynb)

1. Click the badge above to open the notebook in Google Colab
2. Select **Runtime > Change runtime type > T4 GPU**
3. Run all cells
4. Results are saved as JSON and can be pushed to HuggingFace

### HuggingFace Spaces (Dashboard)

Deploy the dashboard to HuggingFace Spaces for free:

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Upload the contents of the `dashboard/` folder (`app.py`, `requirements.txt`, `README.md`)
4. The dashboard auto-deploys and is publicly accessible

## Local Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
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

### 4. View Dashboard Locally

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
│   ├── benchmark.py           # Main benchmark script
│   └── push_to_hf.py          # Push results to HuggingFace Hub
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   ├── requirements.txt       # Dashboard dependencies (for HF Spaces)
│   └── README.md              # HF Spaces metadata
├── notebooks/
│   └── benchmark_colab.ipynb  # Google Colab notebook (free GPU)
├── results/                   # Benchmark results (generated, gitignored)
├── requirements.txt           # All dependencies
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

## Cloud Deployment Options

| Platform | What | Cost | How |
|----------|------|------|-----|
| **Google Colab** | Run benchmark with free T4 GPU | Free | [Open notebook](https://colab.research.google.com/github/tranxuantruongworld/asr-benchmark/blob/init-setup/notebooks/benchmark_colab.ipynb) |
| **HuggingFace Spaces** | Deploy Streamlit dashboard | Free | Upload `dashboard/` folder to a new Space |
| **HuggingFace Dataset Viewer** | View results (zero-config) | Free | Automatic after `push_to_hf.py` |
| **Weights & Biases** | Experiment tracking | Free tier | `wandb.log(metrics)` |
| **MLflow** | ML lifecycle management | Self-hosted | `mlflow.log_metrics(metrics)` |

## Requirements

- Python 3.10+
- GPU recommended for full benchmark (CPU works but is very slow)
- HuggingFace account (for pushing results)
