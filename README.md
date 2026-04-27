# ASR Benchmark — Automated Pipeline

Automated ASR model comparison on Vietnamese speech datasets. Runs on **Kaggle** (free GPU), detects new datasets automatically, pushes results to **HuggingFace**, and displays on a **Streamlit dashboard**.

## Flow

```
1. You upload dataset to HuggingFace ──► Registry repo (datasets.json)
2. Kaggle runs weekly (scheduled)    ──► Detects new datasets
3. Benchmark 2 models on new data    ──► vinai/PhoWhisper-large + openai/whisper-large-v3-turbo
4. Results pushed to HuggingFace     ──► Results repo (leaderboard + per-sample)
5. Dashboard auto-updates            ──► Streamlit on HuggingFace Spaces
```

## Models

| Model | Category | Params |
|-------|----------|--------|
| `vinai/PhoWhisper-large` | Vietnamese-specific | 1550M |
| `openai/whisper-large-v3-turbo` | Multilingual | 809M |

## Quick Start

### 1. Initialize HuggingFace repos (one-time)

```bash
pip install -r requirements.txt
huggingface-cli login

python scripts/init_registry.py \
    --registry-repo your-username/asr-benchmark-registry \
    --results-repo your-username/asr-benchmark-results
```

This creates:
- **Registry repo**: where you add new datasets to benchmark
- **Results repo**: where benchmark results are stored

### 2. Set up Kaggle notebook (one-time)

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Upload `notebooks/benchmark_kaggle.ipynb`
3. Settings:
   - **Accelerator**: GPU T4 x2
   - **Internet**: On
   - **Secrets**: Add these 3 secrets:
     - `HF_TOKEN` — your HuggingFace write token
     - `HF_REGISTRY_REPO` — e.g. `your-username/asr-benchmark-registry`
     - `HF_RESULTS_REPO` — e.g. `your-username/asr-benchmark-results`
4. **Run All** to test
5. **Schedule**: File → Schedule → **Weekly**

### 3. Add new datasets

Edit `datasets.json` in your registry repo on HuggingFace:

```json
[
    {"id": "thanhnew2001/VietSuperSpeech", "split": "validation", "added": "2025-01-01"},
    {"id": "your-username/your-new-dataset", "split": "test", "added": "2025-04-27"}
]
```

The next Kaggle run will automatically detect and benchmark only the new datasets.

**Dataset requirements**: Must have `audio` (file path) and `text` (transcription) columns.

### 4. Deploy dashboard (one-time)

Deploy to HuggingFace Spaces for free:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as SDK
3. Upload files from `dashboard/` folder (`app.py`, `requirements.txt`, `README.md`)
4. Dashboard auto-deploys and loads results from your results repo

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate (lower = better) |
| **CER** | Character Error Rate (lower = better) |
| **RTF** | Real-Time Factor (< 1.0 = faster than real-time) |
| **Cost** | Estimated GPU cost per run |

## Project Structure

```
asr-benchmark/
├── config.yaml                  # Models & datasets config
├── scripts/
│   ├── benchmark.py             # Multi-model benchmark (CLI)
│   ├── push_to_hf.py            # Push results to HuggingFace
│   ├── init_registry.py         # One-time: init HF registry + results repos
│   └── registry.py              # Dataset registry management
├── notebooks/
│   └── benchmark_kaggle.ipynb   # Kaggle notebook (auto-scheduled)
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   ├── requirements.txt         # Dashboard deps (for HF Spaces)
│   └── README.md                # HF Spaces metadata
├── results/                     # Local results (gitignored)
├── requirements.txt
└── README.md
```

## Cost

| Component | Cost |
|-----------|------|
| Kaggle GPU (T4, 30h/week) | **Free** |
| HuggingFace Hub (datasets) | **Free** |
| HuggingFace Spaces (dashboard) | **Free** |
| GitHub (code) | **Free** |
| **Total** | **$0** |
