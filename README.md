# ASR Multi-Model Benchmark

Compare **11 ASR models** on Vietnamese speech datasets — automatically, on free cloud GPU.

## Models Compared

### Vietnamese-specific (VinAI PhoWhisper)
| Model | Params | Fine-tuned for Vietnamese |
|-------|--------|--------------------------|
| `vinai/PhoWhisper-large` | 1550M | Whisper Large V2 |
| `vinai/PhoWhisper-medium` | 769M | Whisper Medium |
| `vinai/PhoWhisper-small` | 244M | Whisper Small |
| `vinai/PhoWhisper-base` | 74M | Whisper Base |
| `vinai/PhoWhisper-tiny` | 39M | Whisper Tiny |

### General Multilingual (OpenAI Whisper)
| Model | Params | Notes |
|-------|--------|-------|
| `openai/whisper-large-v3` | 1550M | Best general quality |
| `openai/whisper-large-v3-turbo` | 809M | 3x faster, minor quality loss |
| `openai/whisper-medium` | 769M | Good balance |
| `openai/whisper-small` | 244M | Fast |
| `openai/whisper-base` | 74M | Very fast |
| `openai/whisper-tiny` | 39M | Fastest |

## Run on Kaggle (Recommended)

**Free T4 GPU, 30h/week, auto-scheduling:**

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Upload `notebooks/benchmark_kaggle.ipynb`
3. Settings → **Accelerator: GPU T4 x2** → **Internet: On**
4. **Run All** cells
5. (Optional) **Schedule**: File → Schedule → Weekly

### Auto-scheduling on Kaggle
- Kaggle lets you schedule notebooks to run automatically (daily/weekly)
- Results are saved as Kaggle output files
- Set `PUSH_TO_HF = True` + add `HF_TOKEN` in Kaggle Secrets to auto-push results to HuggingFace

## Adding More Datasets

Edit the `DATASETS` list in the Kaggle notebook (cell 3) or in `config.yaml`:

```python
DATASETS = [
    {"id": "thanhnew2001/VietSuperSpeech", "split": "validation", "name": "VietSuperSpeech"},
    {"id": "your-dataset/name", "split": "test", "name": "YourDataset"},
]
```

Or in `config.yaml`:
```yaml
datasets:
  - id: "thanhnew2001/VietSuperSpeech"
    split: "validation"
  - id: "your-new-dataset"
    split: "test"
```

## View Results

### Option 1: HuggingFace Dataset Viewer (zero-config)
After pushing results, view at `https://huggingface.co/datasets/your-username/asr-benchmark-results`

### Option 2: Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

Deploy to **HuggingFace Spaces** for free: upload `dashboard/` folder to a new Streamlit Space.

### Option 3: Kaggle Output
Results are saved as JSON + CSV + charts in the Kaggle notebook output.

## Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate (lower = better) |
| **CER** | Character Error Rate (lower = better) |
| **RTF** | Real-Time Factor (< 1.0 = faster than real-time) |
| **Cost** | Estimated GPU cost per benchmark run |

## Project Structure

```
asr-benchmark/
├── config.yaml                  # Models & datasets configuration
├── scripts/
│   ├── benchmark.py             # Multi-model benchmark (CLI)
│   └── push_to_hf.py            # Push results to HuggingFace
├── notebooks/
│   └── benchmark_kaggle.ipynb   # Kaggle notebook (free GPU + scheduling)
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   ├── requirements.txt         # Dashboard deps (for HF Spaces)
│   └── README.md                # HF Spaces metadata
├── results/                     # Generated results (gitignored)
├── requirements.txt
└── README.md
```

## Cost Summary

| Platform | What | Cost |
|----------|------|------|
| **Kaggle** | Run benchmark (T4 GPU, 30h/week) | **Free** |
| **HuggingFace Spaces** | Host dashboard | **Free** |
| **HuggingFace Dataset Viewer** | View results | **Free** |
| **GitHub** | Host code | **Free** |

Total cost: **$0**
