---
title: ASR Benchmark Dashboard
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
license: mit
tags:
  - asr
  - benchmark
  - whisper
  - vietnamese
---

# ASR Benchmark Dashboard

Visualize benchmark results for ASR models on Vietnamese speech datasets.

## How to use

1. Enter a HuggingFace Dataset Repo ID in the sidebar (e.g., `your-username/asr-benchmark-results`)
2. Or upload a benchmark results JSON file

## Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/new-space
2. Select **Streamlit** as the SDK
3. Upload the contents of this `dashboard/` folder
4. The app will auto-deploy
