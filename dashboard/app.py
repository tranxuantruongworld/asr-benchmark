"""
Streamlit dashboard to display ASR benchmark metrics.

Loads results from a HuggingFace dataset repo and visualizes them.
Can also load from a local JSON file.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from datasets import load_dataset

st.set_page_config(
    page_title="ASR Benchmark Dashboard",
    page_icon="🎙️",
    layout="wide",
)

st.title("ASR Benchmark Dashboard")
st.markdown("Evaluate and compare ASR models on Vietnamese speech datasets.")


# --- Data Source Selection ---
st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Load results from:",
    ["HuggingFace Dataset", "Local JSON File"],
)

summary_df = None
samples_df = None

if source == "HuggingFace Dataset":
    repo_id = st.sidebar.text_input(
        "HuggingFace Dataset Repo ID",
        value="",
        placeholder="username/asr-benchmark-results",
    )
    if repo_id:
        try:
            with st.spinner("Loading from HuggingFace..."):
                summary_ds = load_dataset(repo_id, split="summary")
                samples_ds = load_dataset(repo_id, split="per_sample")
                summary_df = summary_ds.to_pandas()
                samples_df = samples_ds.to_pandas()
            st.sidebar.success(f"Loaded from {repo_id}")
        except Exception as e:
            st.sidebar.error(f"Error loading dataset: {e}")
    else:
        st.info("Enter a HuggingFace Dataset Repo ID in the sidebar to get started.")

elif source == "Local JSON File":
    uploaded = st.sidebar.file_uploader("Upload benchmark results JSON", type=["json"])
    if uploaded:
        data = json.load(uploaded)
        metadata = data.get("metadata", {})
        metrics = data.get("metrics", {})
        per_sample = data.get("per_sample_results", [])

        summary_df = pd.DataFrame([{**metadata, **metrics}])
        if per_sample:
            samples_df = pd.DataFrame(per_sample)
        st.sidebar.success("Loaded from uploaded file")
    else:
        # Try loading latest_summary.json from default results dir
        default_path = Path(__file__).resolve().parent.parent / "results" / "latest_summary.json"
        if default_path.exists():
            with open(default_path) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            metrics = data.get("metrics", {})
            summary_df = pd.DataFrame([{**metadata, **metrics}])
            st.sidebar.info(f"Loaded from {default_path}")
        else:
            st.info("Upload a benchmark results JSON file to get started.")


# --- Dashboard Display ---
if summary_df is not None and not summary_df.empty:
    st.header("Summary Metrics")

    row = summary_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wer_val = row.get("wer", row.get("overall_wer", 0))
        st.metric("Word Error Rate (WER)", f"{wer_val:.4f}", f"{wer_val*100:.2f}%")
    with col2:
        cer_val = row.get("cer", row.get("overall_cer", 0))
        st.metric("Character Error Rate (CER)", f"{cer_val:.4f}", f"{cer_val*100:.2f}%")
    with col3:
        st.metric("Samples", int(row.get("num_samples", 0)))
    with col4:
        rtf = row.get("real_time_factor")
        st.metric("Real-Time Factor", f"{rtf:.4f}" if rtf and rtf == rtf else "N/A")

    # Model & dataset info
    st.subheader("Benchmark Info")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"**Model:** `{row.get('model_id', 'N/A')}`")
        st.write(f"**Dataset:** `{row.get('dataset_id', 'N/A')}`")
        st.write(f"**Split:** `{row.get('split', 'N/A')}`")
    with info_col2:
        st.write(f"**Language:** `{row.get('language', 'N/A')}`")
        st.write(f"**Device:** `{row.get('device', 'N/A')}`")
        st.write(f"**Timestamp:** `{row.get('timestamp', 'N/A')}`")

    # Performance details
    st.subheader("Performance")
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Audio Duration", f"{row.get('total_audio_duration_s', 0):.1f}s")
    with perf_col2:
        st.metric("Inference Time", f"{row.get('total_inference_time_s', 0):.1f}s")
    with perf_col3:
        st.metric("Batch Size", int(row.get("batch_size", 0)))

    # --- Per-Sample Results ---
    if samples_df is not None and not samples_df.empty:
        st.header("Per-Sample Results")

        # WER distribution
        st.subheader("WER Distribution")
        st.bar_chart(samples_df["wer"].value_counts().sort_index().head(50))

        # CER distribution
        st.subheader("CER Distribution")
        st.bar_chart(samples_df["cer"].value_counts().sort_index().head(50))

        # Error analysis
        st.subheader("Error Analysis")
        tab1, tab2, tab3 = st.tabs(["Worst Samples", "Best Samples", "All Samples"])

        with tab1:
            worst = samples_df.nlargest(20, "wer")
            st.dataframe(
                worst[["index", "reference", "prediction", "wer", "cer", "duration_s"]],
                use_container_width=True,
            )

        with tab2:
            best = samples_df.nsmallest(20, "wer")
            st.dataframe(
                best[["index", "reference", "prediction", "wer", "cer", "duration_s"]],
                use_container_width=True,
            )

        with tab3:
            st.dataframe(
                samples_df[["index", "reference", "prediction", "wer", "cer", "duration_s"]],
                use_container_width=True,
            )

        # Summary stats
        st.subheader("Statistical Summary")
        stats_cols = ["wer", "cer", "duration_s"]
        st.dataframe(samples_df[stats_cols].describe(), use_container_width=True)

else:
    st.markdown("""
    ### Getting Started

    1. **Run the benchmark:**
       ```bash
       python scripts/benchmark.py --max-samples 100
       ```

    2. **Push results to HuggingFace:**
       ```bash
       python scripts/push_to_hf.py --hf-repo-id your-username/asr-benchmark-results
       ```

    3. **View results here** by entering the HuggingFace repo ID or uploading the JSON file.

    ---

    **Supported Frameworks for Dashboard:**
    - [Streamlit](https://streamlit.io) (this dashboard)
    - [Gradio](https://gradio.app) - great for HuggingFace Spaces
    - [Weights & Biases](https://wandb.ai) - for experiment tracking
    - [MLflow](https://mlflow.org) - for ML lifecycle management
    """)
