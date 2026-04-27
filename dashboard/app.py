"""
Streamlit dashboard for multi-model ASR benchmark results.

Displays leaderboard, model comparisons, cost analysis,
and per-sample error analysis. Loads from HuggingFace or local JSON.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ASR Benchmark Dashboard",
    page_icon="🎙️",
    layout="wide",
)

st.title("ASR Multi-Model Benchmark Dashboard")
st.markdown("Compare ASR models on Vietnamese speech datasets — WER, CER, speed, and cost.")

# --- Data Source ---
st.sidebar.header("Data Source")
source = st.sidebar.radio("Load results from:", ["HuggingFace Dataset", "Local JSON File"])

leaderboard_df = None
samples_df = None
metadata = None

if source == "HuggingFace Dataset":
    repo_id = st.sidebar.text_input(
        "HuggingFace Dataset Repo ID",
        value="",
        placeholder="username/asr-benchmark-results",
    )
    if repo_id:
        try:
            from datasets import load_dataset
            with st.spinner("Loading from HuggingFace..."):
                lb_ds = load_dataset(repo_id, split="leaderboard")
                samples_ds = load_dataset(repo_id, split="per_sample")
                leaderboard_df = lb_ds.to_pandas()
                samples_df = samples_ds.to_pandas()
            st.sidebar.success(f"Loaded from {repo_id}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.info("Enter a HuggingFace Dataset Repo ID in the sidebar.")

elif source == "Local JSON File":
    uploaded = st.sidebar.file_uploader("Upload benchmark results JSON", type=["json"])
    if uploaded:
        data = json.load(uploaded)
        metadata = data.get("metadata", {})
        leaderboard = data.get("leaderboard", [])
        if leaderboard:
            leaderboard_df = pd.DataFrame(leaderboard)

        # Build per-sample from detailed_results
        all_samples = []
        for detail in data.get("detailed_results", []):
            if "error" in detail:
                continue
            for sample in detail.get("per_sample_results", []):
                all_samples.append({"model_id": detail["model_id"], "dataset_id": detail["dataset_id"], **sample})
        if all_samples:
            samples_df = pd.DataFrame(all_samples)
        st.sidebar.success("Loaded from uploaded file")
    else:
        default_path = Path(__file__).resolve().parent.parent / "results" / "latest_leaderboard.json"
        if default_path.exists():
            with open(default_path) as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            leaderboard = data.get("leaderboard", [])
            if leaderboard:
                leaderboard_df = pd.DataFrame(leaderboard)
            st.sidebar.info(f"Loaded from {default_path}")
        else:
            st.info("Upload a benchmark results JSON file to get started.")


# --- Dashboard ---
if leaderboard_df is not None and not leaderboard_df.empty:

    # === LEADERBOARD ===
    st.header("Leaderboard")
    st.markdown("Sorted by **WER** (lower is better)")

    lb_display = leaderboard_df.copy()
    lb_display.index = range(1, len(lb_display) + 1)
    lb_display.index.name = "Rank"

    display_cols = ["model_id", "category", "size_params", "wer", "cer", "rtf",
                    "inference_time_s", "avg_time_per_sample_s", "cost_T4_usd"]
    available_cols = [c for c in display_cols if c in lb_display.columns]
    st.dataframe(
        lb_display[available_cols].style.format({
            "wer": "{:.4f}", "cer": "{:.4f}", "rtf": "{:.4f}",
            "inference_time_s": "{:.1f}", "avg_time_per_sample_s": "{:.4f}",
            "cost_T4_usd": "${:.6f}",
        }, na_rep="N/A"),
        use_container_width=True,
    )

    # === KEY METRICS COMPARISON ===
    st.header("Model Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("WER by Model (lower = better)")
        wer_chart = lb_display.set_index("model_id")[["wer"]].sort_values("wer")
        st.bar_chart(wer_chart)

    with col2:
        st.subheader("CER by Model (lower = better)")
        cer_chart = lb_display.set_index("model_id")[["cer"]].sort_values("cer")
        st.bar_chart(cer_chart)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Inference Speed (RTF, lower = faster)")
        if "rtf" in lb_display.columns:
            rtf_chart = lb_display.dropna(subset=["rtf"]).set_index("model_id")[["rtf"]].sort_values("rtf")
            st.bar_chart(rtf_chart)

    with col4:
        st.subheader("Cost per Benchmark Run (T4 GPU)")
        if "cost_T4_usd" in lb_display.columns:
            cost_chart = lb_display.set_index("model_id")[["cost_T4_usd"]].sort_values("cost_T4_usd")
            st.bar_chart(cost_chart)

    # === CATEGORY COMPARISON ===
    if "category" in lb_display.columns:
        st.header("Vietnamese-specific vs Multilingual")
        categories = lb_display["category"].unique()
        for cat in categories:
            cat_df = lb_display[lb_display["category"] == cat]
            st.subheader(f"Category: {cat}")
            best = cat_df.iloc[0]
            st.metric(f"Best in {cat}", best["model_id"], f"WER: {best['wer']:.4f}")

    # === COST ANALYSIS ===
    st.header("Cost Analysis")
    st.markdown("All benchmarks use **free tier** (Google Colab T4 / Kaggle GPU). Estimated costs shown for reference.")

    if "cost_T4_usd" in lb_display.columns and "wer" in lb_display.columns:
        st.subheader("Cost vs Quality Trade-off")
        scatter_data = lb_display[["model_id", "wer", "cost_T4_usd"]].set_index("model_id")
        st.scatter_chart(scatter_data, x="cost_T4_usd", y="wer")

    # === METADATA ===
    if metadata:
        st.header("Benchmark Info")
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Models Tested", metadata.get("num_models", "N/A"))
        with info_cols[1]:
            st.metric("Datasets", metadata.get("num_datasets", "N/A"))
        with info_cols[2]:
            st.metric("Device", metadata.get("device", "N/A"))
        with info_cols[3]:
            st.metric("Timestamp", metadata.get("timestamp", "N/A"))

    # === PER-SAMPLE ANALYSIS ===
    if samples_df is not None and not samples_df.empty:
        st.header("Per-Sample Analysis")

        model_filter = st.selectbox("Select Model", samples_df["model_id"].unique())
        filtered = samples_df[samples_df["model_id"] == model_filter]

        st.subheader(f"WER Distribution — {model_filter}")
        st.bar_chart(filtered["wer"].value_counts().sort_index().head(50))

        tab1, tab2, tab3 = st.tabs(["Worst Samples", "Best Samples", "All Samples"])
        with tab1:
            st.dataframe(filtered.nlargest(20, "wer")[["index", "reference", "prediction", "wer", "cer"]], use_container_width=True)
        with tab2:
            st.dataframe(filtered.nsmallest(20, "wer")[["index", "reference", "prediction", "wer", "cer"]], use_container_width=True)
        with tab3:
            st.dataframe(filtered[["index", "reference", "prediction", "wer", "cer"]], use_container_width=True)

else:
    st.markdown("""
    ### Getting Started

    1. **Run the benchmark** (on Kaggle with free GPU):
       - Open the Kaggle notebook from the repo
       - Enable GPU accelerator, run all cells
       - Schedule for automatic weekly runs

    2. **Push results to HuggingFace:**
       ```bash
       python scripts/push_to_hf.py --hf-repo-id your-username/asr-benchmark-results
       ```

    3. **View results here** by entering the HuggingFace repo ID or uploading the JSON file.
    """)
