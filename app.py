"""
Streamlit Web App: Receptionist Activity Detection Dashboard.

Run with:
    streamlit run app.py

Features:
  - Configure workstation ROI, detection confidence, and confirmation timings from the sidebar.
  - Run the full detection pipeline with live progress bars.
  - View the processed MP4 video (with green boxes for ROI occupants, red ROI boundary).
  - Browse the activity timeline table.
  - Review ACTIVE / AWAY / UNKNOWN duration metrics and time distribution charts.
  - Download CSV, JSON, and TXT audit reports.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# Make sure project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Receptionist Activity Detection",
    page_icon="📷",
    layout="wide",
)

st.title("📷 Receptionist Activity Detection")
st.caption("CCTV Workstation Occupancy & Activity Detection using YOLOv8 and ROI Analysis")

# ─────────────────────────────────────────────────────────────
# Sidebar: Configuration
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Detection & Occupancy")
    conf_thresh = st.slider(
        "YOLO Confidence Threshold", 0.10, 0.90, float(config.CONFIDENCE_THRESHOLD), step=0.05
    )
    overlap_ratio = st.slider(
        "Min ROI Overlap Ratio", 0.10, 0.90, float(getattr(config, "ROI_MIN_OVERLAP_RATIO", 0.40)), step=0.05
    )

    st.subheader("Activity Confirmation (seconds)")
    active_conf = st.number_input(
        "Active confirmation (s)", min_value=0.1, max_value=10.0,
        value=float(config.ACTIVE_CONFIRMATION_SECONDS), step=0.5
    )
    away_conf = st.number_input(
        "Away confirmation (s)", min_value=0.1, max_value=20.0,
        value=float(config.AWAY_CONFIRMATION_SECONDS), step=0.5
    )

    st.subheader("Workstation ROI Points")
    roi_text = st.text_area(
        "ROI Polygon [x, y] coordinates",
        value=json.dumps(config.ROI_POINTS),
        height=110,
    )

    run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Helper: apply sidebar settings to config
# ─────────────────────────────────────────────────────────────
def apply_sidebar_config():
    config.CONFIDENCE_THRESHOLD = conf_thresh
    config.ROI_MIN_OVERLAP_RATIO = overlap_ratio
    config.ACTIVE_CONFIRMATION_SECONDS = active_conf
    config.AWAY_CONFIRMATION_SECONDS = away_conf
    try:
        roi_parsed = json.loads(roi_text)
        config.ROI_POINTS = roi_parsed
    except json.JSONDecodeError:
        st.warning("ROI JSON is invalid — keeping default ROI from config.py")


# ─────────────────────────────────────────────────────────────
# Pipeline runner in a background thread
# ─────────────────────────────────────────────────────────────
def _run_in_thread(result_holder: dict, progress_state: dict):
    try:
        from main import run_pipeline

        def on_progress(pass_name, current, total):
            progress_state["pass"] = pass_name
            progress_state["current"] = current
            progress_state["total"] = total

        result = run_pipeline(progress_callback=on_progress)
        result_holder["result"] = result
    except Exception as exc:
        result_holder["error"] = exc
    finally:
        progress_state["done"] = True


# ─────────────────────────────────────────────────────────────
# Execution & Progress
# ─────────────────────────────────────────────────────────────
if run_clicked:
    apply_sidebar_config()
    st.info("Pipeline started… processing CCTV footage...")

    result_holder: dict = {}
    progress_state: dict = {"pass": "pass1", "current": 0, "total": 1, "done": False}

    thread = threading.Thread(
        target=_run_in_thread,
        args=(result_holder, progress_state),
        daemon=True,
    )
    thread.start()

    p1_bar = st.progress(0, text="Pass 1: Analyzing frames…")
    p2_bar = st.progress(0, text="Pass 2: Rendering video…")
    status_text = st.empty()

    while not progress_state["done"]:
        p = progress_state.get("pass", "pass1")
        curr = progress_state.get("current", 0)
        total = progress_state.get("total", 1)
        pct = min(curr / max(total, 1), 1.0)

        if p == "pass1":
            p1_bar.progress(pct, text=f"Pass 1: frame {curr}/{total} ({pct*100:.1f}%)")
        else:
            p1_bar.progress(1.0, text="Pass 1: ✅ Done")
            p2_bar.progress(pct, text=f"Pass 2: frame {curr}/{total} ({pct*100:.1f}%)")

        status_text.caption(f"Processing ({p.upper()})…")
        time.sleep(0.5)

    thread.join()
    p1_bar.progress(1.0, text="Pass 1: ✅ Done")
    p2_bar.progress(1.0, text="Pass 2: ✅ Done")
    status_text.empty()

    if "error" in result_holder:
        st.error(f"Pipeline failed: {result_holder['error']}")
    else:
        st.success("✅ Pipeline completed successfully!")
        st.session_state["pipeline_result"] = result_holder["result"]
        st.rerun()

# ─────────────────────────────────────────────────────────────
# Results Section
# ─────────────────────────────────────────────────────────────
result = st.session_state.get("pipeline_result")
output_video = config.OUTPUT_VIDEO_PATH
csv_path = config.TIMELINE_CSV_PATH
json_path = config.SUMMARY_JSON_PATH
txt_path = config.SUMMARY_TXT_PATH

outputs_exist = all(os.path.exists(p) for p in [output_video, csv_path, json_path, txt_path])

if not outputs_exist and result is None:
    st.info("👈 Review settings in the sidebar and click **Run Pipeline** to start.")
    st.stop()

if result is None and os.path.exists(json_path):
    with open(json_path, encoding="utf-8") as f:
        saved = json.load(f)
    result = {
        "durations": saved.get("durations", {}),
        "intervals": saved.get("timeline_intervals", []),
        "video_metadata": saved.get("video_metadata", {}),
    }

# ── Tabs ──────────────────────────────────────────────────────
tab_video, tab_timeline, tab_summary, tab_download = st.tabs(
    ["🎬 Video", "📊 Timeline", "📈 Summary", "⬇️ Download Reports"]
)

# ── Tab 1: Video ──────────────────────────────────────────────
with tab_video:
    if os.path.exists(output_video):
        st.subheader("Processed CCTV Video")
        st.video(output_video, format="video/mp4")
        st.caption(
            "🟢 Green box = person inside reception ROI | "
            "🔴 Red polygon = reception workstation ROI | "
            "Top-Right = status HUD"
        )
    else:
        st.warning("Processed video not found. Run the pipeline first.")

# ── Tab 2: Timeline ───────────────────────────────────────────
with tab_timeline:
    if result and result.get("intervals"):
        import pandas as pd

        df = pd.DataFrame(result["intervals"])
        st.subheader("Activity Timeline Intervals")

        def row_colour(row):
            if row.get("status") == "ACTIVE":
                return ["background-color: #1a3a1a"] * len(row)
            elif row.get("status") == "AWAY":
                return ["background-color: #3a1a1a"] * len(row)
            else:
                return ["background-color: #2a2a2a"] * len(row)

        st.dataframe(df.style.apply(row_colour, axis=1), use_container_width=True)
    elif os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        st.subheader("Activity Timeline (from CSV)")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No timeline data found. Run the pipeline first.")

# ── Tab 3: Summary ────────────────────────────────────────────
with tab_summary:
    if result and result.get("durations"):
        dur = result["durations"]
        meta = result.get("video_metadata", {})

        st.subheader("Workstation Occupancy Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Duration", f"{dur.get('total_video_duration', 0):.2f}s")
        col2.metric(
            "ACTIVE (Occupied)",
            f"{dur.get('active_duration', 0):.2f}s",
            f"{dur.get('active_percentage', 0):.2f}%",
        )
        col3.metric(
            "AWAY (Empty)",
            f"{dur.get('away_duration', 0):.2f}s",
            f"{dur.get('away_percentage', 0):.2f}%",
        )
        col4.metric(
            "UNKNOWN",
            f"{dur.get('unknown_duration', 0):.2f}s",
            f"{dur.get('unknown_percentage', 0):.2f}%",
        )

        st.divider()

        # Distribution Chart
        try:
            import plotly.graph_objects as go

            labels = ["ACTIVE", "AWAY", "UNKNOWN"]
            values = [
                dur.get("active_duration", 0),
                dur.get("away_duration", 0),
                dur.get("unknown_duration", 0),
            ]
            colors = ["#00cc44", "#cc3300", "#888888"]

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.35,
            )])
            fig.update_layout(
                title="Activity Time Distribution (% of Total Video)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            import pandas as pd
            chart_df = pd.DataFrame({
                "Duration (s)": [
                    dur.get("active_duration", 0),
                    dur.get("away_duration", 0),
                    dur.get("unknown_duration", 0),
                ]
            }, index=["ACTIVE", "AWAY", "UNKNOWN"])
            st.bar_chart(chart_df)

        st.divider()
        st.subheader("Video Metadata")
        st.json(meta)

    else:
        st.warning("No summary data found. Run the pipeline first.")

# ── Tab 4: Download ───────────────────────────────────────────
with tab_download:
    st.subheader("Download Audit Reports")

    files = {
        "activity_timeline.csv": (csv_path, "text/csv"),
        "activity_summary.json": (json_path, "application/json"),
        "activity_summary.txt": (txt_path, "text/plain"),
    }

    for filename, (fpath, mime) in files.items():
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"⬇️ Download {filename}",
                data=data,
                file_name=filename,
                mime=mime,
            )
        else:
            st.caption(f"❌ {filename} not found — run pipeline first.")
