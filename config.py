"""
Project Configuration for Receptionist Activity Detection.
Centralized settings for video paths, model parameters, ROI coordinates,
activity confirmation timings, and CCTV OCR extraction.
"""

import os

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
INPUT_VIDEO_PATH = os.path.join("input", "input_video.mp4")
OUTPUT_DIR = "output"
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "processed_reception_video.mp4")
TIMELINE_CSV_PATH = os.path.join(OUTPUT_DIR, "activity_timeline.csv")
SUMMARY_TXT_PATH = os.path.join(OUTPUT_DIR, "activity_summary.txt")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "activity_summary.json")

# ---------------------------------------------------------------------------
# YOLO Settings
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("model", "yolov8n.pt")
CONFIDENCE_THRESHOLD = 0.25
PERSON_CLASS_ID = 0

# ---------------------------------------------------------------------------
# Video Sampling & Codec
# ---------------------------------------------------------------------------
FRAME_SAMPLE_INTERVAL = 5   # Process YOLO on every 5th frame
OUTPUT_CODEC = "mp4v"       # Standard MP4 codec

# ---------------------------------------------------------------------------
# Reception Workstation ROI (Calibrated 4-point polygon)
# ---------------------------------------------------------------------------
ROI_POINTS = [
    [296, 307],
    [501, 223],
    [629, 477],
    [380, 477],
]
WORKSTATION_ROI = ROI_POINTS
ROI_MIN_OVERLAP_RATIO = 0.40  # Robust occupancy: center inside or >=40% bbox overlap

# ---------------------------------------------------------------------------
# Activity Confirmation Settings (in elapsed video seconds)
# ---------------------------------------------------------------------------
ACTIVE_CONFIRMATION_SECONDS = 1.0  # Duration person must be in ROI to enter ACTIVE
AWAY_CONFIRMATION_SECONDS = 2.0    # Duration ROI must be empty to enter AWAY

# ---------------------------------------------------------------------------
# CCTV Timestamp & OCR Settings
# ---------------------------------------------------------------------------
OCR_CROP_BOX = {"ymin": 20, "xmin": 0, "ymax": 45, "xmax": 260}
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CCTV_START_DATETIME = "2026-06-18 09:34:30"
