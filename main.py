"""
Main Application: Receptionist Activity Detection Pipeline.

Architecture:
  Pass 1: Video Analysis
    - Sequential frame reading with VideoProcessor
    - YOLOv8 Person Detection (class ID 0)
    - Workstation polygon ROI occupancy test
    - Temporal Activity Classification (ACTIVE, AWAY, UNKNOWN)
    - CCTV timestamp extraction (OCR with interpolation fallback)
    - Deterministic observation recording in TimelineManager

  Pass 2: Video Rendering
    - Deterministic replay of Pass 1 observations
    - Annotate workstation ROI boundary in red
    - Display bounding boxes ONLY for people inside the ROI (green)
    - Render clean HUD dashboard with status, workstation state, elapsed time, and CCTV time
    - Save MP4 output
"""

import logging
import os
import sys
import time

import config
from src.activity_classifier import ActivityClassifier
from src.duration_calculator import DurationCalculator
from src.person_detector import PersonDetector
from src.report_generator import ReportGenerator
from src.roi_detector import ROIDetector
from src.timeline_manager import TimelineManager
from src.timestamp_extractor import TimestampExtractor
from src.video_annotator import VideoAnnotator
from src.video_processor import VideoProcessor
from utils.time_utils import seconds_to_hms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Pipeline")


def run_pipeline(progress_callback=None):
    """
    Run the end-to-end Receptionist Activity Detection pipeline.

    Args:
        progress_callback: Optional callable(pass_name, current_frame, total_frames)
                           for tracking progress.

    Returns:
        dict containing 'durations', 'intervals', and 'video_metadata'.
    """
    start_time = time.time()
    logger.info("Starting Receptionist Activity Detection Pipeline...")

    # --- Validate input video ---
    if not os.path.exists(config.INPUT_VIDEO_PATH):
        logger.error(f"Input video not found at: {config.INPUT_VIDEO_PATH}")
        raise FileNotFoundError(f"Input video not found: {config.INPUT_VIDEO_PATH}")

    video = VideoProcessor(config.INPUT_VIDEO_PATH)
    video.display_video_info()

    # --- Initialize modules ---
    person_detector = PersonDetector(
        model_path=config.MODEL_PATH,
        confidence=config.CONFIDENCE_THRESHOLD,
        classes=[config.PERSON_CLASS_ID],
    )
    roi_detector = ROIDetector(
        roi_points=config.ROI_POINTS,
        min_overlap_ratio=getattr(config, "ROI_MIN_OVERLAP_RATIO", 0.40),
    )
    activity_classifier = ActivityClassifier(
        active_confirmation=config.ACTIVE_CONFIRMATION_SECONDS,
        away_confirmation=config.AWAY_CONFIRMATION_SECONDS,
    )
    timestamp_extractor = TimestampExtractor(
        crop_box=config.OCR_CROP_BOX,
        tesseract_cmd=config.TESSERACT_CMD,
        cctv_base_time=config.CCTV_START_DATETIME,
    )
    timeline_manager = TimelineManager()
    video_annotator = VideoAnnotator(roi_detector, codec=config.OUTPUT_CODEC)
    report_generator = ReportGenerator(config.OUTPUT_DIR)

    # =========================================================================
    # PASS 1: Video Analysis
    # =========================================================================
    logger.info("PASS 1: Analyzing video frames...")
    try:
        for frame_index, frame in video.read_frames():
            # Sample frames based on configured interval
            if frame_index % config.FRAME_SAMPLE_INTERVAL != 0:
                continue

            current_time = frame_index / video.fps

            # 1. Detect all people using YOLOv8
            people = person_detector.detect_people(frame)

            # 2. Robust ROI occupancy check for each detected person
            inside_count = 0
            for person in people:
                is_inside = roi_detector.is_person_inside_roi(person["bbox"])
                person["is_inside"] = is_inside
                if is_inside:
                    inside_count += 1

            # 3. Temporal activity classification (ACTIVE / AWAY / UNKNOWN)
            status = activity_classifier.update(inside_count, current_time)

            # 4. CCTV timestamp extraction
            ts_info = timestamp_extractor.extract_timestamp(frame, frame_index, video.fps)

            # 5. Record frame observation for deterministic timeline & rendering
            timeline_manager.record_frame(
                frame_index=frame_index,
                video_time=current_time,
                status=status,
                cctv_time=ts_info["formatted"],
                inside_count=inside_count,
                people_count=len(people),
                detections=people,
            )

            if frame_index % 1500 == 0 or frame_index >= video.total_frames - config.FRAME_SAMPLE_INTERVAL:
                pct = (frame_index / video.total_frames) * 100.0
                logger.info(
                    f"Pass 1: frame {frame_index}/{video.total_frames} ({pct:.1f}%) | "
                    f"Status: {status:<7} | Occupants in ROI: {inside_count} | "
                    f"CCTV: {ts_info['formatted']}"
                )

            if progress_callback:
                progress_callback("pass1", frame_index, video.total_frames)

    finally:
        video.release()

    # =========================================================================
    # Build timeline intervals and calculate durations
    # =========================================================================
    logger.info("Building activity timeline and calculating durations...")
    intervals = timeline_manager.generate_intervals(video.total_frames, video.fps)
    timeline_df = timeline_manager.to_dataframe()
    durations = DurationCalculator.calculate(intervals, video.duration)

    logger.info(f"ACTIVE  : {durations['active_duration']}s ({durations['active_percentage']}%)")
    logger.info(f"AWAY    : {durations['away_duration']}s ({durations['away_percentage']}%)")
    logger.info(f"UNKNOWN : {durations['unknown_duration']}s ({durations['unknown_percentage']}%)")
    logger.info(f"Total   : {durations['total_video_duration']}s")

    # =========================================================================
    # Generate audit reports (CSV, JSON, TXT)
    # =========================================================================
    report_data = {
        "video_metadata": {
            "video_path": config.INPUT_VIDEO_PATH,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "total_frames": video.total_frames,
            "duration": video.duration,
        },
        "durations": durations,
        "timeline_intervals": intervals,
        "diagnostics": timestamp_extractor.get_diagnostics(),
    }
    report_generator.generate_all(timeline_df, report_data)
    logger.info("Reports saved to output/ directory.")

    # =========================================================================
    # PASS 2: Video Rendering
    # =========================================================================
    logger.info("PASS 2: Rendering annotated video MP4...")

    def on_render_progress(curr, total):
        if curr % 500 == 0 or curr == total:
            pct = (curr / total) * 100.0
            logger.info(f"Pass 2: rendered frame {curr}/{total} ({pct:.1f}%)")
        if progress_callback:
            progress_callback("pass2", curr, total)

    success = video_annotator.render_video(
        input_video_path=config.INPUT_VIDEO_PATH,
        output_video_path=config.OUTPUT_VIDEO_PATH,
        timeline_manager=timeline_manager,
        video_fps=video.fps,
        video_width=video.width,
        video_height=video.height,
        total_frames=video.total_frames,
        progress_callback=on_render_progress,
    )

    if success:
        logger.info(f"Video saved to: {config.OUTPUT_VIDEO_PATH}")
    else:
        logger.error("Failed to render video.")

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed in {seconds_to_hms(elapsed)} ({elapsed:.2f}s).")

    return {
        "durations": durations,
        "intervals": intervals,
        "video_metadata": report_data["video_metadata"],
    }


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
