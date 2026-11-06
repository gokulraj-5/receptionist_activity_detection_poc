"""
Video Annotator: renders visual bounding boxes, workstation ROI, and HUD overlay
onto CCTV video frames during Pass 2.

Annotation specifications:
  - Workstation ROI boundary drawn as a red polygon.
  - Bounding boxes displayed ONLY around people detected inside the ROI (green).
  - No bounding boxes or labels displayed for people outside the ROI.
  - Activity status HUD: ACTIVE (green), AWAY (red), UNKNOWN (gray).
  - Original CCTV timestamp and video elapsed time.
  - Clean and professional appearance without purple or clothing color labels.
  - Automatic H.264 optimization for seamless web browser playback in Streamlit.
"""

import logging
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, Optional
import cv2
import numpy as np

from src.roi_detector import ROIDetector
from utils.time_utils import seconds_to_hms

logger = logging.getLogger(__name__)


class VideoAnnotator:
    """
    Renders annotations onto video frames based on precomputed observations.
    """

    def __init__(self, roi_detector: ROIDetector, codec: str = "mp4v"):
        self.roi_detector = roi_detector
        self.codec = codec

    def annotate_frame(self, frame: np.ndarray, record: Dict[str, Any]) -> np.ndarray:
        """
        Draw ROI boundary, inside-ROI bounding boxes, and HUD overlay on a single frame.
        """
        annotated = frame.copy()

        # 1. Draw workstation ROI polygon boundary in Red
        annotated = self.roi_detector.draw_roi(annotated, color=(0, 0, 255), thickness=2)

        # 2. Draw bounding boxes ONLY around people detected INSIDE the workstation ROI
        detections = record.get("detections", [])
        for person in detections:
            if not person.get("is_inside", False):
                # Do not display bounding boxes around people outside the ROI
                continue

            bbox = person.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            conf = person.get("confidence", 0.0)
            label = f"OCCUPANT ({conf:.2f})" if conf > 0 else "OCCUPANT"

            # Green bounding box for ROI occupant
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        # 3. Clean and professional HUD Dashboard (Top-Right)
        status = record.get("status", "UNKNOWN").upper()
        inside_count = record.get("inside_count", 0)
        video_time = record.get("video_time", 0.0)
        cctv_time = record.get("cctv_time", "UNKNOWN")

        # Color-code status
        if status == "ACTIVE":
            status_color = (0, 255, 0)      # Green
        elif status == "AWAY":
            status_color = (0, 0, 255)      # Red
        else:
            status_color = (180, 180, 180)  # Gray

        # Dark background card in top-right (leaves top-left CCTV timestamp unobscured)
        cv2.rectangle(annotated, (495, 8), (840, 125), (20, 20, 20), -1)
        cv2.rectangle(annotated, (495, 8), (840, 125), (80, 80, 80), 1)

        cv2.putText(
            annotated,
            f"STATUS: {status}",
            (510, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        cv2.putText(
            annotated,
            f"Workstation: {'OCCUPIED' if inside_count > 0 else 'EMPTY'} ({inside_count})",
            (510, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"Video Time: {seconds_to_hms(video_time)} ({video_time:.1f}s)",
            (510, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            f"CCTV: {cctv_time}",
            (510, 114),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
        )

        return annotated

    def _optimize_for_web(self, output_path: str):
        """
        Transcode MP4 to H.264 (AVC) with faststart so HTML5 web browsers (Chrome, Edge,
        Firefox, Safari, and Streamlit) can play it natively.
        """
        try:
            ffmpeg_exe = shutil.which("ffmpeg")
            if not ffmpeg_exe:
                try:
                    import imageio_ffmpeg
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None

            if not ffmpeg_exe or not os.path.exists(output_path):
                return

            temp_path = output_path + ".h264.mp4"
            cmd = [
                ffmpeg_exe, "-y",
                "-i", output_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-crf", "23",
                "-preset", "veryfast",
                temp_path,
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                os.replace(temp_path, output_path)
                logger.info("Video successfully optimized to H.264 for web playback.")
            elif os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Web optimization skipped: {e}")

    def render_video(
        self,
        input_video_path: str,
        output_video_path: str,
        timeline_manager: Any,
        video_fps: float,
        video_width: int,
        video_height: int,
        total_frames: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """
        Pass 2 Rendering: Reads original CCTV video sequentially, applies precomputed
        observations deterministically, and writes the processed MP4.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open input video for rendering: {input_video_path}")
            return False

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))

        if not writer.isOpened():
            logger.warning(f"Codec {self.codec} failed, falling back to mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (video_width, video_height))

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                record = timeline_manager.get_frame(frame_idx)
                annotated = self.annotate_frame(frame, record)
                writer.write(annotated)

                frame_idx += 1
                if progress_callback and (frame_idx % 500 == 0 or frame_idx == total_frames):
                    progress_callback(frame_idx, total_frames)

            # Ensure H.264 web compatibility for browser / Streamlit playback
            self._optimize_for_web(output_video_path)

            return True
        finally:
            if cap.isOpened():
                cap.release()
            writer.release()
