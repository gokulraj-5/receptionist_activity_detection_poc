"""
Timeline Manager: stores per-frame observations and constructs contiguous,
non-overlapping activity intervals (ACTIVE, AWAY, UNKNOWN).
"""

from typing import Any, Dict, List, Optional
import pandas as pd


class TimelineManager:
    """
    Records observations for each frame and builds the final activity timeline.
    Guarantees deterministic lookup so any frame always yields the exact same state.
    """

    def __init__(self):
        # Lookup table: frame_index -> observation dict
        self.frames: Dict[int, Dict[str, Any]] = {}
        self.intervals: List[Dict[str, Any]] = []

    def record_frame(
        self,
        frame_index: int,
        video_time: float,
        status: str,
        cctv_time: str,
        inside_count: int = 0,
        people_count: int = 0,
        detections: Optional[List[Dict[str, Any]]] = None,
    ):
        """Save observation for a sampled frame."""
        self.frames[frame_index] = {
            "frame_index": frame_index,
            "video_time": video_time,
            "status": status,
            "cctv_time": cctv_time,
            "inside_count": inside_count,
            "people_count": people_count,
            "detections": detections or [],
        }

    def fill_unsampled_frames(self, total_frames: int, fps: float):
        """
        Fill unsampled frames between strides with the most recent sampled observation.
        """
        last_rec = None
        for f in range(total_frames):
            if f in self.frames:
                last_rec = self.frames[f]
            else:
                time_sec = f / fps if fps > 0 else 0.0
                if last_rec is not None:
                    self.frames[f] = {
                        "frame_index": f,
                        "video_time": time_sec,
                        "status": last_rec["status"],
                        "cctv_time": last_rec["cctv_time"],
                        "inside_count": last_rec["inside_count"],
                        "people_count": last_rec["people_count"],
                        "detections": last_rec["detections"],
                    }
                else:
                    self.frames[f] = {
                        "frame_index": f,
                        "video_time": time_sec,
                        "status": "UNKNOWN",
                        "cctv_time": "UNKNOWN",
                        "inside_count": 0,
                        "people_count": 0,
                        "detections": [],
                    }

    def get_frame(self, frame_index: int) -> Dict[str, Any]:
        """Deterministic lookup for a frame (used by Pass 2 rendering)."""
        return self.frames.get(
            frame_index,
            {
                "frame_index": frame_index,
                "video_time": 0.0,
                "status": "UNKNOWN",
                "cctv_time": "UNKNOWN",
                "inside_count": 0,
                "people_count": 0,
                "detections": [],
            },
        )

    def generate_intervals(self, total_frames: int, fps: float) -> List[Dict[str, Any]]:
        """
        Convert frame-by-frame status into contiguous, non-overlapping intervals.
        """
        self.fill_unsampled_frames(total_frames, fps)

        intervals: List[Dict[str, Any]] = []
        curr = None
        interval_id = 1

        for f in range(total_frames):
            rec = self.frames[f]
            status = rec["status"]
            v_time = rec["video_time"]
            c_time = rec["cctv_time"]

            if curr is None:
                curr = {
                    "interval_id": interval_id,
                    "status": status,
                    "start_frame": f,
                    "end_frame": f,
                    "start_time": v_time,
                    "end_time": v_time,
                    "duration": 0.0,
                    "start_cctv_time": c_time,
                    "end_cctv_time": c_time,
                }
            elif curr["status"] == status:
                curr["end_frame"] = f
                curr["end_time"] = v_time
                curr["end_cctv_time"] = c_time
            else:
                # Close current interval
                dur = (curr["end_frame"] - curr["start_frame"] + 1) / fps
                curr["duration"] = round(dur, 2)
                intervals.append(curr)

                interval_id += 1
                curr = {
                    "interval_id": interval_id,
                    "status": status,
                    "start_frame": f,
                    "end_frame": f,
                    "start_time": v_time,
                    "end_time": v_time,
                    "duration": 0.0,
                    "start_cctv_time": c_time,
                    "end_cctv_time": c_time,
                }

        if curr is not None:
            dur = (curr["end_frame"] - curr["start_frame"] + 1) / fps
            curr["duration"] = round(dur, 2)
            intervals.append(curr)

        self.intervals = intervals
        return self.intervals

    def to_dataframe(self) -> pd.DataFrame:
        """Export intervals to a pandas DataFrame."""
        return pd.DataFrame(self.intervals)
