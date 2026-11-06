"""
Report Generator: exports timeline CSV, summary JSON, and executive TXT report.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict
import pandas as pd

from utils.time_utils import seconds_to_hms


class ReportGenerator:
    """
    Exports activity reports:
      - output/activity_timeline.csv
      - output/activity_summary.json
      - output/activity_summary.txt
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(
        self,
        timeline_df: pd.DataFrame,
        report_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """Generate CSV, JSON, and TXT reports simultaneously."""
        csv_path = os.path.join(self.output_dir, "activity_timeline.csv")
        json_path = os.path.join(self.output_dir, "activity_summary.json")
        txt_path = os.path.join(self.output_dir, "activity_summary.txt")

        # 1. CSV Timeline
        timeline_df.to_csv(csv_path, index=False)

        # 2. JSON Summary
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, default=str)

        # 3. Formatted Text Report
        durs = report_data.get("durations", {})
        meta = report_data.get("video_metadata", {})
        intervals = report_data.get("timeline_intervals", [])

        txt_lines = [
            "=" * 76,
            "            RECEPTIONIST ACTIVITY DETECTION SUMMARY REPORT",
            "=" * 76,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "-" * 76,
            "1. EXECUTIVE SUMMARY & KPIS",
            "-" * 76,
            f"Total Video Duration   : {durs.get('total_video_duration', 0.0):.2f}s ({seconds_to_hms(durs.get('total_video_duration', 0.0))})",
            f"Classified Duration    : {durs.get('classified_duration', 0.0):.2f}s ({seconds_to_hms(durs.get('classified_duration', 0.0))}) [ACTIVE + AWAY]",
            f"ACTIVE Duration        : {durs.get('active_duration', 0.0):.2f}s ({durs.get('active_percentage', 0.0):.2f}%)",
            f"AWAY Duration          : {durs.get('away_duration', 0.0):.2f}s ({durs.get('away_percentage', 0.0):.2f}%)",
            f"UNKNOWN Duration       : {durs.get('unknown_duration', 0.0):.2f}s ({durs.get('unknown_percentage', 0.0):.2f}%)",
            "",
            "-" * 76,
            "2. VIDEO METADATA",
            "-" * 76,
            f"Resolution             : {meta.get('width', 0)} x {meta.get('height', 0)}",
            f"Frame Rate (FPS)       : {meta.get('fps', 0.0):.4f}",
            f"Total Frames           : {meta.get('total_frames', 0)}",
            "",
            "-" * 76,
            "3. ACTIVITY TIMELINE INTERVALS",
            "-" * 76,
            f"{'ID':<4} {'Status':<9} {'Frames':<16} {'Video Time (s)':<22} {'Duration':<10} {'CCTV Start':<20} {'CCTV End'}",
            "-" * 76,
        ]

        for it in intervals:
            it_id = it.get("interval_id", 0)
            status = it.get("status", "")
            f_range = f"{it.get('start_frame', 0)}-{it.get('end_frame', 0)}"
            t_range = f"{it.get('start_time', 0.0):.2f}s - {it.get('end_time', 0.0):.2f}s"
            dur = f"{it.get('duration', 0.0):.2f}s"
            cctv_s = str(it.get("start_cctv_time", ""))
            cctv_e = str(it.get("end_cctv_time", ""))
            txt_lines.append(
                f"{it_id:<4} {status:<9} {f_range:<16} {t_range:<22} {dur:<10} {cctv_s:<20} {cctv_e}"
            )

        txt_lines.extend([
            "",
            "=" * 76,
            "                          END OF REPORT",
            "=" * 76,
        ])

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

        return {
            "csv_path": csv_path,
            "json_path": json_path,
            "txt_path": txt_path,
        }
