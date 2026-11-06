"""
Time utilities: formatting seconds to HH:MM:SS and parsing CCTV timestamps.
"""

from datetime import datetime
import re
from typing import Optional


def seconds_to_hms(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_datetime(dt: Optional[datetime]) -> str:
    """Format datetime into 'YYYY-MM-DD HH:MM:SS' string."""
    if dt is None:
        return "UNKNOWN"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_cctv_timestamp(raw_text: str) -> Optional[datetime]:
    """
    Parse a CCTV timestamp string into a datetime object.
    Supports formats like '18-06-2026 Thu 09:34:30', '18-06-2026 09:34:30',
    or '2026-06-18 09:34:30'.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip().replace("\n", " ")

    # Pattern: DD-MM-YYYY [Day] HH:MM:SS
    m = re.search(
        r"(\d{2})[-/.](\d{2})[-/.](\d{4})\s*(?:[A-Za-z]{2,4}\s*)?(\d{2}):(\d{2}):(\d{2})",
        cleaned,
    )
    if m:
        day, month, year, h, minute, s = map(int, m.groups())
        try:
            return datetime(year, month, day, h, minute, s)
        except ValueError:
            return None

    # Pattern: YYYY-MM-DD [Day] HH:MM:SS
    m = re.search(
        r"(\d{4})[-/.](\d{2})[-/.](\d{2})\s*(?:[A-Za-z]{2,4}\s*)?(\d{2}):(\d{2}):(\d{2})",
        cleaned,
    )
    if m:
        year, month, day, h, minute, s = map(int, m.groups())
        try:
            return datetime(year, month, day, h, minute, s)
        except ValueError:
            return None

    return None
