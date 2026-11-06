"""
Duration Calculator: computes total, active, away, and unknown durations and percentages.
Guarantees ACTIVE + AWAY + UNKNOWN = Total video duration, with percentages
calculated using total video duration as the denominator.
"""

from typing import Any, Dict, List


class DurationCalculator:
    """
    Computes activity duration totals and percentages.
    Uses total video duration as the denominator for percentages and reconciles
    rounding differences so ACTIVE + AWAY + UNKNOWN equals total video duration.
    """

    @staticmethod
    def calculate(
        intervals: List[Dict[str, Any]], total_video_duration: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate total durations and percentages from timeline intervals.
        """
        raw_active = 0.0
        raw_away = 0.0
        raw_unknown = 0.0

        for it in intervals:
            dur = float(it.get("duration", 0.0))
            st = it.get("status", "").upper()
            if st == "ACTIVE":
                raw_active += dur
            elif st == "AWAY":
                raw_away += dur
            elif st == "UNKNOWN":
                raw_unknown += dur

        raw_sum = raw_active + raw_away + raw_unknown
        total_dur = round(total_video_duration if total_video_duration > 0 else raw_sum, 2)

        # Round individual categories
        active_duration = round(raw_active, 2)
        away_duration = round(raw_away, 2)
        unknown_duration = round(raw_unknown, 2)

        # Reconcile any rounding difference against total video duration
        diff = round(total_dur - (active_duration + away_duration + unknown_duration), 2)
        if diff != 0.0:
            if unknown_duration > 0:
                unknown_duration = round(unknown_duration + diff, 2)
            elif away_duration >= active_duration:
                away_duration = round(away_duration + diff, 2)
            else:
                active_duration = round(active_duration + diff, 2)

        classified_duration = round(active_duration + away_duration, 2)

        # Percentages using total video duration as denominator
        if total_dur > 0:
            active_percentage = round((active_duration / total_dur) * 100.0, 2)
            away_percentage = round((away_duration / total_dur) * 100.0, 2)
            # Reconcile percentage rounding so sum is 100.0%
            unknown_percentage = round(100.0 - active_percentage - away_percentage, 2)
        else:
            active_percentage = 0.0
            away_percentage = 0.0
            unknown_percentage = 0.0

        return {
            "total_video_duration": total_dur,
            "classified_duration": classified_duration,
            "active_duration": active_duration,
            "away_duration": away_duration,
            "unknown_duration": unknown_duration,
            "active_percentage": active_percentage,
            "away_percentage": away_percentage,
            "unknown_percentage": unknown_percentage,
            "interval_count": len(intervals),
        }
