"""
ROI Detector: identifies whether a detected person is inside the workstation ROI.
Uses a robust occupancy test combining bounding-box center test and polygon area overlap.
"""

from typing import List, Tuple
import cv2
import numpy as np


class ROIDetector:
    """
    Manages the workstation polygon ROI and performs robust occupancy checks
    for detected persons.
    """

    def __init__(self, roi_points: List[List[int]], min_overlap_ratio: float = 0.40):
        """
        Args:
            roi_points: List of [x, y] coordinates defining the polygon ROI.
            min_overlap_ratio: Minimum ratio of bounding box area that must overlap
                               the ROI polygon to count as occupancy if center is outside.
        """
        self.roi_points = np.array(roi_points, dtype=np.int32)
        self.min_overlap_ratio = min_overlap_ratio

    def draw_roi(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        """Draw the workstation ROI polygon boundary on the frame."""
        cv2.polylines(
            frame,
            [self.roi_points],
            isClosed=True,
            color=color,
            thickness=thickness,
        )
        return frame

    def is_person_inside_roi(self, bbox: Tuple[int, int, int, int], min_overlap: float = None) -> bool:
        """
        Check whether a person's bounding box is inside the workstation ROI.

        Robust occupancy strategy:
          1. Primary test: Center point of the bounding box is inside the polygon.
          2. Secondary test: If center is borderline/outside, verify whether the
             bounding box area overlap with the polygon reaches min_overlap_ratio.
        """
        if min_overlap is None:
            min_overlap = self.min_overlap_ratio

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 1. Primary check: Bounding-box center inside polygon
        if cv2.pointPolygonTest(self.roi_points, (center_x, center_y), False) >= 0:
            return True

        # 2. Secondary check: Substantial bounding-box area overlap
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
        if bbox_area <= 0:
            return False

        # Quick reject using the ROI's bounding rectangle
        rx, ry, rw, rh = cv2.boundingRect(self.roi_points)
        ix1 = max(x1, rx)
        iy1 = max(y1, ry)
        ix2 = min(x2, rx + rw)
        iy2 = min(y2, ry + rh)
        if ix1 >= ix2 or iy1 >= iy2:
            return False

        # Intersection mask between the bounding box and ROI polygon
        sub_poly = self.roi_points - np.array([ix1, iy1], dtype=np.int32)
        poly_mask = np.zeros((iy2 - iy1, ix2 - ix1), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [sub_poly], 255)

        inter_area = cv2.countNonZero(poly_mask)
        overlap_ratio = inter_area / float(bbox_area)

        return overlap_ratio >= min_overlap
