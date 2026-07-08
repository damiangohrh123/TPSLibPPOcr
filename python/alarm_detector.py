"""
alarm_detector.py — Detect alarm banners in HMI screenshots using HSV color detection.

A red alarm banner is characterised by:
  - Predominantly red (HSV hue wraps at both ends of the spectrum)
  - Wide horizontal strip (aspect ratio >> 1)
  - Located near the top of the screen

Usage:
    from alarm_detector import AlarmDetector

    detector = AlarmDetector()
    result = detector.detect(img, boxes, rec_res)

    if result["alarm"]:
        print("ALARM:", result["text"])
        print("Region:", result["bbox"])
"""

import cv2
import numpy as np


class AlarmDetector:
    """
    Detect a red alarm banner in a BGR image and extract its text from
    existing OCR results (no second OCR pass required).

    Parameters
    ----------
    min_area : int
        Minimum pixel area for a red region to be considered a banner.
        Filters out small buttons and logos (default: 10 000).
    min_aspect_ratio : float
        Minimum width/height ratio.  An alarm banner spans nearly the full
        screen width while being short, so ratios >> 1.  Buttons are roughly
        square (default: 4.0).
    max_y_fraction : float
        Only consider red regions whose top edge is within this fraction of
        the total image height.  Alarm banners sit near the top of the HMI
        (default: 0.35).
    """

    def __init__(
        self,
        min_area: int = 10_000,
        min_aspect_ratio: float = 4.0,
        max_y_fraction: float = 0.35,
    ):
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_y_fraction = max_y_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img, boxes=None, rec_res=None) -> dict:
        """
        Detect alarm condition and, if OCR results are provided, extract
        the alarm text from within the banner.

        Parameters
        ----------
        img : np.ndarray
            BGR image (as loaded by cv2.imread).
        boxes : list | None
            OCR detection boxes from ppocr (each box is a list of 4 [x,y] points).
        rec_res : list | None
            OCR recognition results: list of (text, score) tuples.

        Returns
        -------
        dict with keys:
            alarm (bool)   : True if a red alarm banner was found.
            bbox  (tuple)  : (x, y, w, h) of the banner, or None.
            text  (str)    : space-joined alarm text from OCR, or None.
        """
        alarm, bbox = self._find_alarm_banner(img)

        if not alarm:
            return {"alarm": False, "bbox": None, "text": None}

        text = None
        if bbox is not None and boxes is not None and rec_res is not None:
            text = self._extract_alarm_text(boxes, rec_res, bbox)

        return {"alarm": True, "bbox": bbox, "text": text}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_alarm_banner(self, img):
        """Return (True, bbox) if a qualifying red banner is found, else (False, None)."""
        h_img, w_img = img.shape[:2]

        # Convert to HSV and build red mask.
        # Red wraps around the hue wheel: 0-10 and 170-180.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
        mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

        # Optional morphological close to join gaps inside the banner
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            # Filter 1: minimum area (removes tiny noise, icons, small buttons)
            if area < self.min_area:
                continue

            # Filter 2: aspect ratio (banner is wide, buttons are roughly square)
            aspect = w / h if h > 0 else 0
            if aspect < self.min_aspect_ratio:
                continue

            # Filter 3: vertical position (banner is near the top of the screen)
            if y / h_img > self.max_y_fraction:
                continue

            if area > best_area:
                best_area = area
                best_bbox = (x, y, w, h)

        if best_bbox is None:
            return False, None

        return True, best_bbox

    def _extract_alarm_text(self, boxes, rec_res, bbox):
        """
        Filter OCR results to those whose centre falls inside the alarm banner,
        then return the recognised text joined by spaces.
        """
        bx, by, bw, bh = bbox
        alarm_texts = []

        for box, rec in zip(boxes, rec_res):
            text, score = rec[0]
            # Use the centre point of the OCR box for the containment check
            pts = np.array(box, dtype=np.float32)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())

            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                alarm_texts.append(text)

        return " ".join(alarm_texts) if alarm_texts else None

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def draw_debug(self, img, result) -> np.ndarray:
        """
        Return a copy of *img* with the alarm banner highlighted.
        Useful for offline verification.
        """
        out = img.copy()
        if result["alarm"] and result["bbox"] is not None:
            x, y, w, h = result["bbox"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"ALARM: {result['text']}" if result["text"] else "ALARM"
            cv2.putText(out, label, (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return out
