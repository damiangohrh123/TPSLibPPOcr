"""
auto_extractor.py — Rule-based field extraction for HMI auto mode screens.

Extracts key operational values from "Single Device Full Automation" screens
using fixed positional windows calibrated to the known screen layout.
No LLM required.

Usage:
    from auto_extractor import extract, compare, print_comparison

    fields = extract(boxes, rec_res, img.shape)
    rows   = compare(fields_1, fields_2)
    print_comparison(rows)
"""

import re

# ── Field definitions ─────────────────────────────────────────────────────────
# Each entry: (name, x_min%, x_max%, y_min%, y_max%)
# Positions match the ocr_dump format: top-left corner of each OCR box,
# expressed as a percentage of image width/height.
#
# Calibrated from ocr_dump on auto_mode_1.jpg and auto_mode_2.jpg (1024x768).

FIELDS = [
    # Top section — current operation parameters (set values, usually 0.000 at start)
    ("Z1 set height",        62, 73, 11, 17),
    ("Z2 set height",        62, 73, 17, 22),
    ("Feed speed (setting)", 62, 73, 22, 31),

    # Machine status — cutting / spinner state
    ("Cut status",           37, 60, 48, 52),   # e.g. CUTTING DONE, WASH DONE
    ("Spinner status",       60, 80, 48, 52),   # e.g. WAITING, READY

    # Machine status — blade wear measurements
    ("Blade height",         45, 60, 53, 57),
    ("Z1 blade wear",        45, 60, 57, 59),
    ("Z2 blade wear",        45, 60, 59, 66),

    # Machine status — actual feed speed
    ("Feed speed (actual)",  44, 62, 63, 71),

    # Blade replacement counters (left status section)
    ("Z1 line count",         8, 23, 46, 50),
    ("Z1 distance",          22, 36, 46, 50),
    ("Z2 line count",         8, 23, 50, 57),
    ("Z2 distance",          22, 36, 50, 57),
]


def extract(boxes, rec_res, img_shape) -> dict:
    """
    Extract named field values from OCR results using positional windows.

    Parameters
    ----------
    boxes     : OCR detection boxes from ppocr_system.run()
    rec_res   : OCR recognition results [(text, score), ...]
    img_shape : (height, width, ...) tuple

    Returns
    -------
    dict of {field_name: value_string}  — missing fields are None.
    """
    h, w = img_shape[:2]

    # Build (x%, y%, text) table using top-left corner of each box
    items = []
    for box, rec in zip(boxes, rec_res):
        text, score = rec[0]
        x_pct = box[0][0] / w * 100
        y_pct = box[0][1] / h * 100
        items.append((x_pct, y_pct, text))

    result = {}
    for name, xmin, xmax, ymin, ymax in FIELDS:
        matches = [
            (x, text) for x, y, text in items
            if xmin <= x <= xmax and ymin <= y <= ymax
        ]
        if not matches:
            result[name] = None
            continue
        # Sort left-to-right so value and unit tokens join in the correct order
        matches.sort(key=lambda m: m[0])
        raw = " ".join(t for _, t in matches)
        result[name] = _normalize(raw)

    return result


def compare(fields1: dict, fields2: dict) -> list:
    """
    Compare two extracted field dicts.

    Returns
    -------
    List of (field_name, val1, val2, changed) tuples in field definition order.
    """
    rows = []
    for name, *_ in FIELDS:
        v1 = fields1.get(name) or "–"
        v2 = fields2.get(name) or "–"
        rows.append((name, v1, v2, v1 != v2))
    return rows


def print_comparison(rows: list, label1="Auto Mode 1", label2="Auto Mode 2"):
    """Print a formatted comparison table to stdout."""
    col_name = max(len(r[0]) for r in rows)
    col_v1   = max(len(r[1]) for r in rows)
    col_v2   = max(len(r[2]) for r in rows)

    sep = "-" * (col_name + col_v1 + col_v2 + 12)
    header = f"{'Field':<{col_name}}  {label1:<{col_v1}}  {label2:<{col_v2}}  Changed"
    print(header)
    print(sep)
    for name, v1, v2, changed in rows:
        marker = "  *" if changed else ""
        print(f"{name:<{col_name}}  {v1:<{col_v1}}  {v2:<{col_v2}}{marker}")
    print()
    changed_count = sum(1 for r in rows if r[3])
    print(f"{changed_count} of {len(rows)} fields changed.")


# ── OCR normalization helpers ─────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Fix common OCR misreads in numeric and unit strings."""
    # "0-000mm" → "0.000mm"  (OCR reads decimal point as dash)
    text = re.sub(r'(\d)-(\d)', r'\1.\2', text)
    # "36071ines" → "3607 lines"  (l misread as 1)
    text = re.sub(r'(\d+)[1Ii]([Ii]?ines?)', r'\1 lines', text)
    # standalone "I ines", "Iines", "1ines" → "lines"
    text = re.sub(r'\b[1Ii]\s*[Ii]?ines?\b', 'lines', text)
    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
