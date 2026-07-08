"""
Print full OCR output for an image — all boxes with x/y positions.
Used to inspect field layout before writing rule-based extractors.

Usage (on board):
    python3 ~/pipeline/test/ocr_dump.py ~/pipeline/test/images/auto_mode_1.jpg
"""
import os, sys, time, types

BOARD_MODEL_DIR    = os.path.expanduser("~/pipeline/model")
BOARD_PIPELINE_DIR = os.path.expanduser("~/pipeline/python")

os.environ.setdefault("PPOCR_CHAR_DICT", os.path.join(BOARD_MODEL_DIR, "ppocr_keys_v6.txt"))
sys.path.insert(0, BOARD_PIPELINE_DIR)

from ppocr_system import TextSystem
import cv2

args = types.SimpleNamespace(
    det_model_path=os.path.join(BOARD_MODEL_DIR, "PP-OCRv6_tiny_det_rk3588.rknn"),
    rec_model_path=os.path.join(BOARD_MODEL_DIR, "PP-OCRv6_tiny_rec_rk3588.rknn"),
    target="rk3588", device_id=None,
)

img_path = sys.argv[1] if len(sys.argv) > 1 else None
if not img_path:
    print("Usage: ocr_dump.py <image_path>")
    sys.exit(1)

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f"Image: {img_path}  ({w}x{h})")
print()

ocr = TextSystem(args)
boxes, rec_res = ocr.run(img)

items = []
for box, rec in zip(boxes, rec_res):
    text, score = rec[0]
    y_pct = int(box[0][1] / h * 100)
    x_pct = int(box[0][0] / w * 100)
    items.append((y_pct, x_pct, text, score))
items.sort()

print(f"{'#':>3}  {'x%':>4} {'y%':>4}  {'score':>6}  text")
print("-" * 60)
for i, (y, x, t, s) in enumerate(items, 1):
    print(f"{i:>3}  {x:>4}% {y:>4}%  {s:>6.3f}  {t}")

print()
print(f"Total: {len(items)} boxes")
