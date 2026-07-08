"""
alarm_test.py — Test AlarmDetector in isolation on one or more images.

Usage (on board):
    python3 ~/pipeline/test/alarm_test.py
    python3 ~/pipeline/test/alarm_test.py --image alarm.jpeg --save_debug
"""

import argparse
import glob
import os
import sys
import time
import types

BOARD_OCR_MDL_DIR  = os.path.expanduser("~/pipeline/model")
BOARD_PIPELINE_DIR = os.path.expanduser("~/pipeline/python")
IMAGES_DIR         = os.path.expanduser("~/pipeline/test/images")

os.environ.setdefault("PPOCR_CHAR_DICT", os.path.join(BOARD_OCR_MDL_DIR, "ppocr_keys_v6.txt"))
sys.path.insert(0, BOARD_PIPELINE_DIR)


def build_ocr_args():
    args = types.SimpleNamespace()
    args.det_model_path = os.path.join(BOARD_OCR_MDL_DIR, "PP-OCRv6_tiny_det_rk3588.rknn")
    args.rec_model_path = os.path.join(BOARD_OCR_MDL_DIR, "PP-OCRv6_tiny_rec_rk3588.rknn")
    args.target         = "rk3588"
    args.device_id      = None
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir",  default=IMAGES_DIR)
    parser.add_argument("--image",      default=None)
    parser.add_argument("--save_debug", action="store_true")
    args = parser.parse_args()

    if args.image:
        images = [os.path.join(args.image_dir, args.image)]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        images = sorted(p for ext in exts for p in glob.glob(os.path.join(args.image_dir, ext)))

    if not images:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)

    from ppocr_system import TextSystem
    import cv2
    from alarm_detector import AlarmDetector

    print("Loading OCR models...")
    ocr = TextSystem(build_ocr_args())
    detector = AlarmDetector()
    print("OK\n")

    for img_path in images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{img_name}: cannot read, skipping")
            continue

        t0 = time.time()
        boxes, rec_res = ocr.run(img)
        ocr_ms = (time.time() - t0) * 1000

        t1 = time.time()
        result = detector.detect(img, boxes, rec_res)
        alarm_ms = (time.time() - t1) * 1000

        print(f"{'─'*60}")
        print(f"Image : {img_name}")
        print(f"OCR   : {len(boxes)} boxes, {ocr_ms:.0f} ms")
        print(f"Alarm : {result['alarm']}  ({alarm_ms:.1f} ms)")
        if result["alarm"]:
            print(f"  bbox : {result['bbox']}")
            print(f"  text : {result['text']}")
        print()

        if args.save_debug and result["alarm"]:
            debug_img = detector.draw_debug(img, result)
            out_path = os.path.join(args.image_dir, f"debug_{img_name}")
            cv2.imwrite(out_path, debug_img)
            print(f"  debug image saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
