"""
auto_compare.py — Compare field values between two auto mode screenshots.

Runs OCR on both images and prints a table showing which values changed.

Usage (on board):
    python3 ~/pipeline/test/auto_compare.py
    python3 ~/pipeline/test/auto_compare.py --img1 auto_mode_1.jpg --img2 auto_mode_2.jpg
"""

import argparse
import os
import sys
import types

BOARD_MODEL_DIR    = os.path.expanduser("~/pipeline/model")
BOARD_PIPELINE_DIR = os.path.expanduser("~/pipeline/python")
IMAGES_DIR         = os.path.expanduser("~/pipeline/test/images")

os.environ.setdefault("PPOCR_CHAR_DICT", os.path.join(BOARD_MODEL_DIR, "ppocr_keys_v6.txt"))
sys.path.insert(0, BOARD_PIPELINE_DIR)


def build_ocr_args():
    args = types.SimpleNamespace()
    args.det_model_path = os.path.join(BOARD_MODEL_DIR, "PP-OCRv6_tiny_det_rk3588.rknn")
    args.rec_model_path = os.path.join(BOARD_MODEL_DIR, "PP-OCRv6_tiny_rec_rk3588.rknn")
    args.target         = "rk3588"
    args.device_id      = None
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1",      default="auto_mode_1.jpg")
    parser.add_argument("--img2",      default="auto_mode_2.jpg")
    parser.add_argument("--image_dir", default=IMAGES_DIR)
    args = parser.parse_args()

    from ppocr_system import TextSystem
    import cv2
    from auto_extractor import extract, compare, print_comparison

    print("Loading OCR models...")
    ocr = TextSystem(build_ocr_args())
    print("OK\n")

    def run(filename):
        path = os.path.join(args.image_dir, filename)
        img  = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        boxes, rec_res = ocr.run(img)
        return img, boxes, rec_res

    img1, boxes1, rec1 = run(args.img1)
    img2, boxes2, rec2 = run(args.img2)

    fields1 = extract(boxes1, rec1, img1.shape)
    fields2 = extract(boxes2, rec2, img2.shape)

    label1 = os.path.splitext(args.img1)[0].replace("_", " ").title()
    label2 = os.path.splitext(args.img2)[0].replace("_", " ").title()

    print(f"Comparing: {args.img1}  vs  {args.img2}\n")
    rows = compare(fields1, fields2)
    print_comparison(rows, label1, label2)


if __name__ == "__main__":
    main()
