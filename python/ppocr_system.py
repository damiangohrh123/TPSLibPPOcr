# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import cv2
import numpy as np
import ppocr_rec as predict_rec
import ppocr_det as predict_det


# ── detection helpers ──────────────────────────────────────────────────────────

def _nms_boxes(boxes, iom_threshold=0.6):
    """
    Suppress duplicate/nested boxes using Intersection-over-Min-Area (IoM).
    Standard IoU NMS misses containment cases (small box inside large box).
    IoM = intersection_area / min(area_i, area_j) — fires when a small box
    is largely swallowed by a larger one. The smaller box is suppressed.
    """
    if len(boxes) <= 1:
        return boxes

    areas = [cv2.contourArea(b.astype(np.float32)) for b in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    suppressed = set()
    keep = []

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            inter_area, _ = cv2.intersectConvexConvex(
                boxes[i].astype(np.float32),
                boxes[j].astype(np.float32)
            )
            iom = inter_area / (min(areas[i], areas[j]) + 1e-6)
            if iom > iom_threshold:
                suppressed.add(j)

    return [boxes[i] for i in sorted(keep)]


def _box_size_ok(b, min_h, min_w):
    """Return True if box is large enough to contain real text (scaled coords)."""
    return (np.linalg.norm(b[0] - b[3]) >= min_h and
            np.linalg.norm(b[0] - b[1]) >= min_w)


def _run_det_tiled(img_det, detector, overlap=256):
    """
    Hybrid det: squash pass + directional tiles, merged and NMS-deduplicated.

    - Wide  (w > h × 1.5): squash + left tile  + right tile  (x-shifted)
    - Tall  (h > w × 1.5): squash + top  tile  + bottom tile (y-shifted)
    - Square-ish          : single pass (no tiling needed)
    """
    h, w = img_det.shape[:2]

    wide = w > h * 1.5
    tall = h > w * 1.5

    if not wide and not tall:
        # Near-square: a single 480×480 pass covers the image without severe compression.
        return detector.run(img_det)

    # Squash pass — full image compressed to 480×480.
    # Reliably captures dense regions (toolbars, status bars) where layout
    # matters more than per-character resolution.
    boxes_squash = detector.run(img_det)
    all_boxes = []
    if boxes_squash is not None and len(boxes_squash):
        all_boxes.extend(list(boxes_squash))

    if wide:
        # Horizontal tiling: left and right halves with `overlap` px of shared content.
        tile_w  = w // 2 + overlap // 2
        offset2 = w - tile_w           # start column of right tile

        boxes1 = detector.run(img_det[:, :tile_w])
        boxes2 = detector.run(img_det[:, offset2:])

        if boxes1 is not None and len(boxes1):
            all_boxes.extend(list(boxes1))
        if boxes2 is not None and len(boxes2):
            for b in boxes2:
                b2 = b.copy()
                b2[:, 0] += offset2    # translate x back to full-image coords
                all_boxes.append(b2)

    else:  # tall
        # Vertical tiling: top and bottom halves with `overlap` px of shared content.
        tile_h  = h // 2 + overlap // 2
        offset2 = h - tile_h           # start row of bottom tile

        boxes1 = detector.run(img_det[:tile_h, :])
        boxes2 = detector.run(img_det[offset2:, :])

        if boxes1 is not None and len(boxes1):
            all_boxes.extend(list(boxes1))
        if boxes2 is not None and len(boxes2):
            for b in boxes2:
                b2 = b.copy()
                b2[:, 1] += offset2    # translate y back to full-image coords
                all_boxes.append(b2)

    return np.array(all_boxes) if all_boxes else None


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(img, input_scale=1.0, binarize=True):
    """Otsu binarization (optional) followed by Lanczos4 upscale (optional).
    Returns the preprocessed image ready for detection."""
    if binarize:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray_bin, cv2.COLOR_GRAY2BGR)

    if input_scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * input_scale), int(h * input_scale)),
                         interpolation=cv2.INTER_LANCZOS4)
    return img


# ── main system ────────────────────────────────────────────────────────────────

class TextSystem:
    def __init__(self, args):
        self.text_detector   = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score  = float(os.environ.get('PPOCR_DROP_SCORE',  '0.5'))
        self.input_scale = float(os.environ.get('PPOCR_INPUT_SCALE', '1.0'))
        self.binarize    = os.environ.get('PPOCR_BINARIZE', '1') != '0'
        # Precompute scaled thresholds — these depend only on init-time env vars.
        self._min_h = float(os.environ.get('PPOCR_MIN_HEIGHT', '10')) * self.input_scale
        self._min_w = float(os.environ.get('PPOCR_MIN_WIDTH',  '8'))  * self.input_scale

    def run(self, img):
        img = preprocess_image(img, self.input_scale, self.binarize)

        # Detection: skip for small images where det produces bad results.
        if img.shape[0] < 80 or img.shape[1] < 400:
            dt_boxes = []
        else:
            raw_boxes = _run_det_tiled(img, self.text_detector)
            if raw_boxes is not None and len(raw_boxes):
                dt_boxes = sorted_boxes(raw_boxes)
                dt_boxes = [b for b in dt_boxes if _box_size_ok(b, self._min_h, self._min_w)]
                dt_boxes = _nms_boxes(dt_boxes)
            else:
                dt_boxes = []

        # Fallback: no boxes found or image too small — pass whole image to rec.
        if not dt_boxes:
            h, w = img.shape[:2]
            rec_res = self.text_recognizer.run([img])
            filter_boxes, filter_rec_res = [], []
            if rec_res:
                text, score = rec_res[0][0]
                if score >= self.drop_score:
                    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_res[0])
            return filter_boxes, filter_rec_res

        # Recognition.
        img_crop_list = [get_rotate_crop_image(img, box) for box in dt_boxes]
        rec_res = self.text_recognizer.run(img_crop_list)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


# ── utilities ──────────────────────────────────────────────────────────────────

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    if dst_img.shape[0] / dst_img.shape[1] >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def sorted_boxes(dt_boxes):
    """Sort text boxes top to bottom, left to right."""
    num_boxes = dt_boxes.shape[0] if hasattr(dt_boxes, 'shape') else len(dt_boxes)
    _boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes


if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser(description='PPOCR-System Python Demo')
    parser.add_argument('--det_model_path', type=str, required=True)
    parser.add_argument('--rec_model_path', type=str, required=True)
    parser.add_argument('--image_path',     type=str, required=True)
    parser.add_argument('--target',         type=str, default='rk3588')
    parser.add_argument('--device_id',      type=str, default=None)
    args = parser.parse_args()

    system_model = TextSystem(args)
    img = cv2.imread(args.image_path)
    if img is None:
        print('ERROR: cannot read image:', args.image_path)
        raise SystemExit(1)

    start = time.time()
    filter_boxes, filter_rec_res = system_model.run(img)
    end = time.time()

    print(filter_rec_res)
    print(f"Inference time: {(end - start) * 1000:.1f} ms")
