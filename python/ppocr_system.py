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
import argparse
import ppocr_rec as predict_rec
import ppocr_det as predict_det


class TextSystem:
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = float(os.environ.get('PPOCR_DROP_SCORE', '0.5'))
        # PPOCR_INPUT_SCALE: upscale input image before det (e.g. 2.0).
        # Larger crops are extracted → rec sees more pixels per character.
        self.input_scale = float(os.environ.get('PPOCR_INPUT_SCALE', '1.0'))

    def run(self, img):
        # Optional: upscale input so det crops contain more pixels per character
        if self.input_scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * self.input_scale), int(h * self.input_scale)),
                             interpolation=cv2.INTER_LANCZOS4)

        # 1. TextDetector — skip for small images where det produces bad results.
        # When the image is a single-word crop (small h or w after any scaling),
        # det stretches it to 480x480 and finds spurious boxes; pass directly to rec.
        ori_im = img.copy()
        if img.shape[0] < 80 or img.shape[1] < 400:
            dt_boxes = []  # force fallback
        else:
            dt_boxes = self.text_detector.run(img)

        # Fallback: if det finds nothing OR image is a small crop, pass the
        # whole image directly to rec. rec hard-resizes to 320x48 internally,
        # so no pre-upscaling needed.
        if dt_boxes is None or len(dt_boxes) == 0:
            h, w = ori_im.shape[:2]
            rec_res = self.text_recognizer.run([ori_im])
            filter_boxes, filter_rec_res = [], []
            if rec_res:
                text, score = rec_res[0][0]
                if score >= self.drop_score:
                    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_res[0])
            return filter_boxes, filter_rec_res

        # 2. TextRecognizer
        dt_boxes = sorted_boxes(dt_boxes)
        img_crop_list = [get_rotate_crop_image(ori_im, box) for box in dt_boxes]
        rec_res = self.text_recognizer.run(img_crop_list)

        # 3. Filter
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


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
    num_boxes = dt_boxes.shape[0]
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
