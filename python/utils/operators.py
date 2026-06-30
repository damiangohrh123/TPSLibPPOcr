# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import cv2
import numpy as np


class NormalizeImage(object):
    """Normalize image: subtract mean, divide by std."""

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std  = std  if std  is not None else [0.229, 0.224, 0.225]
        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std  = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        data['image'] = (img.astype('float32') * self.scale - self.mean) / self.std
        return data


class DetResizeForTest(object):
    """Resize image to a fixed shape (image_shape=[h, w]) for det inference."""

    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        assert 'image_shape' in kwargs, "DetResizeForTest requires image_shape"
        self.image_shape = kwargs['image_shape']

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self._resize(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        if len(data['shape'].shape) == 1:
            data['shape'] = np.expand_dims(data['shape'], axis=0)
        return data

    def _resize(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio_h, ratio_w]
