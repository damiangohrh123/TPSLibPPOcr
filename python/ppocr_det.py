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
import numpy as np
import utils.operators
from utils.db_postprocess import DBPostProcess, DetPostProcess

DET_INPUT_SHAPE = [480, 480]  # h, w

# ImageNet normalization baked into RKNN model; Python sends raw BGR [0-255].
PRE_PROCESS_CONFIG = [
    {'DetResizeForTest': {'image_shape': DET_INPUT_SHAPE}},
    {'NormalizeImage': {'std': [1., 1., 1.], 'mean': [0., 0., 0.], 'scale': '1.', 'order': 'hwc'}},
]

DB_POSTPROCESS_CONFIG = {
    'thresh':         float(os.environ.get('PPOCR_DET_THRESH',   '0.3')),
    'box_thresh':     float(os.environ.get('PPOCR_BOX_THRESH',   '0.4')),
    'unclip_ratio':   float(os.environ.get('PPOCR_UNCLIP_RATIO', '1.5')),
    'max_candidates': 3000,
    'score_mode':     'fast',
}


class TextDetector:
    def __init__(self, args):
        self.model = setup_model(args)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                self.preprocess_funct.append(getattr(utils.operators, key)(**item[key]))
        self.db_postprocess = DBPostProcess(**DB_POSTPROCESS_CONFIG)
        self.det_postprocess = DetPostProcess()

    def run(self, img):
        data = {'image': img}
        for p in self.preprocess_funct:
            data = p(data)
        data['image'] = data['image'][np.newaxis, ...].astype(np.float32)
        preds = {'maps': self.model.run([data['image']])[0].astype(np.float32)}
        result = self.db_postprocess(preds, data['shape'])
        return self.det_postprocess.filter_tag_det_res(result[0]['points'], img.shape)


def setup_model(args):
    from rknn_executor import RKNN_model_container
    return RKNN_model_container(args.det_model_path, args.target, args.device_id)
