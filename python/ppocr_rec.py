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
import utils.operators
from utils.rec_postprocess import CTCLabelDecode

REC_INPUT_SHAPE = [48, 320]  # h, w
CHARACTER_DICT_PATH = os.environ.get('PPOCR_CHAR_DICT', '../model/ppocr_keys_v6.txt')

# [0,1] plain /255 normalisation - standard for v6 tiny rec.
PRE_PROCESS_CONFIG = [{
    'NormalizeImage': {'std': [1, 1, 1], 'mean': [0, 0, 0], 'scale': '1./255.', 'order': 'hwc'}
}]

CTC_POSTPROCESS_CONFIG = {
    'character_dict_path': CHARACTER_DICT_PATH,
    'use_space_char': True,
}


class TextRecognizer:
    def __init__(self, args):
        self.model = setup_model(args)
        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                self.preprocess_funct.append(getattr(utils.operators, key)(**item[key]))
        self.ctc_postprocess = CTCLabelDecode(**CTC_POSTPROCESS_CONFIG)

    def run(self, imgs):
        outputs = []
        for img in imgs:
            data = {'image': cv2.resize(img, (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))}
            for p in self.preprocess_funct:
                data = p(data)
            data['image'] = data['image'][np.newaxis, ...].astype(np.float32)
            outputs.append(self.ctc_postprocess(self.model.run([data['image']])[0].astype(np.float32)))
        return outputs


def setup_model(args):
    from rknn_executor import RKNN_model_container
    return RKNN_model_container(args.rec_model_path, args.target, args.device_id)
