# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class BaseRecLabelDecode(object):
    """Convert between text-label and text-index."""

    def __init__(self, character_dict_path=None, use_space_char=False):
        self.character_str = []
        if character_dict_path is None:
            self.character_str = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(character_dict_path, "rb") as fin:
                for line in fin.readlines():
                    self.character_str.append(line.decode('utf-8').strip("\n").strip("\r\n"))
            if use_space_char:
                self.character_str.append(" ")

        dict_character = self.add_special_char(list(self.character_str))
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """Convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        for batch_idx in range(len(text_index)):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in text_index[batch_idx][selection]]
            conf_list = text_prob[batch_idx][selection] if text_prob is not None else [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            result_list.append((''.join(char_list), np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # CTC blank


class CTCLabelDecode(BaseRecLabelDecode):
    """CTC decoder: argmax + duplicate removal."""

    def __init__(self, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, (tuple, list)):
            preds = preds[-1]
        preds_idx  = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        return text, self.decode(label)

    def add_special_char(self, dict_character):
        return ['blank'] + dict_character
