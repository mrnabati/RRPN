#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################

"""Script to convert Selective Search proposal boxes into the Detectron proposal
file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths
import numpy as np
import h5py
import sys

from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import save_object


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    file_in = sys.argv[2]
    file_out = sys.argv[3]

    ds = JsonDataset(dataset_name)
    roidb = ds.get_roidb()

    boxes = []
    scores = []
    ids = []

    with h5py.File(file_in, 'r') as f:
        raw_boxes = f['boxes']
        num_imgs = len(raw_boxes)
        assert num_imgs == len(roidb)

        for ind in range(num_imgs):
            if ind % 1000 == 0:
                print('{}/{}'.format(ind + 1, len(roidb)))

            ## -------- Working down below
            img_boxes = f[raw_boxes[ind,0]][()]
            
            try:
                swp_boxes = np.swapaxes(img_boxes,0,1)
            except:
                print(img_boxes)
                # print(swp_boxes)
                input('something')

            # selective search boxes are 1-indexed and (y1, x1, y2, x2)
            i_boxes = swp_boxes[:, (1, 0, 3, 2)] - 1

            # input('something')

            boxes.append(i_boxes.astype(np.float32))
            scores.append(np.zeros((i_boxes.shape[0]), dtype=np.float32))
            ids.append(roidb[ind]['id'])

    save_object(dict(boxes=boxes, scores=scores, ids=ids), file_out)
