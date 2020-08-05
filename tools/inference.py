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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder. Allows for using a combination of multiple models.
For example, one model may be used for RPN, another model for Fast R-CNN style
box detection, yet another model to predict masks, and yet another model to
predict keypoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_path
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
from PIL import Image
import logging
import os
import sys

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
import detectron.core.rpn_generator as rpn_engine
import detectron.core.test_engine as model_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.env as envu
import detectron.utils.vis as vis_utils

from rrpn_generator import rrpn_loader
from cocoplus.coco import COCO_PLUS

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# infer.py
#   --im [path/to/image.jpg] \
#   --rpn-model [path/to/rpn/model.pkl] \
#   --rpn-cfg [path/to/rpn/config.yaml] \
#   --output-dir [path/to/output/dir] \
#   [model1] [config1] [model2] [config2] ...


def parse_args():
    parser = argparse.ArgumentParser(description='Inference on an image')
    parser.add_argument(
        '--im_ind', help='input image index', type=int
    )
    parser.add_argument(
        '--ann_file', dest='ann_file',
        help='Annotations file',
        default='./data/nucoco/annotations/instances_mini_val.json'
    )
    parser.add_argument(
        '--imgs_dir', dest='imgs_dir',
        help='Images directory',
        default='./data/nucoco/mini_val'
    )
    parser.add_argument(
        '--rpn-pkl',
        dest='rpn_pkl',
        help='rpn model file (pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer)',
        default='./data/inference',
        type=str
    )
    parser.add_argument(
        'models_to_run',
        help='pairs of models & configs, listed like so: [pkl1] [yaml1] [pkl2] [yaml2] ...',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    dummy_nucoco_dataset = dummy_datasets.get_nucoco_dataset()
    cfg_orig = load_cfg(envu.yaml_dump(cfg))
    
    ## Load image
    coco = COCO_PLUS(args.ann_file, args.imgs_dir)
    image_id = coco.dataset['images'][args.im_ind]['id']
    img_path = os.path.join(args.imgs_dir, coco.imgs[image_id]["file_name"])
    im = cv2.imread(img_path)

    ## Get the proposals for this image
    proposals = rrpn_loader(args.rpn_pkl)
    proposal_boxes = proposals[image_id]['boxes']
    _proposal_scores = proposals[image_id]['scores']
    workspace.ResetWorkspace()

    ## run models
    cls_boxes, cls_segms, cls_keyps = None, None, None
    for i in range(0, len(args.models_to_run), 2):
        pkl = args.models_to_run[i]
        yml = args.models_to_run[i + 1]
        cfg.immutable(False)
        merge_cfg_from_cfg(cfg_orig)
        merge_cfg_from_file(yml)
        if len(pkl) > 0:
            weights_file = pkl
        else:
            weights_file = cfg.TEST.WEIGHTS
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)
        model = model_engine.initialize_model_from_cfg(weights_file)
        with c2_utils.NamedCudaScope(0):
            cls_boxes_, cls_segms_, cls_keyps_ = \
                model_engine.im_detect_all(model, im, proposal_boxes)
        cls_boxes = cls_boxes_ if cls_boxes_ is not None else cls_boxes
        cls_segms = cls_segms_ if cls_segms_ is not None else cls_segms
        cls_keyps = cls_keyps_ if cls_keyps_ is not None else cls_keyps
        workspace.ResetWorkspace()

    out_name = os.path.join(
        args.output_dir, '{}'.format(os.path.basename(img_path) + '.pdf')
    )
    logger.info('Processing {} -> {}'.format(img_path, out_name))

    vis_utils.vis_one_image(
        im[:, :, ::-1],
        img_path,
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dummy_nucoco_dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
    )


def check_args(args):
    # assert (
    #     (args.rpn_pkl is not None and args.rpn_cfg is not None) or
    #     (args.rpn_pkl is None and args.rpn_cfg is None)
    # )
    if args.rpn_pkl is not None:
        args.rpn_pkl = cache_url(args.rpn_pkl, cfg.DOWNLOAD_CACHE)
        assert os.path.exists(args.rpn_pkl)
        # assert os.path.exists(args.rpn_cfg)
    if args.models_to_run is not None:
        assert len(args.models_to_run) % 2 == 0
        for i, model_file in enumerate(args.models_to_run):
            if len(model_file) > 0:
                if i % 2 == 0:
                    model_file = cache_url(model_file, cfg.DOWNLOAD_CACHE)
                    args.models_to_run[i] = model_file
                assert os.path.exists(model_file), \
                    '\'{}\' does not exist'.format(model_file)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    check_args(args)
    main(args)
