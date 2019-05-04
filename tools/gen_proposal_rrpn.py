#!/usr/bin/env python

"""
Script to generate object proposals from the Radar pointclouds in the nucoco
dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths
import numpy as np
import scipy.io as sio
import argparse
import sys
import os
import cv2
# import matplotlib.pyplot as plt

from tqdm import tqdm
from detectron.utils.boxes import clip_boxes_to_image
# import detectron.datasets.dataset_catalog as dataset_catalog
# from detectron.datasets.json_dataset import JsonDataset
from detectron.utils.io import save_object
from pycocotools_plus.coco import COCO_PLUS
from datasets import nuscene_cat_to_coco
from rrpn_generator import get_im_proposals
from visualization import draw_xyxy_bbox


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Generate object proposals from Radar pointclouds.')

    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../output/nucoco/annotations/instances_train.json')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../output/nucoco/train')

    parser.add_argument('--output_file', dest='output_file',
                        help='Output filename',
                        default='../output/proposals/proposal.pkl')
    
    parser.add_argument('--include_depth', dest='include_depth',
                        help='If 1, include depth information from radar',
                        default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_file = args.output_file
    boxes = []
    scores = []
    ids = []
    img_ind = 0

    ## Load the nucoco dataset
    coco = COCO_PLUS(args.ann_file, args.imgs_dir)

    for img_id, img_info in tqdm(coco.imgs.items()):
        img_ind += 1

        if int(args.include_depth)==1:
            proposals = np.empty((0,5), np.float32)
        else:
            proposals = np.empty((0,4), np.float32)

        img_width = img_info['width']
        img_height = img_info['height']

        pointcloud = coco.imgToPointcloud[img_id]
        # pointcloud = coco.pointclouds[pc_id]

        # Generate proposals for points in pointcloud
        for point in pointcloud['points']:
            rois = get_im_proposals(point, 
                                    sizes=(128, 256, 512, 1024),
                                    aspect_ratios=(0.5, 1, 2),
                                    layout=['center','top','left','right'],
                                    beta=8,
                                    include_depth=args.include_depth)
            proposals = np.append(proposals, np.array(rois), axis=0)
            
            ## Plot the proposal boxes
            # img = cv2.imread(coco.imId2path(img_id))
            # if args.include_depth:
            #     distances = proposals[:,-1].tolist()
            #     distances = [str(d) for d in distances]
            #     plotted_image = draw_xyxy_bbox(img, proposals.tolist(), names=distances)
            #     cv2.imwrite('../output/out_bboxes.png', plotted_image)
            #     input('something')
            #     ax = plt.subplot(111)
            #     ax.imshow(plotted_image)
            #     plt.show()

            # plot_xyxy_bbox(img, proposals.tolist())
            # plt.show(block=False)
            # # plt.show()
            # plt.pause(0.1)
            # plt.clf()
            # input('something')

        # Clip the boxes to image boundaries
        proposals = clip_boxes_to_image(proposals, img_height, img_width)

        # # if img_ind % 300 == 0:
        # plot_xyxy_bbox(img, proposals.tolist())
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.clf()

        boxes.append(proposals)
        scores.append(np.zeros((proposals.shape[0]), dtype=np.float32))
        ids.append(img_id)

    print('Saving proposals to disk...')
    save_object(dict(boxes=boxes, scores=scores, ids=ids), output_file)
    print('Proposals saved to {}'.format(output_file))
