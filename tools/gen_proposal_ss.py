import _init_paths
import numpy as np
import argparse
import time
import sys
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
import selective_search.selective_search as ss
import detectron.datasets.dataset_catalog as dataset_catalog
from pycocotools_plus.coco import COCO_PLUS
from detectron.utils.io import save_object


## Generate proposals for the NuCOCO dataset using the Selective Search method.

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Generate object proposals from Radar pointclouds.')
    parser.add_argument('--dataset', dest='dataset', default='nucoco_train',
                        choices=['nucoco_train', 'nucoco_val'],
                        help='Dataset name according to dataset_catalog')

    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../output/nucoco/annotations/instances_train.json')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../output/nucoco/train')

    parser.add_argument('--out_dir', dest='out_dir', default='../output/proposals',
                        help='Output directory')

    args = parser.parse_args()
    args.ann_file = os.path.abspath(args.ann_file)
    args.imgs_dir = os.path.abspath(args.imgs_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    return args



if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    pkl_file_out = os.path.join(args.out_dir, 'proposals_ss_{}.pkl'.format(dataset))
    mat_file_out = os.path.join(args.out_dir, 'proposals_ss_{}.mat'.format(dataset))
    list_filename = os.path.join(args.out_dir, 'image_list.txt')

    boxes = []
    scores = []
    ids = []
    image_names = []

    # Load the nucoco dataset
    coco = COCO_PLUS(args.ann_file, args.imgs_dir)

    with open(list_filename, 'w') as f:
        n_imgs = 10
        for img_id, img_info in coco.imgs.items():
            n_imgs -= 1
            img_path = os.path.join(args.imgs_dir, img_info["file_name"])
            image_names.append(img_path)
            f.write("%s\n" % img_path)
            if n_imgs == 0:
                break

    input('list is done.')
    t = time.time()
    boxes = ss.get_windows(list_filename, 'selective_search', mat_file_out)
    input('get_windows is done.')
    # Delete the temp file
    #os.remove(list_filename)

    print(boxes[:2])
    print("Processed {} images in {:.3f} s".format(
        len(image_names), time.time() - t))

    save_object(boxes, '../output/proposals/proposals_ss_nucoco_val.pkl')
