
import _init_path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import argparse
from tqdm import tqdm
from visualization import draw_xyxy_bbox, draw_points, save_fig
from cocoplus.coco import COCO_PLUS
from rrpn_generator import rrpn_loader
from detectron.utils.io import load_object


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Test the object proposals file')

    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../data/nucoco/v1.0-mini/annotations/instances_train.json')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../data/nucoco/v1.0-mini/train')

    parser.add_argument('--proposals', dest='proposals',
                        help='Proposals file',
                        default='../data/nucoco/v1.0-mini/proposals/proposals_train.pkl')
    
    parser.add_argument('--out_dir', dest='out_dir',
                        help='Test outputs directory',
                        default='../data/test')

    args = parser.parse_args()
    return args

##------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    fig = plt.figure(figsize=(16, 9))

    # Load the dataset and proposals
    coco = COCO_PLUS(args.ann_file)
    proposals = load_object(args.proposals)
    num_imgs = len(coco.imgs)
    assert len(proposals['ids']) == num_imgs, \
        "Number of proposals do not match the number of images in the dataset"

    step = 1
    for i in tqdm(range(0, num_imgs, step)):
        img_id = proposals['ids'][i]
        points = coco.imgToPc[img_id]['points']
        img_path = os.path.join(args.imgs_dir, coco.imgs[img_id]["file_name"])
        img = np.array(plt.imread(img_path))
        boxes = proposals['boxes'][i]

        ## Plot proposals
        img = draw_xyxy_bbox(img, list(boxes), lineWidth=2)
        img = draw_points(img, points, color=(0,0,255), radius=5, thickness=-1, format='RGB')
        ax = fig.add_subplot(1,2,1)
        ax.imshow(img)
        plt.axis('off')
        
        ## Save the image to disk
        out_filename = os.path.join(args.out_dir, 'rrpn_' + str(img_id)+'.pdf')
        save_fig(out_filename)
        print('Results saved in: ', out_filename)
        input("Press Enter to continue...")

        # plt.show(block=False)
        # # plt.show()
        # plt.pause(0.5)
        plt.clf()
