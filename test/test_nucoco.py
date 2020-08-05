
import _init_path
import cv2
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint
from cocoplus.coco import COCO_PLUS
from visualization import save_fig


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Explore NuCOCO')
    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../data/nucoco/v1.0-mini/annotations/instances_train.json')
    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../data/nucoco/v1.0-mini/train')
    parser.add_argument('--out_dir', dest='out_dir',
                        help='Test outputs directory',
                        default='../data/test')

    args = parser.parse_args()
    return args

##------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    coco = COCO_PLUS(args.ann_file)

    print("Object Categories:\n")
    for key, val in coco.cats.items():
        pprint(val)
    
    # Open samples from the dataset
    num_imgs = len(coco.imgs)
    for i in tqdm(range(0, num_imgs)):
        img_id = coco.dataset['images'][i]['id']
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        pointcloud = coco.imgToPc[img_id]['points']
        pc = np.array(pointcloud)
        img_path = os.path.join(args.imgs_dir, coco.imgs[img_id]["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## Visualize the image, annotations and pointclouds
        _, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax = coco.showImgAnn(img, anns, bbox_only=True, BGR=False, ax=ax)
        scatter = ax.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], s=10)
        ax.axis('off')

        ## Save the image to disk
        out_filename = os.path.join(args.out_dir, str(img_id)+'.pdf')
        save_fig(out_filename)
        print('Results saved in: ', out_filename)
        input("Press Enter to continue...")
        plt.cla()
