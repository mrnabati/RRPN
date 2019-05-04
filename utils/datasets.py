# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

import _init_paths
import os
import sys
import cv2

from pycocotools_plus.coco import COCO_PLUS


def get_nucoco_dataset():
    """A dummy NuCOCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

##------------------------------------------------------------------------------
def nuscene_cat_to_coco(name):

    ## Convert nuscene super categories to COCO super categories
    supercat_dict = {'human': 'person',
                     'vehicle': 'vehicle'
                     }

    ## Convert nuscene categories to COCO cat and cat_ids
    cat_dict = {'pedestrian': ['person','1'],
                'car': ['car', '3'],
                'emergency': ['car', '3'],
                'truck': ['truck','8'],
                'construction': ['truck','8'],
                'trailer': ['truck','8'],
                'motorcycle': ['motorcycle','4'],
                'bicycle': ['bicycle','2'],
                'bus': ['bus','6']
                }

    parts = name.split('.')

    try:
        supercat = supercat_dict[parts[0]]
    except KeyError:
        return None, None, None

    try:
        cat, cat_id = cat_dict[parts[1]]
    except KeyError:
        return None, None, None

    return cat, cat_id, supercat


    # allowed_supercats = {'human', 'vehicle'}
    # allowed_cats = {'pedestrian','car','bicycle','motorcycle','bus','truck',
    #                    'emergency','trailer'}
    #
    # name_to_cat={'human': 'person',
    #              'pedestrian': 'person',
    #              'construction': 'trcuk',
    #              'emergency': 'car',
    #              'trailer': 'truck'}
    #
    # parts = name.split('.')
    #
    # supercat = parts[0]
    # if supercat not in allowed_supercats:
    #     return None, None
    #
    # cat = parts[1]
    # if cat not in allowed_cats:
    #     return None, None
    #
    # try:
    #     coco_cat = name_to_cat[cat]
    # except:
    #     coco_cat = cat
    #
    # try:
    #     coco_supercat = name_to_cat[supercategory]
    # except:
    #     coco_supercat = supercat
    #
    # return coco_cat, coco_supercat

##------------------------------------------------------------------------------
def get_coco_img_list(imgs_dir, ann_file, out_file=None):

    img_paths = []

    # Load the nucoco dataset
    coco = COCO_PLUS(ann_file, imgs_dir)

    if out_file is not None:
        f = open(out_file,'w')

    for img_id, img_info in coco.imgs.items():
        img_path = os.path.join(imgs_dir, img_info["file_name"])
        img_paths.append(img_path)
        if out_file is not None:
            f.write("%s\n" % img_path)

    if out_file is not None:
        f.close()

    return img_paths
