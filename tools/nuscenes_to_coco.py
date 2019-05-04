# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

import _init_paths
import os
import sys
import matplotlib
import random
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from tqdm import tqdm
from pycocotools_plus.coco import COCO_PLUS
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud
from nuscenes_utils.radar_utils import *
from datasets import nuscene_cat_to_coco

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    parser.add_argument('--nuscene_root', dest='dataroot',
                        help='NuScenes dataroot',
                        default='../data/nuscenes')

    parser.add_argument('--train_ann_file', dest='train_ann_file',
                        help='Train annotations file',
                        default='../output/nucoco/annotations/instances_train.json')

    parser.add_argument('--val_ann_file', dest='val_ann_file',
                        help='Validation annotations file',
                        default='../output/nucoco/annotations/instances_val.json')

    parser.add_argument('--train_imgs_dir', dest='train_imgs_dir',
                        help='Train output directory',
                        default='../output/nucoco/train')

    parser.add_argument('--val_imgs_dir', dest='val_imgs_dir',
                        help='Validation output directory',
                        default='../output/nucoco/val')

    parser.add_argument('--train_ratio', dest='train_ratio', type=float,
                        help='Train samples ratio to all samples (e.g. 0.90)',
                        default=0.85)

    parser.add_argument('--include_sweeps', dest='include_sweeps',
                        help='If True, include the non key-frame data in dataset')

    args = parser.parse_args()

    assert args.train_ratio >= 0 and args.train_ratio <= 1, \
        "--train_ratio must be in range [0 1]"
    assert args.include_sweeps in ['True','False'], \
        "--include_sweeps must be 'True' or 'False'"

    return args


#-------------------------------------------------------------------------------
if __name__ == '__main__':
    random.seed(13)
    args = parse_args()

    nusc = NuScenes(version='v0.1', dataroot=args.dataroot, verbose=True)
    coco_train = COCO_PLUS(args.train_ann_file, args.train_imgs_dir, new_dataset=True)
    coco_val = COCO_PLUS(args.val_ann_file, args.val_imgs_dir, new_dataset=True)

    for i in tqdm(range(0, len(nusc.scene))):
        scene = nusc.scene[i]
        scene_rec = nusc.get('scene', scene['token'])
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])

        ## Get front sensors data
        f_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
        f_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT'])

        ## Get rear sensors data
        b_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_BACK'])
        br_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_RIGHT'])
        bl_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_LEFT'])

        has_more_data = True
        while has_more_data:
            rnd = random.uniform(0, 1)
            anns_f = []
            anns_b = []
            f_camera_token = f_cam_rec['token']
            b_camera_token = b_cam_rec['token']
            f_radar_token = f_rad_rec['token']
            br_radar_token = br_rad_rec['token']
            bl_radar_token = bl_rad_rec['token']

            ## FRONT CAM + RADAR
            impath_f, boxes_f, camera_intrinsic_f = nusc.get_select_sample_data(
                f_camera_token, min_ann_vis_level=1)
            points_f, coloring_f, image_f = nusc.explorer.map_pointcloud_to_image(
                f_radar_token, f_camera_token)
            points_f[2, :] = coloring_f

            ## Back CAM + RADARs
            impath_b, boxes_b, camera_intrinsic_b = nusc.get_select_sample_data(
                b_camera_token, min_ann_vis_level=1)
            points_br, coloring_br, image_b = nusc.explorer.map_pointcloud_to_image(
                br_radar_token, b_camera_token)
            points_br[2, :] = coloring_br
            points_bl, coloring_bl, _ = nusc.explorer.map_pointcloud_to_image(
                bl_radar_token, b_camera_token)
            points_bl[2, :] = coloring_bl

            ## Concatenate the two back Radar detections
            points_b = np.hstack((points_br,points_bl))

            # Convert to list of points
            points_f = np.squeeze(np.dstack((points_f[0,:], points_f[1,:], points_f[2,:]))).tolist()
            points_b = np.squeeze(np.dstack((points_b[0,:], points_b[1,:], points_b[2,:]))).tolist()
            if not isinstance(points_f[0], list) and len(points_f) == 3:
                points_f = [points_f]
                print(points_f)
            if not isinstance(points_b[0], list) and len(points_b) == 3:
                points_b = [points_b]
                print(points_b)

            for box in boxes_f:
                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(box.name)
                if coco_cat is None:
                    continue

                cat_id = coco_train.addCategory(coco_cat, coco_supercat, int(coco_cat_id))
                cat_id_2 = coco_val.addCategory(coco_cat, coco_supercat, int(coco_cat_id))
                assert cat_id == cat_id_2, "cat_id mismatch"

                # Create annotation in COCO format
                bbox = box.to_coco_bbox(camera_intrinsic_f, image_f.size)
                coco_ann = COCO_PLUS.createAnn(bbox, cat_id)
                anns_f.append(coco_ann)

            for box in boxes_b:
                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(box.name)
                if coco_cat is None:
                    continue

                cat_id = coco_train.addCategory(coco_cat, coco_supercat, int(coco_cat_id))
                cat_id_2 = coco_val.addCategory(coco_cat, coco_supercat, int(coco_cat_id))
                assert cat_id == cat_id_2, "cat_id mismatch"

                bbox = box.to_coco_bbox(camera_intrinsic_b, image_b.size)
                coco_ann = COCO_PLUS.createAnn(bbox, cat_id)
                anns_b.append(coco_ann)

            if rnd <= args.train_ratio:
                coco_train.addSample(np.asarray(image_f), anns_f, points_f,
                                     img_format='RGB')
                coco_train.addSample(np.asarray(image_b), anns_b, points_b,
                                     img_format='RGB')
            else:
                coco_val.addSample(np.asarray(image_f), anns_f, points_f,
                                    img_format='RGB')
                coco_val.addSample(np.asarray(image_b), anns_b, points_b,
                                    img_format='RGB')

            ## -----------------------------------------------------------------
            if args.include_sweeps == 'True':
                # Get the next Sweep
                if not f_cam_rec['next'] == "" and not f_rad_rec['next'] == "" \
                    and not b_cam_rec['next'] == "" and not br_rad_rec['next'] == "" \
                    and not bl_rad_rec['next'] == "":

                    f_cam_rec = nusc.get('sample_data', f_cam_rec['next'])
                    f_rad_rec = nusc.get('sample_data', f_rad_rec['next'])

                    b_cam_rec = nusc.get('sample_data', b_cam_rec['next'])
                    br_rad_rec = nusc.get('sample_data', br_rad_rec['next'])
                    bl_rad_rec = nusc.get('sample_data', bl_rad_rec['next'])
                else:
                    has_more_data = False

            else:
                # Get the next Sample
                if not sample_rec['next'] == "":
                    sample_rec = nusc.get('sample', sample_rec['next'])
                    f_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_FRONT'])
                    f_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_FRONT'])

                    b_cam_rec = nusc.get('sample_data', sample_rec['data']['CAM_BACK'])
                    br_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_RIGHT'])
                    bl_rad_rec = nusc.get('sample_data', sample_rec['data']['RADAR_BACK_LEFT'])

                else:
                    has_more_data = False

    coco_train.saveAnnsToDisk()
    coco_val.saveAnnsToDisk()
