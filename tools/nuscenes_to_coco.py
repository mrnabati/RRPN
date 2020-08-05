import _init_path
import os
import sys
import pickle
import numpy as np
import argparse
from tqdm import tqdm, trange
from cocoplus.coco import COCO_PLUS
from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    
    parser.add_argument('--nusc_root', default='../data/nuscenes',
                        help='NuScenes dataroot')
    
    parser.add_argument('--split', default='mini_train',
                        help='Dataset split (mini_train, mini_val, train, val, test)')

    parser.add_argument('--out_dir', default='../data/nucoco/',
                        help='Output directory for the nucoco dataset')

    parser.add_argument('--nsweeps_radar', default=1, type=int,
                        help='Number of Radar sweeps to include')

    parser.add_argument('--use_symlinks', default='False',
                        help='Create symlinks to nuScenes images rather than copying them')

    parser.add_argument('--cameras', nargs='+',
                        default=['CAM_FRONT',
                                 'CAM_BACK',
                                #  'CAM_FRONT_LEFT',
                                #  'CAM_FRONT_RIGHT',
                                #  'CAM_BACK_LEFT',
                                #  'CAM_BACK_RIGHT',
                                 ],
                        help='List of cameras to use.')
    
    parser.add_argument('-l', '--logging_level', default='INFO',
                        help='Logging level')
                        
    args = parser.parse_args()
    return args

#-------------------------------------------------------------------------------
def main():
    args = parse_args()

    if "mini" in args.split:
        nusc_version = "v1.0-mini"
    elif "test" in args.split:
        nusc_version = "v1.0-test"
    else:
        nusc_version = "v1.0-trainval"

    ## Categories: [category, supercategory, category_id]
    categories = [['person',      'person' ,  1],
                  ['bicylce',     'vehicle',  2],
                  ['car',         'vehicle',  3],
                  ['motorcycle',  'vehicle',  4],
                  ['bus',         'vehicle',  5],
                  ['truck',       'vehicle',  6]
    ]
    
    ## Short split is used for filenames
    anns_file = os.path.join(args.out_dir, 'annotations', 'instances_' + args.split + '.json')

    nusc_dataset = NuscenesDataset(nusc_path=args.nusc_root, 
                                   nusc_version=nusc_version, 
                                   split=args.split,
                                   coordinates='vehicle',
                                   nsweeps_radar=args.nsweeps_radar, 
                                   sensors_to_return=['camera', 'radar'],
                                   pc_mode='camera',
                                   logging_level=args.logging_level)
    
    coco_dataset = COCO_PLUS(logging_level="INFO")
    coco_dataset.create_new_dataset(dataset_dir=args.out_dir, split=args.split)

    ## add all category in order to have consistency between dataset splits
    for (coco_cat, coco_supercat, coco_cat_id) in categories:
        coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)
    
    ## Get samples from the Nuscenes dataset
    num_samples = len(nusc_dataset)
    for i in trange(num_samples):
        sample = nusc_dataset[i]
        img_ids = sample['img_id']

        for i, cam_sample in enumerate(sample['camera']):
            if cam_sample['camera_name'] not in args.cameras:
                continue

            img_id = int(img_ids[i])
            image = cam_sample['image']
            pc = sample['radar'][i]
            cam_cs_record = cam_sample['cs_record']
            img_height, img_width, _ = image.shape

            # Create annotation in coco_dataset format
            sample_anns = []
            annotations = nusc_dataset.pc_to_sensor(sample['annotations'][i], 
                                                    cam_cs_record)
        
            for ann in annotations:
                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(ann.name)
                ## if not a valid category, go to the next annotation
                if coco_cat is None:
                    coco_dataset.logger.debug('Skipping ann with category: {}'.format(ann.name))
                    continue
                
                cat_id = coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)
                bbox = nuscenes_box_to_coco(ann, np.array(cam_cs_record['camera_intrinsic']), 
                                            (img_width, img_height))
                coco_ann = coco_dataset.createAnn(bbox, cat_id)
                sample_anns.append(coco_ann)

            ## Map the Radar pointclouds to image
            pc_cam = nusc_dataset.pc_to_sensor(pc, cam_cs_record)
            pc_depth = pc_cam[2, :]
            pc_image = view_points(pc_cam[:3, :], 
                                 np.array(cam_cs_record['camera_intrinsic']), 
                                 normalize=True)
            
            ## Add the depth information to each point
            pc_coco = np.vstack((pc_image[:2,:], pc_depth))
            pc_coco = np.transpose(pc_coco).tolist()

            ## Add sample to the COCO dataset
            coco_img_path = coco_dataset.addSample(img=image,
                                           anns=sample_anns, 
                                           pointcloud=pc_coco,
                                           img_id=img_id,
                                           other=cam_cs_record,
                                           img_format='RGB',
                                           write_img= not args.use_symlinks,
                                           )
            
            if args.use_symlinks:
                try:
                    os.symlink(os.path.abspath(cam_sample['cam_path']), coco_img_path)
                except FileExistsError:
                    pass
            
            ## Uncomment to visualize
            # coco_dataset.showImgAnn(np.asarray(image), sample_anns, bbox_only=True, BGR=False)
        
    coco_dataset.saveAnnsToDisk()

if __name__ == '__main__':
    main()
