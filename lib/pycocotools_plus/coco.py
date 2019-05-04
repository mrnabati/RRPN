# --------------------------------------------------------
# RRPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ramin Nabati
# --------------------------------------------------------

"""
An enhanced interface for the Microsoft COCO dataset.

This API adds more functionalities to the original COCO API available here:
https://github.com/cocodataset/cocoapi
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import os
import json
import cv2
import types
import sys
import datetime
import copy
import time

from tqdm import tqdm
from skimage import measure
from pycocotools.coco import COCO
from collections import defaultdict
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class COCO_PLUS(COCO):

    CAT_ID = 100            # Initial value for new Category IDs
    ANN_ID = 100000000000   # Initial value for new Annotation IDs
    IMG_ID = 10000000       # Initial value for new Image IDs
    PCL_ID = 10000000       # # Initial value for new Pointcloud IDs

    def __init__(self, ann_file, imgs_dir, new_dataset=False, info=None):
        """
        Constructor of helper class for generating, reading and visualizing
        annotations for the coco dataset.
        :param
        :return:
        """

        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.catNameToId, self.pointclouds, self.imgToPointcloud = dict(), dict(), dict()

        self.imgs_dir = imgs_dir
        self.ann_file = ann_file
        anns_dir = os.path.dirname(ann_file)

        if new_dataset:
            print('Creating new COCO-style dataset.')
            os.makedirs(imgs_dir, exist_ok=True)
            os.makedirs(anns_dir, exist_ok=True)
            self.initiate_dataset(info)

        else:
            if not os.path.exists(ann_file):
                raise Exception("Annotation file '{}' not found.".format(ann_file))
            if not os.path.exists(imgs_dir):
                raise Exception('COCO images directory not found.')

            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(ann_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()


    ##-------------------------------------------------------------------------
    def initiate_dataset(self, info):

        self.dataset = {'annotations':[], 'images':[], 'categories':[]}

        if info is not None:
            self.dataset['info'] = info['info']
            self.dataset['licenses'] = info['licenses']

        else:
            self.dataset['info'] = {
                "description": "...",
        		"url": "...",
        		"version": "1.0",
        		"year": 2018,
        		"contributor": "...",
        		"date_created": "..."
            }
            self.dataset['licenses'] = [
                {
    			"url": "...",
    			"id": 1,
    			"name": "..."
                }
            ]

    ##-------------------------------------------------------------------------
    def createIndex(self):
        """
        Create index for the annotations
        """

        catNameToId = dict()
        pointclouds = dict()
        imgToPointcloud = dict()
        super(COCO_PLUS, self).createIndex()

        if 'pointclouds' in self.dataset:
            for pc in self.dataset['pointclouds']:
                imgToPointcloud[pc['img_id']] = pc
                pointclouds[pc['id']] = pc

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                catNameToId[cat['name']] = cat['id']

        self.catNameToId = catNameToId
        self.pointclouds = pointclouds
        self.imgToPointcloud = imgToPointcloud
        print('index created.')



    ##-------------------------------------------------------------------------
    def createImageInfo(self, img, img_id=None, license=0, flickr_url='N/A',
                        coco_url='N/A', date_captured=None):
        """
        Generate image info in COCO format

        :param img: nparray
        :param img_id
        """

        if date_captured is None:
            date_captured = datetime.datetime.utcnow().isoformat(' ')

        if img_id is not None:
            filename = self.imId2name(img_id)
        else:
            filename = None

        img_height, img_width = img.shape[0], img.shape[1]

        img_info={"id" : img_id,
                  "width" : img_width,
                  "height" : img_height,
                  "file_name" : filename,
                  "license" : license,
                  "flickr_url" : flickr_url,
                  "coco_url" : coco_url,
                  "date_captured" : date_captured
                  }

        return img_info


    ##-------------------------------------------------------------------------
    def addSample(self, img, anns, pointcloud=None, img_format='BGR'):
        """
        Add a new image and its annotations to the dataset.

        :param img (nparray)
        :param anns (list of dict)
        :param pointcloud (list): list of the points in the pointcloud
        :param img_format (str): 'BGR' or 'RGB'
        """

        # Sanity check
        assert img_format=='BGR' or img_format=='RGB', "Image format not supported."
        assert isinstance(anns, (list,)), "Annotations must be provided in a list."
        assert isinstance(img, np.ndarray), "Image must be a numpy array."

        # Create image info
        img_id = self._getNewImgId()
        img_info = self.createImageInfo(img, img_id)

        # Update the dataset and index
        self.dataset['images'].append(img_info)
        self.imgs[img_id] = img_info

        ## Add the new annotation to the dataset
        for ann in anns:
            assert ann['category_id'] in self.cats, \
            "Category '{}' does not exist in dataset.".format(ann['category_id'])

            ann['image_id'] = img_id
            if ann['id'] is None:
                ann['id'] = self._getNewAnnId()

            # Update the dataset and index
            self.dataset['annotations'].append(ann)
            self.anns[ann['id']] = ann
            self.catToImgs[ann['category_id']].append(ann['image_id'])
            self.imgToAnns[img_id].append(ann)

        ## Add the new pointcloud to the dataset if applicable
        if pointcloud is not None:
            assert isinstance(pointcloud,(list,)), "Pointcloud must be a list of points."

            pc_id = self._getNewPclId()
            pc = {'id': pc_id,
                  'img_id': img_id,
                  'points': pointcloud}

            if 'pointclouds' not in self.dataset:
                self.dataset['pointclouds'] = []

            self.dataset['pointclouds'].append(pc)
            self.pointclouds[pc_id] = pc
            self.imgToPointcloud[img_id] = pc

        if img_format == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        ## Save the new image to disk
        img_path = os.path.join(self.imgs_dir, img_info['file_name'])
        cv2.imwrite(img_path, img)

    ##-------------------------------------------------------------------------
    def addCategory(self, category, supercat, new_cat_id=None):

        """
        Add a new category to the dataset

        :param category
        :param supercategory
        """

        try:
            # If category already exists, just return the cat_id
            cat_id = self.catNameToId[category]

        except KeyError:
            # Add the new category
            if new_cat_id is None:
                cat_id = self._getNewCatId()

            else:
                assert new_cat_id not in self.cats, \
                "cat_id '{}' already exists.".format(new_cat_id)
                cat_id = new_cat_id

            coco_cat = {'id':cat_id, 'name':category, 'supercategory':supercat}

            # Add the category to the dataset
            self.dataset['categories'].append(coco_cat)
            self.catNameToId[category] = cat_id
            self.cats[cat_id] = coco_cat

        except:
            raise

        return cat_id


    ##-------------------------------------------------------------------------
    def saveAnnsToDisk(self, ann_file=None):
        """
        Save the annotations to disk
        """

        ## Write the annotations to file
        if ann_file is None:
            ann_file = self.ann_file

        with open(ann_file, 'w') as fp:
            json.dump(self.dataset, fp)


    ##-------------------------------------------------------------------------
    def _getNewImgId(self):
        """ Generate a new image ID

        :return (int): img_id
        """

        newImgId = COCO_PLUS.IMG_ID
        COCO_PLUS.IMG_ID += 1

        return newImgId


    ##-------------------------------------------------------------------------
    def _getNewPclId(self):
        """ Generate a new pointcloud ID

        :return (int): pc_id
        """
        newPclId = COCO_PLUS.PCL_ID
        COCO_PLUS.PCL_ID += 1

        return newPclId

    ##-------------------------------------------------------------------------
    def _getNewAnnId(self):
        """
        Generate a new annotation ID

        :return (int): ann_id
        """

        newAnnId = COCO_PLUS.ANN_ID
        COCO_PLUS.ANN_ID += 1

        return newAnnId

    ##-------------------------------------------------------------------------
    def _getNewCatId(self):
        """
        Generate a new category ID

        :return (int): cat_id
        """
        newCatId = COCO_PLUS.CAT_ID
        COCO_PLUS.CAT_ID += 1

        return newCatId


    ##-------------------------------------------------------------------------
    @staticmethod
    def createAnn(bbox, cat_id, segmentation=None, area=None, iscrowd=0,
                  ann_id=None):
        """
        Create an annotation in COCO annotation format
        """

        bbox = [float(format(elem, '.2f')) for elem in bbox]

        if segmentation is None:
            segmentation = []

        if area is None:
            # Use bounding box area
            area = bbox[2] * bbox[3]
        area = np.float32(area).tolist()

        annotation = {
            "id": ann_id,
            "image_id": None,
            "category_id": cat_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd
            }

        return annotation

    ##-------------------------------------------------------------------------
    def poly2rle(self, poly, im_height, im_width):
        """
        Convert polygon annotation segmentation to RLE.
        :param poly (list): Input polygon
        :param
        :param
        :return: RLE
        """
        assert type(poly) == list, "Poly must be a list of polygon vertices"

        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        h, w = im_height, im_width
        rles = mask.frPyObjects(poly, h, w)
        rle = mask.merge(rles)

        return rle

    ##-------------------------------------------------------------------------
    def imId2path(self, im_id):
        """
        Returns the COCO image path given its ID
        :im_id (int): image ID
        :return (str): image path
        """

        file_name = self.imgs[im_id]["file_name"]
        file_path = os.path.join(self.imgs_dir, file_name)

        return file_path

    ##-------------------------------------------------------------------------
    @staticmethod
    def imId2name(im_id):
        """
        Returns the COCO image name given its ID
        :im_id (int): image ID
        :return (str): image name
        """

        name = str(im_id).zfill(12) + '.jpg'
        return name

    ##-------------------------------------------------------------------------
    def showImgAnn(self, img, anns=None, bbox_only=False, BGR=True, ax=None):
        """
        Display an image and its annotations

        :param img (numpy array): The background image
        :param ann (list): A list of annotations. If empty, only the image
            is displayed.
        :param BGR (binary): Image format, BGR or RGB
        """
        plt.cla()
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img); plt.axis('off')

        if anns is not None:
            self.showAnns(anns, bbox_only, ax)
        plt.show()


    ##-------------------------------------------------------------------------
    def showAnns(self, anns, bbox_only=False, ax=None):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            if ax is None:
                ax = plt.gca()
                ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6 + 0.4).tolist()[0]

                if bbox_only:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)
                    continue

                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])
