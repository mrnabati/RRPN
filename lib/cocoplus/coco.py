"""
An enhanced interface for the Microsoft COCO dataset.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import json
import cv2
import types
import datetime
import time
import pprint
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask
from collections import defaultdict
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from cocoplus.utils.logging import initialize_logger
from cocoplus.utils.coco_utils import show_class_name_plt

class COCO_PLUS(COCO):

    CAT_ID = 0       # Initial value for Category IDs
    ANN_ID = 0       # Initial value for Annotation IDs
    IMG_ID = 0       # Initial value for Image IDs
    PCL_ID = 0       # Initial value for Pointcloud IDs
    STR_ID_LEN = 8   # String ID length for filenames
    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self, 
                 annotation_file=None, 
                 logging_level="INFO"):
        """
        :param annotation_file (str): an existing coco annotation file
        :param logging_level (str): set the logging level (DEBUG, INFO, WARN, ERROR, CRITICAL)
        """

        self.logger = initialize_logger(__name__, level=logging_level)
        self.annotation_file = annotation_file
        self.pointclouds, self.imgToPc = dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
                

        if not annotation_file == None:
            self.logger.info('loading COCO annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, \
                'annotation file format {} not supported'.format(type(dataset))
            
            self.logger.info('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    
    ##-------------------------------------------------------------------------
    def createIndex(self):
        """
        Create index for the annotations
        """

        super(COCO_PLUS, self).createIndex()
        catNameToId = dict()
        pointclouds = dict()
        imgToPc = dict()

        if 'pointclouds' in self.dataset:
            for pc in self.dataset['pointclouds']:
                imgToPc[pc['img_id']] = pc
                pointclouds[pc['id']] = pc

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                catNameToId[cat['name']] = cat['id']

        self.catNameToId = catNameToId
        self.pointclouds = pointclouds
        self.imgToPc = imgToPc
        self.logger.info('index created.')

    ##-------------------------------------------------------------------------
    def create_new_dataset(self,
                           dataset_dir, 
                           split,
                           description="",
                           url="",
                           version="",
                           year=0,
                           contributor="",
                           date_created="",
                           license_url="",
                           license_id=0,
                           license_name=""
                           ):
        """
        Create a new COCO-style dataset
        :param dataset_dir (str): location of new dataset
        :param split (str): Dataset split {'train', 'val', 'test'}
        :param description (str):
        :param url (str):
        :param version (str):
        :param year (int):
        :param contributor (str):
        :param date_created (str):
        :param license_url (str):
        :param license_id (int):
        :param license_name (str):
        """

        self.dataset_dir = os.path.abspath(dataset_dir)
        self.logger.info('Creating an empty COCO dataset at {}'.format(self.dataset_dir))
        assert self.annotation_file is None, \
            "COCO dataset is already initialized with the annotation file: {}".format(self.annotation_file)
        
        ## Create the dataset directory
        self.imgs_dir = os.path.join(dataset_dir, split)
        os.makedirs(self.imgs_dir, exist_ok=True)
        anns_dir = os.path.join(dataset_dir, 'annotations')
        os.makedirs(anns_dir, exist_ok=True)

        self.annotation_file = os.path.join(anns_dir, 
                                            "instances_{}.json".format(split))        
        ## Create class members
        self.catNameToId = {}
        self.pointclouds = {}
        self.imgToPc = {}
        self.dataset = {'annotations':[], 'images':[], 'categories':[], 'pointclouds':[]}
        self.dataset['info'] = {
            "description": description,
            "url": url,
            "version": version,
            "year": year,
            "contributor": contributor,
            "date_created": date_created}
        self.dataset['licenses'] = [{
            "url": license_url,
            "id": license_id,
            "name": license_name}]


    ##-------------------------------------------------------------------------
    def addSample(self,
                  img, 
                  anns, 
                  pointcloud=None,
                  img_id=None,
                  img_format='BGR', 
                  write_img=True,
                  other=None):
        """
        Add a new sample (image + annotations [+ pointcloud]) to the dataset.

        :param img (nparray)
        :param anns (list of dict)
        :param pointcloud (list): list of the points in the pointcloud
        :param img_id (int): image ID 
        :param img_format (str): 'BGR' or 'RGB'
        :param write_img (bool): save the image to the image directory
        :param other (dict): any additional information to be stored in img_info
        """

        # Sanity check
        assert img_format in ['BGR','RGB'], "Image format not supported."
        assert isinstance(anns, (list,)), "Annotations must be provided in a list."
        assert isinstance(img, np.ndarray), "Image must be a numpy array."

        if img_id is None:
            img_id = self._getNewImgId()
        else:
            assert isinstance(img_id, int), "Image ID must be an integer."
            assert img_id not in self.imgs, "Image ID {} already exists.".format(img_id)

        # Create the image info
        heigth, width, _ = img.shape
        img_info = self._createImageInfo(height=heigth, 
                                         width=width, 
                                         img_id=img_id,
                                         other=other)
        # Update the dataset and index
        self.dataset['images'].append(img_info)
        self.imgs[img_id] = img_info
        
        ## Add the new annotations to dataset
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

        ## Add the pointcloud to the dataset if applicable
        if pointcloud is not None:
            assert isinstance(pointcloud,(list,)), "Pointcloud must be a list of points."

            pc_id = self._getNewPclId()
            pc = {'id': pc_id,
                  'img_id': img_id,
                  'points': pointcloud}

            self.dataset['pointclouds'].append(pc)
            self.pointclouds[pc_id] = pc
            self.imgToPc[img_id] = pc
        
        if self.imgs[img_id]['id'] != pc['img_id']:
            raise Exception("Image ID not matching the corresponding pointcloud")

        img_path = os.path.join(self.imgs_dir, img_info['file_name'])
        
        if write_img:
            ## Write the image to disk        
            if img_format == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)
        
        return img_path

    ##-------------------------------------------------------------------------
    def _createImageInfo(self, 
                         height,
                         width, 
                         img_id=None, 
                         license=0, 
                         flickr_url='',
                         coco_url='', 
                         date_captured=None,
                         other=None,
                         ):
        """
        Generate image info in COCO format
        :param height: Image height
        :param width: Image width
        :param img_id : Image ID
        :param license (int)
        :param flickr_url (str)
        :param coco_url (str)
        :param date_captured (str)
        :param other (dict)
        """

        if date_captured is None:
            date_captured = datetime.datetime.utcnow().isoformat(' ')

        if img_id is not None:
            filename = self.imId2name(img_id)
        else:
            filename = None

        img_info={"id" : img_id,
                  "width" : width,
                  "height" : height,
                  "file_name" : filename,
                  "license" : license,
                  "flickr_url" : flickr_url,
                  "coco_url" : coco_url,
                  "date_captured" : date_captured,
                  "other": other
                  }

        return img_info


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
                assert isinstance(new_cat_id, int), "CAT ID must be an integer."
                assert new_cat_id not in self.cats, \
                    "cat_id '{}' already exists.".format(new_cat_id)
                cat_id = new_cat_id

            coco_cat = {'id':cat_id, 'name':category, 'supercategory':supercat}

            # Add the category to the dataset
            self.dataset['categories'].append(coco_cat)
            self.catNameToId[category] = cat_id
            self.cats[cat_id] = coco_cat

        return cat_id


    ##-------------------------------------------------------------------------
    def saveAnnsToDisk(self, ann_file=None):
        """
        Save the annotations to disk
        """

        if ann_file is None:
            ann_file = self.annotation_file

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
    def createAnn(bbox, 
                  cat_id, 
                  img_id=None,
                  segmentation=None, 
                  area=None, 
                  iscrowd=0,
                  id=None):
        """
        Create an annotation in COCO annotation format
        :param bbox (list):
        :param cat_id (int):
        :param img_id (int):
        :param segmentation (list):
        :param area (float):
        :param iscrowd (bool):
        :param id (int):
        """

        bbox = [float(format(elem, '.2f')) for elem in bbox]

        if segmentation is None:
            segmentation = []

        if area is None:
            # Use bounding box area
            area = bbox[2] * bbox[3]
        area = np.float32(area).tolist()

        annotation = {
            "id": id,
            "image_id": img_id,
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
        :param im_height (int): Image height
        :param im_width (int): Image width
        :return: RLE
        """

        assert type(poly) == list, "Poly must be a list of polygon vertices"

        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask.frPyObjects(poly, im_height, im_width)
        rle = mask.merge(rles)

        return rle

    ##-------------------------------------------------------------------------
    def imId2name(self, im_id):
        """
        Returns the COCO image name given its ID
        :im_id (int): image ID
        :return (str): image name
        """
        
        if isinstance(im_id, int):
            name = str(im_id).zfill(self.STR_ID_LEN) + '.jpg'
        elif isinstance(im_id, str):
            name = im_id + '.jpg'
        else:
            raise AssertionError('Image ID should be of type string or int')
        return name

    ##-------------------------------------------------------------------------
    def showImgAnn(self, img, anns=None, bbox_only=False, BGR=True, ax=None):
        """
        Display an image and its annotations

        :param img (numpy array): The background image
        :param ann (list): A list of annotations. If empty, only the image
            is displayed.
        :param BGR (binary): True for BGR, False for RGB
        """

        plt.cla()
        if BGR:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

        ax.imshow(img); 
        plt.axis('off')

        if anns is not None:
            ax = self.showAnns(anns, bbox_only, ax)
        # plt.show()
        return ax


    ##-------------------------------------------------------------------------
    def showAnns(self, anns, bbox_only=False, ax=None):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :param bbox_only (bool):
        :param ax (axis):
        :return: None
        """

        if len(anns) == 0:
            return 0

        if ax is None:
                ax = plt.gca()
                ax.set_autoscale_on(False)
        
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        
        if datasetType == 'instances':
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

                    cat_id = ann['category_id']
                    clss_txt = str(cat_id) + ':' + self.cats[cat_id]['name']
                    show_class_name_plt([bbox_x, bbox_y], clss_txt, ax, c)
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
                            rle = mask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = mask.decode(rle)
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

        return ax