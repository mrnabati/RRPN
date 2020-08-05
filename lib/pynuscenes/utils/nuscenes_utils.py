#!/usr/bin/env python3
################################################################################
## Date Created  : Fri Jun 14 2019                                            ##
## Authors       : Landon Harris, Ramin Nabati                                ##
## Last Modified : September 2nd, 2019                                        ##
## Copyright (c) 2019                                                         ##
################################################################################

import numpy as np
import math
from pyquaternion import Quaternion
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points
from shapely.geometry import LineString
from pynuscenes.utils import constants as NS_C


def bbox_to_corners(bboxes):
    """
    Convert 3D bounding boxes in [x,y,z,w,l,h,ry] format to [x,y,z] coordinates
    of the corners
    :param bboxes: input boxes np.ndarray <N,7>
    :return corners: np.ndarray <N,3,8> where x,y,z is along each column
    """
    x = np.expand_dims(bboxes[:,0], 1)
    y = np.expand_dims(bboxes[:,1], 1)
    z = np.expand_dims(bboxes[:,2], 1)

    w = np.expand_dims(bboxes[:,3], 1)
    l = np.expand_dims(bboxes[:,4], 1)
    h = np.expand_dims(bboxes[:,5], 1)
    
    yaw_angle = bboxes[:,6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = (l / 2) * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = (w / 2) * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = (h / 2) * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.dstack((x_corners, y_corners, z_corners))
    # Rotate
    for i, box in enumerate(corners):
        rotation_quat = Quaternion(axis=(0, 0, 1), angle=yaw_angle[i])
        corners[i,:,:] = np.dot(rotation_quat.rotation_matrix, box.T).T

    # Translate
    corners[:,:,0] += x
    corners[:,:,1] += y
    corners[:,:,2] += z

    corners = np.swapaxes(corners, 1,2)
    return corners

##------------------------------------------------------------------------------
def quaternion_to_ry(quat: Quaternion):
    v = np.dot(quat.rotation_matrix, np.array([1,0,0]))
    yaw = np.arctan2(v[1], v[0])
    return yaw

##------------------------------------------------------------------------------
def corners3d_to_image(corners, cam_cs_record, img_shape):
    """
    Return the 2D box corners mapped to the image plane
    :param corners (np.array <N, 3, 8>)
    :param cam_cs_record (dict): calibrated sensor record of a camera dictionary from nuscenes dataset
    :param img_shape (tuple<width, height>)
    :return (ndarray<N,2,8>, list<N>)
    """
    
    cornerList = []
    mask = []
    for box_corners in corners:
        box_corners = NuscenesDataset.pc_to_sensor(box_corners, cam_cs_record)
        this_box_corners = view_points(box_corners, np.array(cam_cs_record['camera_intrinsic']), normalize=True)[:2, :]
        
        visible = np.logical_and(this_box_corners[0, :] > 0, this_box_corners[0, :] < img_shape[0])
        visible = np.logical_and(visible, this_box_corners[1, :] < img_shape[1])
        visible = np.logical_and(visible, this_box_corners[1, :] > 0)
        visible = np.logical_and(visible, box_corners[2, :] > 1)
        in_front = box_corners[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.
        isVisible = any(visible) and all(in_front)
        mask.append(isVisible)
        if isVisible:
            cornerList.append(this_box_corners)

    return np.array(cornerList), mask

##------------------------------------------------------------------------------
def box_corners_to_2dBox(corners_2d, imsize, mode='xywh'):
    """
    Convert the 3d box to the 2D bounding box in COCO format (x,y,h,w)

    :param view
    :param imsize: Image size in pixels
    :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
    :return: <np.float: 2, 4>. Corners of the 2D box
    """
    bboxes = []
    for corner_2d in corners_2d:
        neighbor_map = {0: [1,3,4], 1: [0,2,5], 2: [1,3,6], 3: [0,2,7],
                        4: [0,5,7], 5: [1,4,6], 6: [2,5,7], 7: [3,4,6]}
        border_lines = [[(0,imsize[1]),(0,0)],
                        [(imsize[0],0),(imsize[0],imsize[1])],
                        [(imsize[0],imsize[1]),(0,imsize[1])],
                        [(0,0),(imsize[0],0)]]

        # Find corners that are outside image boundaries
        invisible = np.logical_or(corner_2d[0, :] < 0, corner_2d[0, :] > imsize[0])
        invisible = np.logical_or(invisible, corner_2d[1, :] > imsize[1])
        invisible = np.logical_or(invisible, corner_2d[1, :] < 0)
        ind_invisible = [i for i, x in enumerate(invisible) if x]
        corner_2d_visible = np.delete(corner_2d, ind_invisible, 1)

        # Find intersections with boundary lines
        for ind in ind_invisible:
            # intersections = []
            invis_point = (corner_2d[0, ind], corner_2d[1, ind])
            for i in neighbor_map[ind]:
                if i in ind_invisible:
                    # Both corners outside image boundaries, ignore them
                    continue

                nbr_point = (corner_2d[0,i], corner_2d[1,i])
                line = LineString([invis_point, nbr_point])
                for borderline in border_lines:
                    intsc = line.intersection(LineString(borderline))
                    if not intsc.is_empty:
                        corner_2d_visible = np.append(corner_2d_visible, np.asarray([[intsc.x],[intsc.y]]), 1)
                        break

        # Construct a 2D box covering the whole object
        x_min, y_min = np.amin(corner_2d_visible, 1)
        x_max, y_max = np.amax(corner_2d_visible, 1)

        # Get the box corners
        corner_2d = np.array([[x_max, x_max, x_min, x_min],
                            [y_max, y_min, y_min, y_max]])

        # Convert to the MS COCO bbox format
        # bbox = [corner_2d[0,3], corner_2d[1,3],
        #         corner_2d[0,0]-corner_2d[0,3],corner_2d[1,1]-corner_2d[1,0]]
        if mode == 'xyxy':
            bbox = [x_min, y_min, x_max, y_max]
        elif mode == 'xywh':
            bbox = [x_min, y_min, abs(x_max-x_min), abs(y_max-y_min)]
        else: 
            raise Exception("mode of '{}'' is not supported".format(mode))
        bboxes.append(bbox)

    return bboxes

##------------------------------------------------------------------------------
def nuscene_cat_to_coco(nusc_ann_name):

    ## Convert nuscene categories to COCO cat, cat_ids and supercats
    try:
        coco_equivalent = NS_C.COCO_CLASSES[nusc_ann_name]
    except KeyError:
        return None, None, None
    
    coco_cat = coco_equivalent['category']
    coco_id = coco_equivalent['id']
    coco_supercat = coco_equivalent['supercategory']

    return coco_cat, coco_id, coco_supercat

##------------------------------------------------------------------------------
def nuscenes_box_to_coco(box, view, imsize, wlh_factor: float = 1.0, mode='xywh'):
    """
    Convert the 3d box to the 2D bounding box in COCO format (x,y,w,h)

    :param view
    :param imsize: Image size in pixels
    :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
    :return: <np.float: 2, 4>. Corners of the 2D box
    """
    # box = copy.deepcopy(box)
    # corners = np.array([corner for corner in box.corners().T if corner[2] > 0]).T
    # if len(corners) == 0:
    #     return None
    corner_2d = view_points(box.corners(), view, normalize=True)[:2]
    # bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))
    # corner_2d = imcorners
 
    neighbor_map = {0: [1,3,4], 1: [0,2,5], 2: [1,3,6], 3: [0,2,7],
                    4: [0,5,7], 5: [1,4,6], 6: [2,5,7], 7: [3,4,6]}
    border_lines = [[(0,imsize[1]),(0,0)],
                    [(imsize[0],0),(imsize[0],imsize[1])],
                    [(imsize[0],imsize[1]),(0,imsize[1])],
                    [(0,0),(imsize[0],0)]]

    # Find corners that are outside image boundaries
    invisible = np.logical_or(corner_2d[0, :] < 0, corner_2d[0, :] > imsize[0])
    invisible = np.logical_or(invisible, corner_2d[1, :] > imsize[1])
    invisible = np.logical_or(invisible, corner_2d[1, :] < 0)
    ind_invisible = [i for i, x in enumerate(invisible) if x]
    corner_2d_visible = np.delete(corner_2d, ind_invisible, 1)

    # Find intersections with boundary lines
    for ind in ind_invisible:
        # intersections = []
        invis_point = (corner_2d[0, ind], corner_2d[1, ind])
        for i in neighbor_map[ind]:
            if i in ind_invisible:
                # Both corners outside image boundaries, ignore them
                continue

            nbr_point = (corner_2d[0,i], corner_2d[1,i])
            line = LineString([invis_point, nbr_point])
            for borderline in border_lines:
                intsc = line.intersection(LineString(borderline))
                if not intsc.is_empty:
                    corner_2d_visible = np.append(corner_2d_visible, np.asarray([[intsc.x],[intsc.y]]), 1)
                    break

    # Construct a 2D box covering the whole object
    x_min, y_min = np.amin(corner_2d_visible, 1)
    x_max, y_max = np.amax(corner_2d_visible, 1)

    # Get the box corners
    corner_2d = np.array([[x_max, x_max, x_min, x_min],
                        [y_max, y_min, y_min, y_max]])

    # Convert to the MS COCO bbox format
    # bbox = [corner_2d[0,3], corner_2d[1,3],
    #         corner_2d[0,0]-corner_2d[0,3],corner_2d[1,1]-corner_2d[1,0]]
    if mode == 'xyxy':
        bbox = [x_min, y_min, x_max, y_max]
    elif mode == 'xywh':
        bbox = [x_min, y_min, abs(x_max-x_min), abs(y_max-y_min)]
    else: 
        raise Exception("mode of '{}'' is not supported".format(mode))

    return bbox

##------------------------------------------------------------------------------