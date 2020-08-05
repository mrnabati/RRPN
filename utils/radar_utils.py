import matplotlib
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud
from nuscenes_utils.geometry_utils import view_points


def draw_poly(corners, color, ax, lineWidth=1):
    """
    Draw a polygon given its corners are ordered continuously, e.g.:
    0 *--------------* 1
       \            /
        \          /
       3 *--------* 2
    """

    # plt.figure(figureID)
    corners = list(zip(corners[0,:], corners[1,:]))
    prev = corners[-1]
    for corner in corners:
        ax.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color,
                linewidth=lineWidth)
        prev = corner
# ------------------------------------------------------------------------------


def pointcloud_to_bbox(points, coloring, im, scale=False):
    """
    Convert points in a pointcloud to rectangular bounding boxes. The points
    need to be mapped to the image perspective first.

    :param points:
    :param coloring:
    :param im:
    :param scale:
    :return allCorners:
    """

    ## TODO: Change corners foramt to compy with the original devkit format
    allCorners = []

    for i, (x, y) in enumerate(zip(points[0], points[1])):
        # Get the bbox size based on its distance
        width, height = get_bbox_size(coloring[i])

        # find box corners
        corners = [(x-width/2, y-height), (x+width/2, y-height),
                   (x+width/2, y), (x-width/2, y)]
        allCorners.append(corners)

    return allCorners
# ------------------------------------------------------------------------------


def get_bbox_size(distance, category=None):
    """
    Calculate the bounding box size for Radar detections based on their distance
    from the vehicle

    :param distance: distance (m) from the vehicle
    :param category (str): object category
    :return bbox_size: BBox size fromatted as [width, height]
    """

    # Power2 fitted model for the bounding box size w.r.t distance
    ## W = a * x^b + c
    # a = 1428
    a = 1428
    b = -0.8087
    c = -11.79
    scale = [1, 1]
    bb_size = np.add(np.multiply(a, np.power(distance, b)), c) * 1.1

    if category is None:
        scale = [1, 0.8]
    else:
        # Write bbox aspect rations for different object classes here
        pass

    bbox_size = [bb_size * scale[0], bb_size * scale[1]]
    return bbox_size
# ------------------------------------------------------------------------------


def filter_pointcloud(points, threshold, distance=None):
    """
    Filter the pointcloud based on points' proximity and distance to the vehicle

    :param points <np.array>:
    :param threshold:
    :param distance: A list of points' distances to the vehicle
    :return sparse_points <np.array>:
    """

    sparse_points, good_indices = sparse_subset(points, threshold)

    if distance is not None:
        distance = [distance[i] for i in good_indices]

    return sparse_points, distance
# ------------------------------------------------------------------------------


def sparse_subset(points, r):
    """
    Return a maximal list of elements of points such that no pairs of points in
    the result have distance less than r. The distance function is defined
    outside this funcion.

    :param points <np.array>: points [[x0 x1 ...] [y0 y1 ...] [z0 z1 ...]]
    :param r:
    :return result, good_indices:
    """
    result = []
    good_indices = []
    stacked_points = np.squeeze(np.dstack((points[0], points[1], points[2])))

    # Check the distance between every point and all the others
    for ind, p in enumerate(stacked_points):
        if all(dist(p, q) >= r for q in result):
            result.append(p)
            good_indices.append(ind)

    # Change the points' format back to the original
    result = np.array(result)
    result = np.array([result[:, 0], result[:, 1], result[:, 2]])

    return result, good_indices
# ------------------------------------------------------------------------------


def dist(p, q):
    """
    Distance between two points.
    """

    # 2D distance
    # return math.hypot(p[0] - q[0], p[1] - q[1])

    # 3D distance
    return np.linalg.norm(p-q)
# ------------------------------------------------------------------------------


def box_3d_to_2d(boxes, intrinsic):
    """
    Get x and y of the 8 corners of a 3D box after mapping it to the image
    viewpoint
    """

    corners_list = []
    for box in boxes:
        corners_3d = box.corners()
        corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]
        corners_list.append(corners_img)

    return corners_list
