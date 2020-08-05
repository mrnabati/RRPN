
import os
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from PIL import Image
from copy import deepcopy
from detectron.utils.io import load_object

def rrpn_loader(rpn_pkl_file):
    """
    Open, index and return a pickled proposals dictionary
    """

    pkl = load_object(rpn_pkl_file)
    proposals = {}
    for boxes, scores, id in zip(pkl['boxes'], pkl['scores'], pkl['ids']):
        proposals[id] = {'boxes':boxes, 'scores':scores}

    return proposals


def get_im_proposals(point, sizes=(64, 128, 256, 512), aspect_ratios=(0.5, 1, 2),
                     layout=['center'], beta=8, include_depth=0):
    """
    Generate RRPN proposals for a single image
    param: centered (str): 'center', 'bottom'
    """
    anchors = _generate_anchors(point,
                                np.array(sizes, dtype=np.float),
                                np.array(aspect_ratios, dtype=np.float),
                                layout, 
                                beta,
                                include_depth=include_depth)

    anchors = _filter_anchors(anchors)

    return anchors


def _generate_anchors(point, sizes, aspect_ratios, layout, beta, include_depth):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.

    param include_depth: 1 or 0
    """

    distance = point[2]
    base_size = sizes[0]
    scales = sizes[1:] / base_size
    # beta = 8
    scales = (beta/distance)*scales

    center = (point[0], point[1])
    anchor = np.array([center[0] - base_size/2.0, center[1] - base_size/2.0,
                       center[0] + base_size/2.0, center[1] + base_size/2.0],
                      dtype=np.float)

    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )

    all_anchors = np.empty((0,4))
    for l in layout:
        new_anchors = _shift_anchors(anchors, l)
        all_anchors = np.vstack((all_anchors, new_anchors))

    if int(include_depth)==1:
        # Add the distance as the 5th element to all anchors
        new_shape = (all_anchors.shape[0], all_anchors.shape[1]+1)
        new_anchors = np.ones(new_shape) * distance
        new_anchors[:,:-1] = all_anchors
        all_anchors = new_anchors

    return all_anchors


def _filter_anchors(anchors):
    """Filter anchors based on their size
    """
    #TODO: Implement this function
    return anchors


def _shift_anchors(anchors, direction):
    """Shift anchors to the specified direction
    """
    new_anchors = deepcopy(anchors)
    if direction == 'center':
        pass

    elif direction == 'top':
        heights = new_anchors[:,3] - new_anchors[:,1] + 1
        heights = heights[:,np.newaxis]
        new_anchors[:,[1,3]] = new_anchors[:,[1,3]] - heights/2

    elif direction == 'bottom':
        heights = new_anchors[:,3] - new_anchors[:,1] + 1
        heights = heights[:,np.newaxis]
        new_anchors[:,[1,3]] = new_anchors[:,[1,3]] + heights/2

    elif direction == 'right':
        widths = new_anchors[:,2] - new_anchors[:,0] + 1
        widths = widths[:,np.newaxis]
        new_anchors[:,[0,2]] = new_anchors[:,[0,2]] + widths/2

    elif direction == 'left':
        widths = new_anchors[:,2] - new_anchors[:,0] + 1
        widths = widths[:,np.newaxis]
        new_anchors[:,[0,2]] = new_anchors[:,[0,2]] - widths/2

    return new_anchors



def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""

    w, h, x_ref, y_ref = _whctrs(anchor)

    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ref, y_ref)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""


    w, h, x_ref, y_ref = _whctrs(anchor)

    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ref, y_ref)
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _whbtms(anchor):
    """Return width, height, x bottom, and y bottom for an anchor (window)."""

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_btm = anchor[0] + 0.5 * (w - 1)
    y_btm = anchor[1] + 1.0 * (h - 1)
    return w, h, x_btm, y_btm


def _mkanchors(ws, hs, x_ref, y_ref):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]

    anchors = np.hstack(
        (
            x_ref - 0.5 * (ws - 1),
            y_ref - 0.5 * (hs - 1),
            x_ref + 0.5 * (ws - 1),
            y_ref + 0.5 * (hs - 1)
        )
    )
    return anchors
