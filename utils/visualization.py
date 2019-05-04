
import os
import sys
import cv2
import json
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


## -----------------------------------------------------------------------------
def draw_xywh_bbox(img, bboxes, color=(0,255,0), lineWidth=3, format='BGR', 
                   names=None):

    assert format in ['RGB', 'BGR'], "Format must be either 'BGR' or 'RGB'."
    if names is not None:
        assert len(bboxes) == len(names), "Bboxes and names must have the same length"
        
    if format == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(bboxes):
        box = [int(elem) for elem in box]
        cv2.rectangle(img,(box[0],box[1]), (box[0]+box[2], box[1]+box[3]),
                      color,lineWidth)
        if names is not None:
            vis_class(img, box[:2], names[idx])

    if format == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

## -----------------------------------------------------------------------------
def draw_xyxy_bbox(img, bboxes, color=(0,255,0), lineWidth=3, format='BGR', 
                   names=None):

    assert format in ['RGB', 'BGR'], "Format must be either 'BGR' or 'RGB'."
    if names is not None:
        assert len(bboxes) == len(names), "Bboxes and names must have the same length"

    if format == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(bboxes):
        box = [int(elem) for elem in box]
        cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), color, lineWidth)
        if names is not None:
            img = vis_class(img, box[:2], names[idx])
    
    if format == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

## -----------------------------------------------------------------------------
def draw_points(img, points, color=(0,255,0), radius=3, thickness=-1, format='BGR'):

    assert format in ['RGB', 'BGR'], "Format must be either 'BGR' or 'RGB'."
    if format == 'RGB':
        # Change format to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for point in points:
        point = [int(elem) for elem in point]
        cv2.circle(img,(point[0], point[1]), radius, color, thickness)

    if format == 'RGB':
        # Change back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


## -----------------------------------------------------------------------------
def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img