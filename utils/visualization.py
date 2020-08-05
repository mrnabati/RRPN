
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
def save_fig(filepath, fig=None):
    '''
    Save the current image with no whitespace in pdf format
    '''
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0,0,1,1,0,0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches = 0, bbox_inches='tight', format='pdf')