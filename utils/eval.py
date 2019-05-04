
import os
import sys
import json
import numpy as np
from scipy.io import savemat

## -----------------------------------------------------------------------------
def parse_train_log(log_file, mat_file=None):

    stats = {'accuracy_cls':[], 'iter':[], 'loss':[],'loss_bbox':[],
             'loss_cls':[], 'lr':[], 'time':[]}
    json_log = []

    with open(log_file) as f:
        text = f.readlines()
    # json_log = [line[11:] for line in text if line[:10] == 'json_stats']
    for i in range(len(text)):
        line=text[i]
        if line[:10] == 'json_stats':
            json_log.append(line[11:])
        # Get the calss precision Stats
        if '~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~' in line:
            raw_cls_APs = text[i+2:i+8]

        # Get the recall Stats
        if '~~~~ Summary metrics ~~~~' in line:
            raw_APs = text[i+1:i+7]      # Next 6 lines are APs
            raw_ARs = text[i+7:i+13]     # Next 6 lines are ARs

    ## Get training stats
    for item in json_log:
        data = json.loads(item)
        stats['accuracy_cls'].append(float(data['accuracy_cls']))
        stats['iter'].append(data['iter'])
        stats['loss'].append(float(data['loss']))
        stats['loss_bbox'].append(float(data['loss_bbox']))
        stats['loss_cls'].append(float(data['loss_cls']))
        stats['lr'].append(float(data['lr']))
        stats['time'].append(float(data['time']))

    ## Convert to numpy arrays
    for key, val in stats.items():
        if len(val) > 0:
            stats[key] = np.vstack(np.asarray(val))
        else:
            print("WARNING: No data found for '{}'".format(key))

    ## Get the Average Precision stats
    AP = text[-1].split()[-1]
    AP = AP.split(',')
    stats['AP'] = float(AP[0])
    stats['AP50'] = float(AP[1])
    stats['AP75'] = float(AP[2])
    stats['APs'] = float(AP[3])
    stats['APm'] = float(AP[4])
    stats['APl'] = float(AP[5])

    ## Get the Average Recall stats
    stats['ARmd1'] = float(raw_ARs[0].split('|')[-1].split('=')[-1])
    stats['ARmd10'] = float(raw_ARs[1].split('|')[-1].split('=')[-1])
    stats['ARmd100'] = float(raw_ARs[2].split('|')[-1].split('=')[-1])
    stats['ARs'] = float(raw_ARs[3].split('|')[-1].split('=')[-1])
    stats['ARm'] = float(raw_ARs[4].split('|')[-1].split('=')[-1])
    stats['ARl'] = float(raw_ARs[5].split('|')[-1].split('=')[-1])

    ## Get the Class Average Precision stats


    if mat_file is not None:
        savemat(mat_file, {'stats':stats})

    return stats


## -----------------------------------------------------------------------------
def parse_val_log(log_file, mat_file=None):

    stats = {}

    with open(log_file) as f:
        text = f.readlines()

    for i in range(len(text)):
        line=text[i]

        # Get the calss precision Stats
        if '~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~' in line:
            raw_cls_APs = text[i+2:i+8]

        # Get the recall Stats
        if '~~~~ Summary metrics ~~~~' in line:
            raw_APs = text[i+1:i+7]      # Next 6 lines are APs
            raw_ARs = text[i+7:i+13]     # Next 6 lines are ARs
            raw_APs.extend(text[i+13:i+19])     # Next 6 lines are APs with IoU 0.5 and 0.75

    ## Get the Average Precision stats
    stats['AP'] = float(raw_APs[0].split('|')[-1].split('=')[-1])
    stats['AP50'] = float(raw_APs[1].split('|')[-1].split('=')[-1])
    stats['AP75'] = float(raw_APs[2].split('|')[-1].split('=')[-1])
    stats['APs'] = float(raw_APs[3].split('|')[-1].split('=')[-1])
    stats['APm'] = float(raw_APs[4].split('|')[-1].split('=')[-1])
    stats['APl'] = float(raw_APs[5].split('|')[-1].split('=')[-1])
    stats['AP50s'] = float(raw_APs[6].split('|')[-1].split('=')[-1])
    stats['AP50m'] = float(raw_APs[7].split('|')[-1].split('=')[-1])
    stats['AP50l'] = float(raw_APs[8].split('|')[-1].split('=')[-1])
    stats['AP75s'] = float(raw_APs[9].split('|')[-1].split('=')[-1])
    stats['AP75m'] = float(raw_APs[10].split('|')[-1].split('=')[-1])
    stats['AP75l'] = float(raw_APs[11].split('|')[-1].split('=')[-1])

    ## Get the Average Recall stats
    stats['ARmd1'] = float(raw_ARs[0].split('|')[-1].split('=')[-1])
    stats['ARmd10'] = float(raw_ARs[1].split('|')[-1].split('=')[-1])
    stats['ARmd100'] = float(raw_ARs[2].split('|')[-1].split('=')[-1])
    stats['ARs'] = float(raw_ARs[3].split('|')[-1].split('=')[-1])
    stats['ARm'] = float(raw_ARs[4].split('|')[-1].split('=')[-1])
    stats['ARl'] = float(raw_ARs[5].split('|')[-1].split('=')[-1])

    ## Get the Class Average Precision stats
    stats['AP_cls0'] = float(raw_cls_APs[0].split(':')[-1])
    stats['AP_cls1'] = float(raw_cls_APs[1].split(':')[-1])
    stats['AP_cls2'] = float(raw_cls_APs[2].split(':')[-1])
    stats['AP_cls3'] = float(raw_cls_APs[3].split(':')[-1])
    stats['AP_cls4'] = float(raw_cls_APs[4].split(':')[-1])
    stats['AP_cls5'] = float(raw_cls_APs[5].split(':')[-1])

    if mat_file is not None:
        savemat(mat_file, {'stats':stats})

    return stats