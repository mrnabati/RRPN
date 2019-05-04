import _init_paths
import numpy as np
import argparse
import sys
import os

from detectron.utils.io import save_object
from eval import parse_train_log, parse_val_log


## Generate proposals for the NuCOCO dataset using the Selective Search method.

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='{Parse the detectron training log and save as .mat file.}')
    parser.add_argument('--logfile', dest='logfile',
                        default='../output/nucoco/models/train_log.txt',
                        help='Trainig log file to be parsed')

    parser.add_argument('--matfile', dest='matfile',
                        help='mat file to be created',
                        default='../output/nucoco/models/train_stats.mat')
    
    parser.add_argument('--log_type', dest='log_type',
                        help='log file type: train or val',
                        default='val')

    args = parser.parse_args()
    args.logfile = os.path.abspath(args.logfile)
    args.matfile = os.path.abspath(args.matfile)

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.log_type == 'train':
        stats = parse_train_log(args.logfile, args.matfile)
    else:
        stats = parse_val_log(args.logfile, args.matfile)
