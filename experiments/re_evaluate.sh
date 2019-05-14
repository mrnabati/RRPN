#! /bin/sh

# Finetune Fast-RCNN (Pretrained on COCO2017) on the NuCOCO dataset.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Set the proposal files in the config file
MODEL="fast_rcnn_X-101-32x8d-FPN_1x_nucoco_rrpn"
DATASET='nucoco'

##------------------------------------------------------------------------------
MODEL_DIR="$ROOT_DIR/data/models/$MODEL"
OUTPUT_DIR="$MODEL_DIR/test/nucoco_val/generalized_rcnn"
LOG_FILE="$MODEL_DIR/val_results.txt"
THIS_SCRIPT=`basename "$0"`

set -e

echo "INFO $THIS_SCRIPT: Starting evaluation..."
cd $ROOT_DIR/detectron
python tools/reval.py \
$OUTPUT_DIR \
--dataset nucoco_val \
| tee $LOG_FILE
