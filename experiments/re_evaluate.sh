#! /bin/sh

# Finetune Fast-RCNN (Pretrained on COCO2017) on the NuCOCO dataset.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Set the proposal files in the config file
MODEL="R_101_FPN_2x_ft30000_nucoco_sw_fb_rrpn_v6"
DATASET='nucoco_sw_fb'

##------------------------------------------------------------------------------
MODEL_DIR="$ROOT_DIR/output/models/$MODEL"
OUTPUT_DIR="$MODEL_DIR/test/nucoco_val/generalized_rcnn"
LOG_FILE="$MODEL_DIR/val_results.txt"
THIS_SCRIPT=`basename "$0"`

set -e
ln -sfn $ROOT_DIR/output/datasets/$DATASET $ROOT_DIR/output/nucoco

echo "INFO $THIS_SCRIPT: Starting evaluation..."
cd $ROOT_DIR/detectron
python tools/reval.py \
$OUTPUT_DIR \
--dataset nucoco_val \
| tee $LOG_FILE
