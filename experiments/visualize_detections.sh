#! /bin/sh

# Visualize detections from a detections.pkl file for val/test dataset
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"


MODEL_DIR="$ROOT_DIR/output/models/X_101_32x8d_FPN_1x_ft30000_nucoco_sw_fb_ss"
DATASET='nucoco_sw_fb'       ## Must be the same dataset model is trained on
THRESH=0.7
FIRST=0     # Only visualize the first K images

##------------------------------------------------------------------------------
set -e
DETECTIONS_PKL="$MODEL_DIR/test/nucoco_val/generalized_rcnn/detections.pkl"
OUT_DIR="$MODEL_DIR/results/"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found."
    exit 1
fi

## Set symlink to the dataset
ln -sf $ROOT_DIR/output/datasets/$DATASET $ROOT_DIR/output/nucoco

echo "INFO: Visualizing results..."
mkdir -p $OUT_DIR
cd $ROOT_DIR/detectron
python tools/visualize_results.py \
    --dataset nucoco_val \
    --detections $DETECTIONS_PKL \
    --thresh $THRESH \
    --output-dir $OUT_DIR \
    --first $FIRST

echo "INFO: Detection images saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
