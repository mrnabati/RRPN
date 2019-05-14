#! /bin/sh

# Run inference on a single image.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Parameters
PROPOSAL_METHOD='rrpn'
MODEL='fast_rcnn_X-101-32x8d-FPN_1x_nucoco_rrpn'
CFG='fast_rcnn_X-101-32x8d-FPN_1x_finetune_nucoco.yaml'

##------------------------------------------------------------------------------
DATASET='nucoco'
MODEL_PKL="$ROOT_DIR/data/models/$MODEL/train/nucoco_train/generalized_rcnn/model_final.pkl"
MODEL_CFG="$ROOT_DIR/experiments/cfgs/$CFG"
PROPOSAL_PKL="$ROOT_DIR/data/proposals/$DATASET/$PROPOSAL_METHOD/proposals_nucoco_val.pkl"
OUT_DIR="$ROOT_DIR/data/models/$MODEL/inference_results"
ANN_FILE="$ROOT_DIR/data/datasets/$DATASET/annotations/instances_val.json"
IMGS_DIR="$ROOT_DIR/data/datasets/$DATASET/val"

echo "INFO: Running inference... "
cd $ROOT_DIR/tools
python infer_nucoco.py \
    --rpn-pkl $PROPOSAL_PKL \
    --output-dir $OUT_DIR \
    --ann_file $ANN_FILE \
    --imgs_dir $IMGS_DIR \
    $MODEL_PKL \
    $MODEL_CFG

echo "INFO: Results saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"

