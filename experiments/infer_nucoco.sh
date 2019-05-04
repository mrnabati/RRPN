#! /bin/sh

# Run inference on a single image.
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Parameters
MODEL_PKL="$ROOT_DIR/data/models/R_50_C4_2x_original/model_final.pkl"
MODEL_CFG="$ROOT_DIR/experiments/cfgs/fast_rcnn_R-50-C4_2x_infer_nucoco.yaml"
PROPOSAL_PKL="$ROOT_DIR/data/proposals/proposals_nucoco_val.pkl"
OUT_DIR="$ROOT_DIR/data/inference_results"
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_val.json"
IMGS_DIR="$ROOT_DIR/data/nucoco/val"

##------------------------------------------------------------------------------
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

