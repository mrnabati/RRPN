# Run inference on a single image from the nucoco

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

export CUDA_VISIBLE_DEVICES=0
MODEL_PKL="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb/model_final.pkl"
MODEL_CFG="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb/fast_rcnn_R-101-FPN_2x_nucoco.yaml"
OUT_DIR="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb/inference"
VAL_SPLIT="mini_train"
IMG_IND=0

##------------------------------------------------------------------------------
PROPOSAL_PKL="$ROOT_DIR/data/nucoco/proposals/proposals_$VAL_SPLIT.pkl"
IMGS_DIR="$ROOT_DIR/data/nucoco/$VAL_SPLIT"
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_$VAL_SPLIT.json"

echo "INFO: Running inference... "
set -e
cd $ROOT_DIR/tools
python inference.py \
    --im_ind $IMG_IND \
    --imgs_dir $IMGS_DIR \
    --ann_file $ANN_FILE \
    --rpn-pkl $PROPOSAL_PKL \
    --output-dir $OUT_DIR \
    $MODEL_PKL \
    $MODEL_CFG \

echo "INFO: Results saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"

