# Test trained model

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

export CUDA_VISIBLE_DEVICES=0
MODEL_PKL="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb/model_final.pkl"
MODEL_CFG="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb/fast_rcnn_R-101-FPN_2x_nucoco.yaml"
OUT_DIR="$ROOT_DIR/data/models/R_101_FPN_2x_nucoco_it30000_fb"
DATASET="mini_val"

##------------------------------------------------------------------------------
TEST_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$DATASET.pkl"
IMGS_DIR="$ROOT_DIR/data/nucoco/$DATASET"
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_$DATASET.json"

echo "INFO: Running inference... "
cd $ROOT_DIR/detectron
python tools/test_net.py \
    --cfg $MODEL_CFG \
    --vis \
    OUTPUT_DIR $OUT_DIR \
    TEST.DATASETS "('nucoco_$DATASET',)" \
    TEST.WEIGHTS $MODEL_PKL \
    TEST.PROPOSAL_FILES "('$TEST_PROP_FILES',)" \
    VIS True \

echo "INFO: Results saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"

