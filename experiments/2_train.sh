# Finetune Fast-RCNN (Pretrained on COCO2017) on the NuCOCO dataset.

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

export CUDA_VISIBLE_DEVICES=0

TRAIN_SPLIT='mini_train'
VAL_SPLIT='mini_val'
CFG="$ROOT_DIR/configs/fast_rcnn_X-101-32x8d-FPN_1x_finetune_nucoco.yaml"
TRAIN_WEIGHTS="$ROOT_DIR/data/models/X_101_32x8d_FPN_1x_original/model_final.pkl"
OUT_DIR="$ROOT_DIR/data/models/X_101_32x8d_FPN_1x_nucoco"

##------------------------------------------------------------------------------
TRAIN_PROP_FILES="('$ROOT_DIR/data/nucoco/proposals/proposals_$TRAIN_SPLIT.pkl',)"
TEST_PROP_FILES="('$ROOT_DIR/data/nucoco/proposals/proposals_$VAL_SPLIT.pkl',)"
TRAIN_DATASETS="('nucoco_$TRAIN_SPLIT',)"
TEST_DATASETS="('nucoco_$VAL_SPLIT',)"
RES_DIR="$OUT_DIR/results"

set -e
mkdir -p $OUT_DIR
mkdir -p $RES_DIR
cp $CFG $OUT_DIR

echo "INFO: Starting training..."
cd $ROOT_DIR/detectron
python tools/train_net.py \
--cfg $CFG \
OUTPUT_DIR $OUT_DIR \
TRAIN.DATASETS $TRAIN_DATASETS \
TRAIN.PROPOSAL_FILES $TRAIN_PROP_FILES \
TRAIN.WEIGHTS $TRAIN_WEIGHTS \
TEST.DATASETS $TEST_DATASETS \
TEST.PROPOSAL_FILES $TEST_PROP_FILES

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
