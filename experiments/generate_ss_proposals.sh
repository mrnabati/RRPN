#! /bin/sh

# Generate Selective Search proposals for a COCO style dataset
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

DATASET='nucoco_sw_fb'   # Dataset to generate proposals for
FAST_MODE=1
IM_WIDTH=800

##------------------------------------------------------------------------------
PROP_OUT_DIR="$ROOT_DIR/output/proposals/$DATASET/ss"
DATASET_DIR="$ROOT_DIR/output/datasets/$DATASET"

TR_MAT_FILE="$PROP_OUT_DIR/proposals_nucoco_train.mat"
TR_PKL_FILE="$PROP_OUT_DIR/proposals_nucoco_train.pkl"
TR_IMGS_DIR="$DATASET_DIR/train"
TR_ANN_FILE="$DATASET_DIR/annotations/instances_train.json"
TR_LIST="$PROP_OUT_DIR/tmp_train_img_list.txt"
VA_MAT_FILE="$PROP_OUT_DIR/proposals_nucoco_val.mat"
VA_PKL_FILE="$PROP_OUT_DIR/proposals_nucoco_val.pkl"
VA_IMGS_DIR="$DATASET_DIR/val"
VA_ANN_FILE="$DATASET_DIR/annotations/instances_val.json"
VA_LIST="$PROP_OUT_DIR/tmp_val_img_list.txt"

ln -sfn $ROOT_DIR/output/datasets/$DATASET $ROOT_DIR/output/nucoco
mkdir -p $PROP_OUT_DIR
set -e

if [ ! -d "$TR_IMGS_DIR" ]; then
  echo "ERROR: Train image directory not found."
  exit 1
fi
if [ ! -d "$VA_IMGS_DIR" ]; then
  echo "ERROR: Val image directory not found."
  exit 1
fi

echo "INFO: Generating list of train and val images..."
cd $ROOT_DIR/utils
python -c "from datasets import get_coco_img_list as gl; gl('$TR_IMGS_DIR', '$TR_ANN_FILE', '$TR_LIST')"
python -c "from datasets import get_coco_img_list as gl; gl('$VA_IMGS_DIR', '$VA_ANN_FILE', '$VA_LIST')"

echo "INFO: Generating Selective Search proposals for val images..."
cd $ROOT_DIR/lib/selective_search
COMMAND="selective_search_rcnn('$VA_LIST', '$VA_MAT_FILE', $FAST_MODE, $IM_WIDTH)"
matlab -nodesktop -nosplash -nodisplay -nojvm -r "$COMMAND; exit"

echo "INFO: Converting val .mat proposals to .pkl format..."
cd $ROOT_DIR/tools
python convert_ss.py nucoco_val $VA_MAT_FILE $VA_PKL_FILE


echo "INFO: Generating Selective Search proposals for train images..."
cd $ROOT_DIR/lib/selective_search
COMMAND="selective_search_rcnn('$TR_LIST', '$TR_MAT_FILE', $FAST_MODE, $IM_WIDTH)"
matlab -nodesktop -nosplash -nodisplay -nojvm -r "$COMMAND; exit"

echo "INFO: Converting training .mat proposals to .pkl format..."
cd $ROOT_DIR/tools
python convert_ss.py nucoco_train $TR_MAT_FILE $TR_PKL_FILE

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
