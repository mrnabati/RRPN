#! /bin/sh

# Generate Edge Boxed proposals for a COCO style dataset
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

DATASET='nucoco_sw_fb'   # Dataset to generate proposals for

##------------------------------------------------------------------------------
DATASET_DIR="$ROOT_DIR/output/datasets/$DATASET"
PROP_OUT_DIR="$ROOT_DIR/output/proposals/$DATASET/eb"

TR_MAT_FILE="$PROP_OUT_DIR/proposals_nucoco_train.mat"
TR_PKL_FILE="$PROP_OUT_DIR/proposals_nucoco_train.pkl"
TR_IMGS_DIR="$DATASET_DIR/train"
TR_ANN_FILE="$DATASET_DIR/annotations/instances_train.json"
TR_LIST="$DATASET_DIR/train_img_list.txt"
VA_MAT_FILE="$PROP_OUT_DIR/proposals_nucoco_val.mat"
VA_PKL_FILE="$PROP_OUT_DIR/proposals_nucoco_val.pkl"
VA_IMGS_DIR="$DATASET_DIR/val"
VA_ANN_FILE="$DATASET_DIR/annotations/instances_val.json"
VA_LIST="$DATASET_DIR/val_img_list.txt"

set -e
mkdir -p $PROP_OUT_DIR
ln -sfn $ROOT_DIR/output/datasets/$DATASET $ROOT_DIR/output/nucoco
THIS_SCRIPT=`basename "$0"`

if [ ! -d "$TR_IMGS_DIR" ] || [ ! -d "$VA_IMGS_DIR" ]; then
  echo "ERROR $THIS_SCRIPT: images directory not found."
  exit 1
fi

if [ ! -f $TR_LIST ]; then
    echo "INFO $THIS_SCRIPT: Generating list of training images..."
    cd $ROOT_DIR/utils
    python -c "from datasets import get_coco_img_list as gl; gl('$TR_IMGS_DIR', '$TR_ANN_FILE', '$TR_LIST')"
fi

if [ ! -f $VA_LIST ]; then
    echo "INFO $THIS_SCRIPT: Generating list of validation images..."
    cd $ROOT_DIR/utils
    python -c "from datasets import get_coco_img_list as gl; gl('$VA_IMGS_DIR', '$VA_ANN_FILE', '$VA_LIST')"
fi

echo "INFO $THIS_SCRIPT: Generating Edge Boxes proposals for val images..."
cd $ROOT_DIR/lib/edge_boxes
COMMAND="edge_boxes_rcnn('$VA_LIST', '$VA_MAT_FILE')"
matlab -nodesktop -nosplash -nodisplay -nojvm -r "$COMMAND; exit"

echo "INFO $THIS_SCRIPT: Converting val .mat proposals to .pkl format..."
cd $ROOT_DIR/tools
python convert_ss.py nucoco_val $VA_MAT_FILE $VA_PKL_FILE


echo "INFO $THIS_SCRIPT: Generating Edge Boxes proposals for train images..."
cd $ROOT_DIR/lib/edge_boxes
COMMAND="edge_boxes_rcnn('$TR_LIST', '$TR_MAT_FILE')"
matlab -nodesktop -nosplash -nodisplay -nojvm -r "$COMMAND; exit"

echo "INFO $THIS_SCRIPT: Converting training .mat proposals to .pkl format..."
cd $ROOT_DIR/tools
python convert_ss.py nucoco_train $TR_MAT_FILE $TR_PKL_FILE

echo "INFO $THIS_SCRIPT: Done!"
echo "-------------------------------------------------------------------------"
