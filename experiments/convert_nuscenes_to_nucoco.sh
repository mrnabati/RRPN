#! /bin/sh

# Convert the Nuscenes dataset to COCO format
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

DATASET='nucoco_s_fb'   # New dataset's name
TRAIN_RATIO=0.85        # Training data percentage
INCLUDE_SWEEPS='False'   # If 'True', include non key-frame samples

##------------------------------------------------------------------------------
## Do not change anything below this line!
NUSCENES_DIR="$ROOT_DIR/data/datasets/nuscenes"
TR_IMGS_DIR="$ROOT_DIR/data/datasets/$DATASET/train"
VA_IMGS_DIR="$ROOT_DIR/data/datasets/$DATASET/val"
TR_ANNS_DIR="$ROOT_DIR/data/datasets/$DATASET/annotations/instances_train.json"
VA_ANNS_DIR="$ROOT_DIR/data/datasets/$DATASET/annotations/instances_val.json"

echo "INFO: Converting from NuScenes to COCO format..."

cd $ROOT_DIR/tools
python nuscenes_to_coco.py \
  --nuscene_root $NUSCENES_DIR \
  --include_sweeps $INCLUDE_SWEEPS \
  --train_ann_file $TR_ANNS_DIR \
  --val_ann_file $VA_ANNS_DIR \
  --train_imgs_dir $TR_IMGS_DIR \
  --val_imgs_dir $VA_IMGS_DIR \
  --train_ratio $TRAIN_RATIO

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
