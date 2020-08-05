# Convert the Nuscenes dataset to COCO format

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
## Modify these parameters as needed

NUSC_SPLIT='mini_val'
NUM_RADAR_SWEEPS=1       # number of Radar sweeps
USE_SYMLINKS='True'      # use symlinks instead of copying nuScenes images

##------------------------------------------------------------------------------
NUSC_DIR="$ROOT_DIR/data/nuscenes"
OUT_DIR="$ROOT_DIR/data/nucoco"
# create symbolic link to the nucoco dataset for Detectron
ln -s $ROOT_DIR/data/nucoco $ROOT_DIR/detectron/detectron/datasets/data/nucoco

echo "INFO: Converting nuScenes to COCO format..."

cd $ROOT_DIR/tools
python nuscenes_to_coco.py \
  --nusc_root $NUSC_DIR \
  --split $NUSC_SPLIT \
  --out_dir $OUT_DIR \
  --nsweeps_radar $NUM_RADAR_SWEEPS \
  --use_symlinks $USE_SYMLINKS \


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
