#! /bin/sh

# Generate object proposals using the Radar Region Proposal Network (RRPN)
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

DATASET='nucoco_s_fb'   # Dataset to generate proposals for
INCLUDE_DEPTH=0
PROP_OUT_DIR="$ROOT_DIR/data/proposals/$DATASET/rrpn_v6"

##------------------------------------------------------------------------------
TR_IMGS_DIR="$ROOT_DIR/data/datasets/$DATASET/train"
TR_ANNS_FILE="$ROOT_DIR/data/datasets/$DATASET/annotations/instances_train.json"
TR_PROP_FILE="$PROP_OUT_DIR/proposals_nucoco_train.pkl"
VA_IMGS_DIR="$ROOT_DIR/data/datasets/$DATASET/val"
VA_ANNS_FILE="$ROOT_DIR/data/datasets/$DATASET/annotations/instances_val.json"
VA_PROP_FILE="$PROP_OUT_DIR/proposals_nucoco_val.pkl"
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

echo "INFO: Generating proposals for NuCOCO train images..."
cd $ROOT_DIR/tools
python gen_proposal_rrpn.py \
  --ann_file $TR_ANNS_FILE \
  --imgs_dir $TR_IMGS_DIR \
  --output_file $TR_PROP_FILE \
  --include_depth $INCLUDE_DEPTH

echo "INFO: Generating proposals for NuCOCO validation images..."
python gen_proposal_rrpn.py \
  --ann_file $VA_ANNS_FILE \
  --imgs_dir $VA_IMGS_DIR \
  --output_file $VA_PROP_FILE \
  --include_depth $INCLUDE_DEPTH

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
