# Generate RRPN proposals from the nucoco dataset

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

SPLIT='mini_val'

##------------------------------------------------------------------------------
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_${SPLIT}.json"
IMGS_DIR="$ROOT_DIR/data/nucoco/${SPLIT}"
OUT_FILE="$ROOT_DIR/data/nucoco/proposals/proposals_${SPLIT}.pkl"

echo "INFO: Creating proposals..."

cd $ROOT_DIR/tools
python generate_rrpn_proposals.py \
  --ann_file $ANN_FILE \
  --imgs_dir $IMGS_DIR \
  --out_file $OUT_FILE \


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
