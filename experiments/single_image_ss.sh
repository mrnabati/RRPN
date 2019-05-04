#! /bin/sh

# Generate Selective Search proposals for a single image
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

IMAGE="$ROOT_DIR/output/image.png"   # Image to generate proposals for
FAST_MODE=1
IM_WIDTH=800

##------------------------------------------------------------------------------
MAT_FILE="$ROOT_DIR/output/tmp_proposals_ss.mat"
PKL_FILE="$ROOT_DIR/output/tmp_proposals_ss.pkl"
TMP_LIST="$ROOT_DIR/output/tmp_list.txt"
echo "$IMAGE" > $TMP_LIST

echo "INFO: Generating Selective Search proposals for validation images..."
cd $ROOT_DIR/lib/selective_search
COMMAND="selective_search_rcnn('$TMP_LIST', '$MAT_FILE', $FAST_MODE, $IM_WIDTH)"
matlab -nodesktop -nosplash -nodisplay -nojvm -r "$COMMAND; exit"

echo "INFO: Converting validation .mat proposals to .pkl format..."
cd $ROOT_DIR/tools
python convert_ss.py nucoco_val $MAT_FILE $PKL_FILE

echo "INFO: Visualizing proposals..."
cd $ROOT_DIR/tests
python vis_ss_proposal.py --image_file $IMAGE --proposals_file $PKL_FILE

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
