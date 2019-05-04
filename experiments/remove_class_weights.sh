#! /bin/sh

# Remove the last layers' weights from a pre-trained model
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

INPUT_MODEL="$ROOT_DIR/output/models/Faster_RCNN_R_101_FPN_2x_original/model_final.pkl"
OUTPUT_MODEL="$ROOT_DIR/output/models/Faster_RCNN_R_101_FPN_2x_original_RW/model_final.pkl"

##------------------------------------------------------------------------------

echo "INFO: Removing last layer weights of $INPUT_MODEL"
cd $ROOT_DIR/tools
python remove_weight_blobs.py \
  --input_model $INPUT_MODEL \
  --output_model $OUTPUT_MODEL

echo "INFO: New model saved to: $OUTPUT_MODEL"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
