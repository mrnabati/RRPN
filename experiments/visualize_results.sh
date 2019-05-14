#! /bin/sh

# Visualize training and validation results from training log file
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

MODEL="fast_rcnn_X-101-32x8d-FPN_1x_nucoco_rrpn"
TITLE="$MODEL"
LOG_TYPE="val"

##------------------------------------------------------------------------------
MODEL_DIR="$ROOT_DIR/data/models/$MODEL"
OUT_DIR="$MODEL_DIR/results"

set -e

if [ $LOG_TYPE = "train" ]; then
  LOG_FILE="$MODEL_DIR/train_log.txt"
  MAT_FILE="$MODEL_DIR/train_log.mat"
elif [ $LOG_TYPE = "val" ]; then
  LOG_FILE="$MODEL_DIR/val_results.txt"
  MAT_FILE="$MODEL_DIR/val_results.mat"
else
  echo "ERROR: LOG_TYPE not valid."
  exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: Model directory not found."
  exit 1
fi

## Convert the log file to .mat format
if [ ! -f $MAT_FILE ]; then
    echo "INFO: Converting log file to .mat format... "
    cd $ROOT_DIR/tools
    python parse_log.py --logfile $LOG_FILE --matfile $MAT_FILE --log_type $LOG_TYPE
fi

echo "INFO: Plotting results..."
cd $ROOT_DIR/tools
mkdir -p $OUT_DIR
COMMAND="vis_results('$MAT_FILE', '$OUT_DIR')"
matlab -nodesktop -nosplash -nodisplay -r "$COMMAND; exit"

echo "INFO: Plots saved to: $OUT_DIR"

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
