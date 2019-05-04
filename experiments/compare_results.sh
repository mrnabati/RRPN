#! /bin/sh

# Generate Selective Search proposals for a COCO style dataset
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

## Can compare as many models as you want if you add them to the COMMAND below
MODEL1="X_101_32x8d_FPN_1x_ft30000_nucoco_sw_fb_rrpn_v4"
MODEL2="X_101_32x8d_FPN_1x_ft30000_nucoco_sw_fb_rrpn_v6"
OUT_DIR="$ROOT_DIR/output/results"

##------------------------------------------------------------------------------
MODEL1_DIR="$ROOT_DIR/output/models/$MODEL1"
MODEL2_DIR="$ROOT_DIR/output/models/$MODEL2"
MAT_FILE1="$MODEL1_DIR/val_results.mat"
MAT_FILE2="$MODEL2_DIR/val_results.mat"
LOG_FILE1="$MODEL1_DIR/val_results.txt"
LOG_FILE2="$MODEL2_DIR/val_results.txt"
COMMAND="comp_results('$OUT_DIR', '$MAT_FILE1', '$MODEL1', '$MAT_FILE2', '$MODEL2')"

if [ ! -d "$MODEL1_DIR" ]; then
  echo "ERROR: Model 1 directory not found."
  exit 1
fi
if [ ! -d "$MODEL2_DIR" ]; then
  echo "ERROR: Model 2 directory not found."
  exit 1
fi

## Check if the .mat files exist
if [ ! -f $MAT_FILE1 ]; then
    echo "INFO: Converting model 2 log file to .mat format... "
    cd $ROOT_DIR/tools
    python parse_log.py --logfile $LOG_FILE1 --matfile $MAT_FILE1
fi

if [ ! -f $MAT_FILE2 ]; then
    echo "INFO: Converting model 1 log file to .mat format "
    cd $ROOT_DIR/tools
    python parse_log.py --logfile $LOG_FILE2 --matfile $MAT_FILE2
fi

echo "INFO: Plotting results..."
mkdir -p $OUT_DIR
cd $ROOT_DIR/tools
matlab -nodesktop -nosplash -nodisplay -r "$COMMAND; exit"
# matlab -nodesktop -nosplash -nodisplay -r "$COMMAND;exit"

echo "INFO: Plots saved to: $OUT_DIR"
echo "-------------------------------------------------------------------------"
