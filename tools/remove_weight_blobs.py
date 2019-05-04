
import _init_paths
import os
import pickle as pkl
import argparse
from detectron.utils.io import load_object, save_object

## Remove the final layer weights in a trained model so that the difference
## in number of classes does not raise error

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    parser.add_argument('--input_model', dest='input_model',
                        help='Trained model weights',
                        default='../output/models/R_50_C4_2x_original/model_final.pkl')
    parser.add_argument('--output_model', dest='output_model',
                        help='Ouput model weights',
                        default='../output/models/R_50_C4_2x_original_RW/model_final.pkl')

    args = parser.parse_args()
    args.input_model = os.path.abspath(args.input_model)
    args.output_model = os.path.abspath(args.output_model)

    return args



if __name__ == '__main__':
    args = parse_args()
    out_dir = os.path.dirname(args.output_model)
    os.makedirs(out_dir, exist_ok=True)

    wts = load_object(args.input_model)

    for blob in list(wts['blobs'].keys()):
        if blob.startswith('cls_score_') or blob.startswith('bbox_pred_'):
            del wts['blobs'][blob]

    save_object(wts, args.output_model)
    #with open(args.output_model, 'wb') as f:
    #    pkl.dump(wts, f)
