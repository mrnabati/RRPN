import tempfile
import subprocess
import shlex
import os
import numpy as np
import scipy.io

script_dirname = os.path.abspath(os.path.dirname(__file__))


def get_windows(image_fnames, cmd='selective_search',
                output_filename='./ss_proposals.mat'):
    """
    Run MATLAB Selective Search code on the given image filenames to
    generate window proposals.

    Parameters
    ----------
    image_filenames: strings
        Paths to file containing image names to run on.
    cmd: string
        selective search function to call:
            - 'selective_search' for a few quick proposals
            - 'selective_seach_rcnn' for R-CNN configuration for more coverage.
    """
    input('starting get_windows...')
    image_fnames = "'{}'".format(image_fnames)
    mat_filename = "'{}'".format(output_filename)

    command = "{}({}, {})".format(cmd, image_fnames, mat_filename)
    # print(command)

    # Execute command in MATLAB.
    mc = "matlab -nodesktop -nosplah -nodisplay -nojvm -r \"{}; exit\"".format(command)
    print(mc)
    input('Starting MATLAB command...')
    pid = subprocess.Popen(shlex.split(mc), stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, cwd=script_dirname)
    # pid.stdout.decode('utf-8')
    #output, err = pid.communicate()
    retcode = pid.wait()
    #print(output)
    #print(err)
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")
    input('Matlab code returned...')
    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    # Remove temporary file, and return.
    # os.remove(output_filename)
    # if len(all_boxes) != len(image_fnames):
    #     raise Exception("Something went wrong computing the windows!")
    print(all_boxes)
    input('something')
    return all_boxes

# if __name__ == '__main__':
#     """
#     Run a demo.
#     """
#     import time
#
#     image_filenames = [
#         script_dirname + '/000015.jpg',
#         script_dirname + '/cat.jpg'
#     ] * 4
#     t = time.time()
#     boxes = get_windows(image_filenames)
#     print(boxes[:2])
#     print("Processed {} images in {:.3f} s".format(
#         len(image_filenames), time.time() - t))
