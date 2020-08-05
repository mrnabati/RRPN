import os.path
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

## Add lib to PYTHONPATH
lib_path = os.path.abspath(os.path.join(this_dir, '..', 'lib'))
add_path(lib_path)

## Add utils to PYTHONPATH
utils_path = os.path.abspath(os.path.join(this_dir, '..', 'utils'))
add_path(utils_path)

## Add lib to PYTHONPATH
detectron_path = os.path.abspath(os.path.join(this_dir, '..', 'detectron'))
add_path(detectron_path)