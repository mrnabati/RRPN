## **[DEPRECATED] This version of RRPN was developed for the NuScenes Teaser dataset (v0.1) and is deprecated. A new version of the code compatible with the full Nuscenes dataset (v1.0) will be available here soon. Stay tuned!**
---

# RRPN: Radar Region Proposal Network for Sensor Fusion in Autonomous Vehicles

## Introduction

RRPN is a Region Proposal Network (RPN) exploiting Radar detections to propose
Regions of Interest (RoI) for object detection in autonomous vehicles. RRPN provides
real-time RoIs for any two-stage object detection network while achieving precision
and recall values higher than or on par with vision based RPNs. We evaluate RRPN
using the [Fast R-CNN](https://arxiv.org/abs/1504.08083) network on the
[NuScenes](https://www.nuscenes.org/) dataset and compare the results with the
[Selective Search](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf)
algorithm. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1905.00526).

## Citing RRPN

If you find RRPN useful in your research, please consider citing:

```
@article{Nabati2019RRPN,
    title={RRPN: Radar Region Proposal Network for Object Detection in Autonomous Vehicles},
    author={Nabati, Ramin and Qi, Hairong},
    journal={arXiv:1905.00526},
    year={2019}
}
```

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Prepare Datasets](#prepare-datasets)
4. [Demo](#demo)
5. [Usage](#usage)

--------------------------------------------------------------------------------

## Requirements

- ### Python3.7

  RRPN requires Python >= 3.7.0 to work. You can download Python [here](https://www.python.org/downloads/).

- ### Virtual Environment
  **Using a virtual environment is optional but highly recommended.*
  1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/installation/) and
    [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html)

  1. Create a virtual environment for Python 3.7

      ```bash
      mkvirtualenv RRPN --python /usr/local/bin/python3.7
      ```

      Note: path to your Python3.7 may be different.

- ### Caffe2
  - Follow the steps [here](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=prebuilt) to install Caffe2 with GPU support.

--------------------------------------------------------------------------------

## Installation

1. Clone the repo and install the prerequisites in a Python3.7 virtual
environment. Here the repository is cloned to the `~/RRPN`
folder. Change the `BASE_DIR` variable below if you want to clone it to another
directory.

    ```bash
    BASE_DIR=~/RRPN
    git clone https://github.com/mrnabati/RRPN $BASE_DIR
    cd $BASE_DIR
    
    # Install prerequisites in the virtual environment
    workon RRPN
    pip install -r requirements.txt
    ```

1. Install the COCO API (pycocotools)

    ```bash
    COCOAPI_DIR=~/cocoapi
    git clone https://github.com/cocodataset/cocoapi.git $COCOAPI_DIR
    cd $COCOAPI_DIR/PythonAPI
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages:
    python setup.py install
    ```

1. Set up Python modules for Detectron and add symlink to dataset:

    ```bash
    cd $BASE_DIR/detectron && make
    ln -s $BASE_DIR/data/datasets/nucoco $BASE_DIR/detectron/detectron/datasets/data/nucoco
    # Optional: Check that Detectron tests pass
    python $BASE_DIR/detectron/detectron/tests/test_spatial_narrow_as_op.py
    ```

## Prepare Datasets

1. Download the NuScenes teaser dataset archive files from its [Download Page](https://www.nuscenes.org/download), 
  unpack the archive files to `$BASE_DIR/data/datasets/nuscenes/` _without_
  overwriting folders that occur in multiple archives. Eventually you should
  have the following folder structure:

    ```bash
    $BASE_DIR/data/datasets/nuscenes/
      |- maps
      |- samples
      |- sweeps
      |_ v0.1
    ```

1. Convert the NuScenes dataset to the COCO dataset format. We call this new dataset `nucoco`:

    ```bash
    bash $BASE_DIR/experiments/convert_nuscenes_to_nucoco.sh
    ```

    By default, both the key-frame and non key-frame images from the front and 
    back cameras in the
    NuScenes dataset are used to generate the new dataset. Change the
    `INCLUDE_SWEEPS` parameter in the script to `False` if you only want to use
    the key-frame images.


## Demo

To be added soon.


## Usage
  
1. Generate RRPN proposals for the nucoco dataset:
   
    ```bash
    bash $BASE_DIR/experiments/generate_rrpn_proposals.sh
    ```

1. Download ppretrained Fast R-CNN models from the detectron
   [model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)
   into the `$BASE_DIR/data/models` directory. RRPN results are calculated 
   based on the 
   [R-101-FPN 2x](https://dl.fbaipublicfiles.com/detectron/36228933/12_2017_baselines/fast_rcnn_R-101-FPN_2x.yaml.09_26_27.jkOUTrrk/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl) 
   (model no. 36228933) and
   [X-101-32x8d-FPN 1x](https://dl.fbaipublicfiles.com/detectron/37119777/12_2017_baselines/fast_rcnn_X-101-32x8d-FPN_1x.yaml.06_38_03.d5N36egm/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl)
   (model no. 37119777) models.

1. Before starting finetuning on the nucoco dataset, remove last layer's weights in the
   the pretrained model by running the `remove_class_weights.sh` script in
   the `experiments` folder. Change the `INPUT_MODEL` and `OUPUT_MODEL` veriables
   in the script based on the path to pretrained models.

    ```bash
    bash $BASE_DIR/experiments/remove_class_weights.sh
    ```

1. Use the `finetune_nucoco.sh` script to start training using the RRPN proposals.
   Change the variables under `Parameters` in the script as needed.
    
    ```bash
    bash $BASE_DIR/experiments/finetune_nucoco.sh
    ```

1. After training is complete, use the `infer_nucoco.sh` script to run inference
   on the validation set images. Change the veriables under `Parameters` 
   in the script as needed.

    ```bash
    bash $BASE_DIR/experiments/infer_nucoco.sh
    ```

    results are saved in the model directory under `results\vis`.