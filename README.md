# Exploring Test-Time Adaptation for Object Detection in Continually Changing Environments

[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 

<p align="center">
  <img src="model.png"/ width="1200">
</p>

## Contents
1. [Installation Instructions](#installation-instructions)
2. [Dataset Preparation](#dataset-preparation)
3. [Execution Instructions](#execution-instructions)
    - [Source Pretraining](#source-pretraining)
    - [Adaptation](#adaptation)
4. [Acknowledgement](#acknowledgement)

## Installation Instructions
- We use Python 3.8, PyTorch 1.9.0 (CUDA 11.1 build).
- We codebase is built on [Detectron2](https://github.com/facebookresearch/detectron2).

```angular2
conda create -n AMROD python=3.8

Conda activate AMROD

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

cd AMROD
python -m pip install -e detectron2
cd detectron2
cd tools
```


## Dataset Preparation

* **Cityscapes-C**: Please refer to the official website [Cityscapes](https://www.cityscapes-dataset.com/downloads/) to download the validation set and the label of the Cityscapes (leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip), and covert the label format into coco format by cityscapes_to_coco.py (refer to [link](https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion). This is optional since detectron provide register method for cityscapes format.). Then apply the 12 corruptions at the severity level 5 to the validation set of clean cityscapes with corrupt.py (refer to [link](https://github.com/bethgelab/imagecorruptions)) to get Cityscapes-C.
* **SHIFT**: Please refer to the official website [SHIFT](https://www.vis.xyz/shift/) to download the front view of val set of the SHIFT-discrete and covert the Scalabel format into coco format (refer to /scalabel/label/to_coco.py in the [link](https://github.com/scalabel/scalabel)) for different condition based on the "weather_coarse" attributes.
* **ACDC**: Please refer to to the official website [ACDC](https://acdc.vision.ee.ethz.ch/download) to download the validation set and the label of ACDC.

Make sure that the AMROD is placed in the current user directory for dataset register. Download all the dataset into "AMROD/datasets" folder in the COCO format. 
Please refer to AMROD/detectron2/detectron2/data/datasets/builtin.py for the code of dataset register.

Please follow dataset structure below:
```
    - AMROD
        - datasets
            - ACDC
                - gt_detection
                    - fog
                        - instancesonly_fog_val_gt_detection.json
                    - night
                    ...
                - rgb_anon
                    - fog
                    - night
                    ...
            - shift
                - annotations
                    - gtfine_cloudy_val.json
                    - gtfine_foggy_val.json
                    ...
                - rgb_anon
                    - images
                        - val
                            - front
                                - 0aee-69fd
                                ...             
            - fog
                - annotations
                    - instancesonly_filtered_gtFine_val.json
                - leftImg8bit
                    - val
                        - frankfurt
                        ...
            - frost
                - annotations
                    - instancesonly_filtered_gtFine_val.json
                - leftImg8bit
                    - val
                        - frankfurt
                        ...
            ...
        - detectron2
        ...
```

## Execution Instructions

### Source Pretraining

- Download the source-trained [model weights](https://drive.google.com/file/d/1pjnmfRzz9zL_CuT-bXfR5W8J0KGw9Va4/view?usp=sharing) and and put them in AMROD/detectron2/tools/output/res50_fbn_1x and AMROD/detectron2/tools/output/res50_shift folder respectively

The two models was pretrained respectively in the training set of Cityscapes and clear condition of SHIFT with a learning rate of 0.001. 

### Adaptation

- After training, load the source-trained weights and perform adaptation.
There are four config file corresponding to the four adaptation tasks in the paper, i.e. cfg_cityscapes_c_short.yaml, cfg_cityscapes_c_long.yaml, cfg_shift_short.yaml, and cfg_ACDC_long.yaml.yaml.

For examble, using
```angular2
cd AMROD/detectron2/tools
CUDA_VISIBLE_DEVICES=$GPU_ID python adapt.py --config-file cfg_cityscapes_c_long.yaml
```
for Cityscapes-to-Cityscapes-C long-term CTTA task.

## Acknowledgement

We thank the developers and authors of [Detectron2](https://github.com/facebookresearch/detectron2) for releasing their helpful codebases.