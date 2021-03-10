# Detecting Human-Object Interaction via Fabricated Compositional Learning

This code is based on PMFNet. Thanks for their excellent work!
We only change a few parts based on PMFNet and run the code with a single V100 GPU. You can review the code according to the git status.

**This code follows the implementation architecture of [roytseng-tw/mask-rcnn.pytorch](https://github.com/roytseng-tw/mask-rcnn.pytorch.git).** 


## Getting Started

### Requirements

Tested under python3.

- python packages
  - pytorch==0.4.1
  - torchvision==0.2.2
  - pyyaml==3.12
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - packaging
  - ipdb
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.

Assume the project is located at $ROOT.

### Compilation

Compile the NMS code:

```
cd $ROOT/lib 
sh make.sh
```

### Data and Pretrained Model Preparation

Create a data folder under the repo,

```
cd $ROOT
mkdir data
```

- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download).

  **Our data**:
  Download the our dataset annotations and detection/keypoint proposals from [our data](https://pan.baidu.com/s/1tgBgTm8LEpAZvlQrrahyBA).
  
  **Pose estimatiotn**
  We use the repo [pytorch-cpn](https://github.com/GengDavid/pytorch-cpn) to train our pose estimator. We have released our keypoint predictions of vcoco dataset on our data.
  
  And make sure to put the files as the following structure:

  ```
  data
  ├───coco
  │   ├─images
  │   │  ├─train2014
  │   │  ├─val2014 
  │   │
  │   ├─vcoco
  │      ├─annotations
  │      ├─annotations_with_keypoints
  │      ├─vcoco
  │
  ├───cache
  │   ├─addPredPose
  │
  ├───pretrained_model
      ├─e2e_faster_rcnn_R-50-FPN_1x_step119999.pth
      ├─vcoco_best_model_on_test.pth

  ```

## Training

```
cd $ROOT
sh script/train_vcoco_fcl.sh
```


## Test

```
cd $ROOT
sh script/test_vcoco_fcl.sh
```


