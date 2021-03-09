# Custom_Train_MaskRCNN
![](https://img.shields.io/badge/<implementation>-<customtrain>-<success>)


[![ko-fi](https://www.ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R5R11K2H4)

Training on custom dataset with (multi/unique class) of a Mask RCNN

### Requirements (no specific version requirements)
```
  tensorflow-gpu==1.14.0
  keras==2.2.5
  python3
  pycocotools
  matplotlib
  mrcnn
  tqdm
  numpy
  pylab
  skimage
```
Note: installation for mrcnn will be explained in the medium article linked in the repo.
### Structure
- dataset: folder where you put the train and val folders (read inside to know what to put)
- logs: folder where we store the intermediate/checkpoints and final weights after training
- weights: weights for the model, we fetch the weights from here for the test script
- weights_db: tells if weights is coco or not, to drop some layers for new classes.
- detect_segment_test.py: test script for the segmentation, displays mask on top of input image, usage given by --h argument
- train.py: main script for this section, read medium article to know what to modify

### Usage 
First training usage, more options showed in the train.py script as comment:
```
   python3 train.py train --dataset=./dataset --weights=./weights/mask_rcnn_coco.h5 --weights_db=coco
```

