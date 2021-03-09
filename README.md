# Custom_Train_MaskRCNN

Training on custom dataset with (multi/unique class) of a Mask RCNN

### Requirements (no specific version requirements)
```
  python3.6
  cuda==10.0
  cudnn==7.6.5
  tensorflow-gpu==1.14.0
  keras==2.2.5
  pycocotools
  matplotlib
  mrcnn
  tqdm
  numpy
  pylab
  scikit-image
```

### Structure
- dataset: folder where you put the train and val folders (read inside to know what to put).
- logs: folder where we store the intermediate/checkpoints and final weights after training
- weights: weights for the model (.h5 file), we fetch the weights from here for the test script
- weights_db: tells if weights is coco or not, to drop some layers for new classes.
- train_data.py: main script for train the model with the data.
- test_data.py: script to test the model with camera.

### Usage 
First training usage, with **--weights_db=coco** loads weights and drops some output layers to fit the new data:
```
   python train_model.py --dataset=./dataset --weights=./weights/mask_rcnn_coco.h5 --weights_db=coco
```
To test model trained use;
```
  python test_model.py
```

### Hardware requeriments

#### Train: 

For training is need atleast 12 GB of GPU and 8 of ram memory.  Inference works with cpu, but is very slow.

#### Val: 

For test is need atleast 2 GB of GPU and 4 of ram memory.  Inference works with cpu, but is quite slow.
