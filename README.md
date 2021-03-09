# Custom_Train_MaskRCNN

Training on custom dataset with (multi/unique class) of a Mask RCNN

### Requirements 
Specific version
```
  python3.6-3.8
  cuda==10.0
  cudnn==7.6.5
  tensorflow-gpu==1.14.0
  keras==2.2.5
  scikit-image==0.16.2
  mrcnn==2.0.2
```
Non-Specific version

```
  Pillow
  opencv-python
  pycocotools
  matplotlib
  
  tqdm
  numpy
  pylab
```
Installation

```
  sudo apt-get install python3 python3-pip
  sudo pip install -r requeriments.txt
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

For training is need atleast 12 GB of GPU and 8 of ram memory.  Train works with cpu, but is very slow.

#### Val: 

For test is need atleast 2 GB of GPU and 4 of ram memory.  Inference works with cpu, but is quite slow.

### Dataset

The folder where the images and annotations should be place:

dataset/  
--train/  
----via_export_json_train.json  
----img1.jpg  
----img2.jpg   
----...  
--val/  
----via_export_json_val.json  
----img21.jpg  
----img22.jpg   
---....  
  
Where **via_export_json_val.json and via_export_json_train.json** are obtained from label all images  
using VGG label software. https://gitlab.com/vgg/via via-2.x.y
