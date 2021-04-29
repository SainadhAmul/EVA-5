
# Project:

## Problem Statement:
### Given input imag:

#### Generate 2 dense predictions:
- Depth Estimation - MiDaS
- Plane/planar surfaces Detection - PlanarRCNN

#### 1 object localisation:
- YOLO v3

### ARCHITECTURE PLAN:

![Custom](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/Arch%20Thanos%20Net%20(1).png)



### Approach:

The common thing in these networks is they all use a Head/Backbone(feature extractor) that usually is RESNET! so we can use this to connect all three network to same-head. We can lock the params in the head/backbone network and the train 3 branches from the head.

The 3 different backbones  in each of the networks:
- MidasNet - ResNext101_32x8d_wsl
- Planercnn - ResNet101
- Yolov3 - Darknet-53

- First try to combine midas and yolo.

#### First ITERATION:
- MIDAS + YOLO:

![YOLO+MIDAS](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/Yolo_MIDAS_ARC.png)






## MIDAS UNDERSTANDING:

This is a pretrained Monocular depth estimatation network

### MIDAS ARCHITECTURE:

![midas](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/midas.jpeg)


As Midas alraedy has good pretrained backbone. The repo provides 2 options

### MIDAS BACKBONE

#### Large model
- model1: weights model-f6b98070.pt 
- back_bone = resnet101 44.5 m paramas

#### Small model
- model2 : model-small-70d6b9c8.pt 
- ient lite 3 EfficientNet B0 4.9M params




## YOLO understanding:

Object localization network


### YOLO Architecture:

![YOLO](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/YOLO%20v3.jpeg)




## CODE STRUCTURE (PLAN):

### BASE -> MIDAS REPO + YOLO REPO + PLanarRCNN REPO

#### train module: 
- Train the model on 3000 images and 3000*3 pre genreated images using the individual models.

#### evalute module:
- Generate inference on inp images

#### Dataset/Data loader/transforms module:
- Dataset creation (len, get item -- 1 img 3 labels)

#### model skeleton :
- Model architecure ( MiDaS + YOLO + PlanarRCNN)


### Directory Structure:


```bash

 Root
    ├── Arc Images 
    ├── MiDaS  ( FULL Repo) 
    ├── Yolov3  ( FULL Repo) 
    ├── cfg.sjon  ( Yolo Config) 
    ├── train.py  
    ├── evalute.py 
    ├── dataset.py
    ├── Midas_YOLO
    ├── Final_Arc.py      

Lables
    ├── Annotated Images YOLO
    |   ├── images
    |   |   ├── image_0001.jpg
    |   |   ├── image_0002.jpg
    |   |   ├── ...
    |   |   ├── image_3521.jpg
    |   ├── labels
    |   |   ├── image_0001.txt
    |   |   ├── image_0002.txt
    |   |   ├── ...
    |   |   ├── image_3521.txt
    ├── Depth Images using MiDaS
    |   ├── image_0001.png
    |   ├── image_0002.png
    |   ├── ...
    |   └── image_3521.png
    |── Planer images using PlanerCNN
        ├── image_0001.png
        ├── image_0002.png
        ├── ...
        └── image_3521.png    
```

