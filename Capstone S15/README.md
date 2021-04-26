
# Project:

## Problem Statement:
### Given input imag:

#### generate 2 dense predictions
- Depth Estimation - MiDaS
- Plane/planar surfaces Detection - PlanarRCNN

#### 1 obj localization pred:
- YOLO v3

### ARCHITECTURE PLAN:

![Custom](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/Arch%20Thanos%20Net%20(1).png)
![Custom](EVA-5/Capstone S15/Architecture diagrams/Arch Thanos Net (1).png)


### Approach:

First ITERATION:
MIDAS + YOLO:




## MIDAS UNDERSTANDING:

This is a pretrained Monocular depth estimatation network

### MIDAS ARCHITECTURE:

![midas](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/midas.jpeg)
![midas](EVA-5/Capstone S15/Architecture diagrams/midas.jpeg)

As Midas alraedy has good pretrained backbone. The repo provides 2 options

### MIDAS BACKBONE

model-1: weights model-f6b98070.pt 
back_bone = resnet101 44.5 m paramas

model2 : model-small-70d6b9c8.pt 
back_bone =  efficient lite 3 EfficientNet B0 4.9M params




## YOLO understanding:

Object localization network


### YOLO Architecture:

![YOLO](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/YOLO%20v3.jpeg)
![YOLO](EVA-5/Capstone S15/Architecture diagrams/YOLO v3.jpeg)



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

Construction PPE Kit
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

