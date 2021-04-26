
Project:

Problem Statement:
Given input imag:

generate 2 dense predictions
Depth Estimation - MiDaS
Plane/planar surfaces Detection - PlanarRCNN

1 obj localization pred:
YOLO v3

ARCHITECTURE PLAN:

![alt text](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/Arch%20Thanos%20Net%20(1).png)


Approach:

First ITERATION:

MIDAS + YOLO:




MIDAS UNDERSTANDING:

This is a pretrained Monocular depth estimatation network

MIDAS ARCHITECTURE:

![alt text](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/midas.jpeg)


As Midas alraedy has good pretrained backbone. The repo provides 2 options

MIDAS BACKBONE

model-1: weights model-f6b98070.pt 
back_bone = resnet101 44.5 m paramas

model2 : model-small-70d6b9c8.pt 
back_bone =  efficient lite 3 EfficientNet B0 4.9M params




YOLO understanding:

Object localization network


YOLO Architecture:

![alt text](https://github.com/SainadhAmul/EVA-5/blob/main/Capstone%20S15/Architecture%20diagrams/YOLO%20v3.jpeg)


