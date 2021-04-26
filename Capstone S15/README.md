
Project:

Problem Statement:
Given input imag:

generate 2 dense predictions
Depth Estimation - MiDaS
Plane/planar surfaces Detection - PlanarRCNN

1 obj localization pred:
YOLO v3

Approach:

First ITERATION:

MIDAS + YOLO:

As Midas alraedy has good pretrained backbone. The repo provides 2 options

MIDAS BACKBONE

model-1: weights model-f6b98070.pt 
back_bone = resnet101 44.5 m paramas

model2 : model-small-70d6b9c8.pt 
back_bone =  efficient lite 3 EfficientNet B0 4.9M params





