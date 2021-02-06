
Resnet 18 trained on CIFAR 10 Data set 

Transformation : (using Albementations package)

Data augmentation (DA) is an effective alternative to obtaining valid data, and can generate new labeled data based on existing data using "label-preserving transformations". 

Designing appropriate DA policies requires a lot of expert experience and is time-consuming, and the evaluation of searching the optimal policies is costly. Moreover, the policies generated using DA policies are usually not reusable.

Generally, the more data, the better deep neural networks can do. Insufficient data can lead to model over-fitting, which will reduce the generalization performance of the model on the test set.

Source : we find that the performance on vision tasks increases logarithmically based on the volume of training data size

 

Techniques such as:

DropOut
Batch Normalization
L1/L2 regularization 
Layer normalization 
have been proposed to help combat over-fitting, but they will fall short if data is limited

CUT OUT is a great agumentation technique , that implements what dropout does but with more effectiveness. FORCES Network to learn all possible features from a CLASS.



DIAGNOSING DNN

Grad CAM:

Steps: 

Load a pre-trained model
Load an image which can be processed by this model (224x224 for VGG16 why?)
Infer the image and get the topmost class index
Take the output of the final convolutional layer
Compute the gradient of the class output value w.r.t to L feature maps
Pool the gradients over all the axes leaving out the channel dimension
Weigh the output feature map with the computed gradients (+ve)
Average the weighted feature maps along channels
Normalize the heat map to make the values between 0 and 1


![plot](./test/dog/dog1.png)
![plot](./Gradcam_out/map.jpg)

