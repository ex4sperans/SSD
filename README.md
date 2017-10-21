# SSD
Single Shot Multibox Detector (https://arxiv.org/abs/1512.02325) implemented in Tensorflow.

NOTE: Repo is under development.

Some preliminary results obtained from SSD300 trained on PASCAL VOC2007 trainval dataset:

<img src="https://github.com/ex4sperans/SSD/blob/master/images/image1.jpg?raw=true" width="300"/> <img src="https://github.com/ex4sperans/SSD/blob/master/images/image2.jpg?raw=true" width="330"/> 
<img src="https://github.com/ex4sperans/SSD/blob/master/images/image3.jpg?raw=true" width="300"/> <img src="https://github.com/ex4sperans/SSD/blob/master/images/image4.jpg?raw=true" width="330"/>

#### General notes on implementation.

This version of Single Shot Multibox Detector is mostly implemented according to the original paper, albeit it could insignificantly differ in details. For instance, I found [Adam](https://arxiv.org/abs/1412.6980) to be more decent optimizer for this problem compared to regular SGD with momentum. 
