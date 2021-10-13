This repository contains explanation of Yolov1 object detection architecture and its implementation in pytorch.

You might have worked on Image Classification, where you pass an image as input to the network and it returns the probability of the image to be in certain class.

<br>Object detection involves classification of the object along with localising its position in the image.
<br>Finding location as well as class of the object is called object detection. But there is a slight twist. We don't know how many classes and objects could be there in the image.
<br>Our final goal in object detection task is to output some set of bounding boxes around different objects present in the image.

<br>
If we look into existing object detection then we can divide them into two broad categories:

1. Two-stage object detectors
2. Single-stage object detectors

## Two-stage object detectors

This type of object detectors performs the detection in two stages. In the first stage, they find all the possible regions of the image where an object can be present and in the second stage, they perform regression to find out the boundary boxes.

e.g: RCNN, Fast RCNN, Faster RCNN

If you've heard the above terms first time, then don't worry. We won't be covering those in this article but not gonna use them too.

## Single-stage object detectors

These types of detectors perform the region finding and regression to find the boundary boxes in single-stage. It only takes one pass through the neural network to find the boundary boxes of an object.

e.g: **SSD** and **YOLO** are the single stage detectors.
<br>Read the paper here: [Yolov1](https://arxiv.org/pdf/1506.02640v1)

