## Fine-grained image recognition

[简体中文](README_CN.md)<br/>
English

### Introduction
This repository contains recognition code for VGG16 and ResNet50 as backbone networks, and CBAM modules for refining image features and obtaining image fine-grained. An easy-to-use website is also provided to implement the recognition logic.

### Contain
- `images/`: code to store the open source address of the dataset and divide it into subclasses (for information only)
- `resnet/`: network framework, training code, recognition code and CBAM module using ResNet50 as the backbone network
- `vgg/`: network framework, training code, recognition code with VGG16 as the backbone network
- `web`: store the front-end web pages
- `class_indices.json`: stores subclass sequences, used for recognition comparison, generated automatically during training

### Requirements
- Python 3.9
- TensorFlow 2.0
> TensorFlow, Python, Keras and other wheels used need to be installed, and because it is running on macOS, the reference format needs to be modified to suit the system.

### Learning
- The components and operating principles of CNNs, such as convolution and pooling operations
- Pre-processing operations on data
- The components of the spatial attention mechanism and the channel attention mechanism
- The process of image recognition in action