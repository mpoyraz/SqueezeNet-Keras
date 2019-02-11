# SqueezeNet-Keras for Vehicle Classification

This repository contains a Keras implementation of SqueezeNet, Convolutional Neural Networks based image classifier.
SqueezeNet model is trained on MIO-TCD classification dataset to correctly label each image.

## References
* [SqueezeNet Paper](https://arxiv.org/pdf/1602.07360.pdf)
* [SqueezeNet Official Repository](https://github.com/DeepScale/SqueezeNet)
* [MIO-TCD: A New Benchmark Dataset for Vehicle Classification and Localization](https://ieeexplore.ieee.org/document/8387876)
* [MIO-TCD dataset](http://podoce.dinf.usherbrooke.ca/challenge/dataset/)

## Implementation
The SqueezeNet architecture is implemented using Keras Functional API with TensorFlow backend. SqueezeNet architecture implemented in this repo has 9 fire modules as described in the paper but number of filters in convolutional layers are reduced for MIO-TCD dataset with 11 classes.

The following library versions are used:
* Keras 2.2.4
* TensorFlow 1.8.0
