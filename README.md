# SqueezeNet-Keras for Vehicle Classification

This repository contains a Keras implementation of SqueezeNet, Convolutional Neural Networks (CNN) based image classifier.
SqueezeNet model is trained on MIO-TCD classification dataset to correctly label each image.

## References
* [SqueezeNet Paper](https://arxiv.org/pdf/1602.07360.pdf)
* [SqueezeNet Official Repository](https://github.com/DeepScale/SqueezeNet)
* [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)
* [MIO-TCD: A New Benchmark Dataset for Vehicle Classification and Localization](https://ieeexplore.ieee.org/document/8387876)
* [MIO-TCD dataset](http://podoce.dinf.usherbrooke.ca/challenge/dataset/)

## Implementation
The SqueezeNet architecture is implemented using Keras Functional API with TensorFlow backend. SqueezeNet architecture implemented in this repo has 9 fire modules as described in the paper but number of filters in convolutional layers are reduced for MIO-TCD dataset with 11 classes.

The following library versions are used:
* Keras 2.2.4
* TensorFlow 1.8.0

## Dataset
MIO-TCD classification dataset consists of 648,959 images divided into 11 categories {Articulated truck, Bicycle, Bus, Car, Motorcycle, 
Non-motorized vehicle, Pedestrian, Pickup truck, Single unit truck, Work van, Background}.
* Download the dataset from [here](http://podoce.dinf.usherbrooke.ca/static/dataset/MIO-TCD-Classification-Code.tar).
* Unzip the images and corresponding labels.

## Training
The training details of SqueezeNet for Vehicle Classification:
* SGD with momentum is used to train the CNN.
* Initial learning rate is 0.001 and "Linear Learning Rate Decay" policy is used as described in the paper [Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228). The learning rate is decreased linearly by each SGD batch update.
* Validation split of 0.1 is used to evaluate the performance after each epoch.
* Run train.py script to start training the network.
  ```
  python ./train.py --dir /home/dataset/train --batchsize 64 --epochs 10
  ```
* After training for 10 epocs, the SqueezeNet model reaches 92 % validation accuracy.
* Training history of accuracy, loss and learning rate for 10 epochs:
<p align="center">
  <img src="https://github.com/mpoyraz/SqueezeNet-Keras/blob/master/images/training_history.png" width="400">
</p>

## Results
After training finishes, train.py saves the SqueezeNet model and model parameters to be used for prediction.
  ```
  ./model/squeezenet_model.h5
  ./model/model_parms.json
  ```
To predict the label for a vehicle, run the predict.py script.
  ```
  python ./predict.py --test-image ./images/test_image.jpg
  ```
 The prediction results from test images in the dataset:
 <p align="center">
  <img src="https://github.com/mpoyraz/SqueezeNet-Keras/blob/master/images/test_images_prediction.png" width="600">
</p>
