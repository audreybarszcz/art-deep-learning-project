# art-deep-learning-project
USF MSDS Deep Learning Final Project

Contributors: Audrey Barszcz & Daniel Carrera

The dataset can be found [here](https://www.kaggle.com/c/painter-by-numbers/data)

## Goal
This project aims to predict whether two artworks are by the same artist or not using a neural network.

Same artist:

<img src="https://d26jxt5097u8sr.cloudfront.net/s3fs-public/Full_matisse2.jpg" width="312" height="253">  <img src="https://www.goldmarkart.com/images/stories/virtuemart/product/La-Gerbe1.jpg" width="300" height="239">

## About the dataset
The dataset includes 103,250 unique images of artworks by 1659 different artists, with up to 500 works per artist.

For this project, however, a subset of artists were chosen to train and test on.

## Data Preprocessing
All images used for this task were resized to 224x224.

## Model
A MobileNetV3-small pretrained network is used for this project. The final two linear layers are re-trained for this particular task.
