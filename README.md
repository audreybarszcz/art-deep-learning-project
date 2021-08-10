# art-deep-learning-project
USF MSDS Deep Learning Final Project

Contributors: Audrey Barszcz & Daniel Carrera

The dataset can be found [here](https://www.kaggle.com/c/painter-by-numbers/data)

## Goal
This project aims to predict whether two artworks are by the same artist or not using a neural network.

Same artist (Matisse):

<img src="https://d26jxt5097u8sr.cloudfront.net/s3fs-public/Full_matisse2.jpg" width="312" height="253">  <img src="https://www.goldmarkart.com/images/stories/virtuemart/product/La-Gerbe1.jpg" width="315" height="251">

Different artists (Seurat & Signac):

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg/1200px-A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg" width="444" height="202">  <img src="https://impressionistarts.com/static/81cb87fd29c30d1cd5e1e0c46b827e3e/14b42/paul-signac-in-the-time-of-harmony.jpg" width="400" height="300">

## About the dataset
The dataset includes 103,250 unique images of artworks by 1659 different artists, with up to 500 works per artist.

For this project, however, a subset of artists were chosen to train and test on.

## Data Preprocessing
All images used for this task were resized to 224x224.

## Model
A MobileNetV3-small pretrained network is used for this project. The final two linear layers are re-trained for this particular task.
