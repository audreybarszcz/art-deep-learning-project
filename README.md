# art-deep-learning-project
USF MSDS Deep Learning Final Project

Contributors: Audrey Barszcz & Daniel Carrera

The original dataset can be found [here](https://www.kaggle.com/c/painter-by-numbers/data).

## Goal
This project aims to predict whether two artworks are by the same artist or not using a neural network. This is a difficult problem, as portrayed by the examples below: a single artist can have multiple styles or paint radically different subjects, but should still be classified as the same artist, or multiple artists can have very similar styles with a similar subject, but should be classified as not the same artist.

Same artist (Matisse):

<img src="https://d26jxt5097u8sr.cloudfront.net/s3fs-public/Full_matisse2.jpg" width="312" height="253">  <img src="https://www.goldmarkart.com/images/stories/virtuemart/product/La-Gerbe1.jpg" width="315" height="251">

Different artists (Seurat & Signac):

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg/1200px-A_Sunday_on_La_Grande_Jatte%2C_Georges_Seurat%2C_1884.jpg" width="391" height="263">  <img src="https://impressionistarts.com/static/81cb87fd29c30d1cd5e1e0c46b827e3e/14b42/paul-signac-in-the-time-of-harmony.jpg" width="350" height="263">

Additionally, this problem has highly imbalanced classes. Most paintings are not by the same artist, the original Kaggle dataset has 1.3% same artist pairs and 98.7% different artist pairs.

## About the dataset
The dataset includes 103,250 unique images of artworks by 1659 different artists, with up to 500 works per artist.

For this project, however, a subset of artists were chosen to train and test on. 

Only artists with works in both the training and test dataset were chosen. Of those 1500 artists, 52 artists were chosen. Most of these artists were impressionist/post-impressionist artists; many of the artists had similar styles. A total of 13,894 images were used in the task (10,720 train, 3,174 test). Each artist had between 43-500 works across both training and test sets.

To create the matched pairs dataset, each of the 13,894 works were randomly paired with 36 other works to yield the valid and invalid pairs. 

To make the training set more balanced, and managable in terms of size, 5% of the invalid training pairs were chosen from all invalid pairs to yield a ratio of 1.75:1 invalid to valid pairs.

## How to run the notebook
Download the data from [Kaggle](https://www.kaggle.com/c/painter-by-numbers/data).

The Siamese model notebook includes all the data preprocessing steps to choose a subset of artists to work with, create the matched pairs dataframe, and resize the images.

Functions.py includes 3 simple functions used in the notebooks. It includes 2 functions to resize images and 1 function to read in the images while training/testing.

In the notebook where pre-trained models are loaded in `model = models.{alexnet}(pretrained=True)`, AlexNet can be replaced with other pre-trained image classification models or pre-trained can be set to `False` to use the model architecture and train all the weights of the model.

## Data Preprocessing
All images used for this task were resized first resized to an area of 256x256 with the original aspect ratio maintained. After resizing, a center crop of size 224x224 was used as input to the image classification model.

## Models
Two different approaches to this task were taken: a Siamese network and a simple classification model.
Both approaches used a pre-trained AlexNet image classification model to extract features from images.

For the Siamese network, the final classifier layer was removed. Training involved passing 2 different images through the same AlexNet model, followed by concatenating the output of the two images from AlexNet. The combined tensor of features was then passed through 2 linear layers to yield the binary output of whether or not two works were by the same artist.

For the classification model, the final classifier layer was replaced with a linear layer going from 4096 features to 52 features, which we could then pass through softmax to yield the artist prediction for a single image. To yield predictions of whether or not two works were by the same artist, both images were passed through the model trained to classify artists and the two outputs were compared to yield the final binary prediction.

## Results
Various hyperparameters were tried for the two approaches:

## Future directions
With more time, we wanted to try training our own image classifier network trained on artworks, since perhaps features that are good for classifying photo images are not as relevant for art.

We also wanted to try performing this task using different pre-train image classification networks such as VGG-19 or ResNet34.

Another approach we could have taken was to train 52 individual models to classify whether a work was by a single artist of not and then aggregate the results of the individal models to then answer the question of same artist or different artist.
