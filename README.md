# art-deep-learning-project
USF MSDS Deep Learning Final Project

Contributors: Audrey Barszcz & Daniel Carrera

The dataset can be found [here](https://www.kaggle.com/c/painter-by-numbers/data)

## Goal
This project aims to predict whether two artworks are by the same artist or not using a neural network.

![Guernica](https://upload.wikimedia.org/wikipedia/commons/0/0b/GUERNICA.jpg)
and
![Irises](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Irises-Vincent_van_Gogh.jpg/1200px-Irises-Vincent_van_Gogh.jpg)
are not by the same artist.

## About the dataset
The dataset includes 103,250 unique images of artworks by 1659 different artists, with up to 500 works per artist.

For this project, however, a subset of artists were chosen to train and test on.

## Data Preprocessing
All images used for this task were resized to 224x224.

## Model
A MobileNetV3-small pretrained network is used for this project. The final two linear layers are re-trained for this particular task.
