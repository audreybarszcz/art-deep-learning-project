# art-deep-learning-project
USF MSDS Deep Learning Final Project

Contributors: Audrey Barszcz & Daniel Carrera

The dataset can be found [here](https://www.kaggle.com/c/painter-by-numbers/data)

## Goal
This project aims to predict whether two artworks are by the same artist or not using a neural network.

Not the same artist:

<img src="https://www.vangoghgallery.com/skin/img/sunflower_full.jpg" width="375" height="476"> <img src="https://uploads1.wikiart.org/images/paul-cezanne/still-life-with-skull-1898.jpg!Large.jpg" width="345" height="300">

Same artist:
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/b/bc/Old_guitarist_chicago.jpg/1200px-Old_guitarist_chicago.jpg" width="400" height="600"> <img src="https://media.architecturaldigest.com/photos/6054f6d5b1a16752e2a94e24/1:1/w_1529,h_1529,c_limit/4%20%20%C2%A9%20akg-images%EF%80%A2Andre%CC%81%20Held%20%EF%80%A2%202019%20Estate%20of%20Pablo%20Picasso%20%EF%80%A2%20Artists%20Rights%20Society%20(ARS),%20New%20York.jpg" width="383" height="383"> 



## About the dataset
The dataset includes 103,250 unique images of artworks by 1659 different artists, with up to 500 works per artist.

For this project, however, a subset of artists were chosen to train and test on.

## Data Preprocessing
All images used for this task were resized to 224x224.

## Model
A MobileNetV3-small pretrained network is used for this project. The final two linear layers are re-trained for this particular task.
