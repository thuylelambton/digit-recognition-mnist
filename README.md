# Introduction
This project aims to train a Convolutional Neural Network (CNN) model to recognise hand-written digits. The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used to train the model.

# Installation
```
pip install -r ./requirements.txt
```

# Train
Run `python train.py` to train the CNN model.

# Predict
1. Create/save the images containing hand-written digits to the `./data` directory.
1. Run `python predict.py` to see the predictions.

# Results

We run the model on some user generated images.
The visualization below displays the prediction result for each of the input images.
 
![alt text](/public/mnist-results.png "Results")
