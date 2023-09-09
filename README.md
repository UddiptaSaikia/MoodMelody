# Emotion-Based Music Recommender System

## Overview

This project aims to create an Emotion-Based Music Recommender System using Convolutional Neural Networks (CNNs) and Transfer Learning. The system takes input from a webcam to detect the user's emotion and recommends music based on the detected emotion, favorite artist, and language preferences.

## Table of Contents

- [Initial Dataset and Pre-processing](#initial-dataset-and-pre-processing)
- [Building CNN Sequential Model and Metrics Achieved](#building-cnn-sequential-model-and-metrics-achieved)
- [Recommender System](#recommender-system)
- [Transfer Learning to Improve Accuracy](#transfer-learning-to-improve-accuracy)

## Initial Dataset and Pre-processing

The project began with an initial dataset containing seven emotion classes: 'happy,' 'angry,' 'neutral,' 'surprised,' 'fear,' 'sad,' and 'disgust.' The dataset consisted of approximately 39,000 images. To simplify the music recommendation system, we removed four classes: 'surprised,' 'fear,' 'neutral,' and 'disgust,' resulting in a reduced dataset of 26,000 images.

The images were pre-processed as follows:
- Grayscaled
- Scaled down to 48x48 pixels
- Feature values ranged between 0 and 255 (RGB values of a 2D array)
- Classes were label-encoded using Scikit-learn.

## Building CNN Sequential Model and Metrics Achieved

The CNN model architecture includes:
- 4 Conv2D layers
- 2 Dense layers with ReLU activation functions
- 1 dense output layer with a softmax activation function
- 6 dropout layers to prevent overfitting.

The model was compiled with the 'adam' optimizer and 'categorical_crossentropy' loss function. A total of 300 epochs were run, resulting in the following metrics:
- Validation accuracy: 73.83%
- Overall accuracy: 95.45%.

The trained model was saved as a JSON file for integration with a Streamlit front-end API.

## Recommender System

The system utilizes the detected emotion from the webcam input, as well as user-provided information such as favorite artist and language preferences. Based on this data, the system redirects the user to YouTube and suggests songs that match their mood and preferences.

## Transfer Learning to Improve Accuracy

Due to the relatively low validation accuracy of the initial model, we explored the use of transfer learning techniques to enhance the model's performance. Further research and experimentation were conducted to increase the accuracy and scope of the emotion-based music recommender system.


