# Face Recognition Using Convolutional Neural Networks (CNN)

This repository contains a Jupyter Notebook that implements a Convolutional Neural Network (CNN) for face recognition using the ORL faces dataset. The model is designed to classify grayscale face images into 20 unique classes.

---

## Project Overview

This project demonstrates:
- Loading and preprocessing of the ORL face dataset.
- Building a Convolutional Neural Network (CNN) using Keras and TensorFlow.
- Training the model on the training data and validating its performance on a validation set.
- Evaluation of the model using classification reports and confusion matrices on both the training and test datasets.
- Visualization of model accuracy and loss over the training epochs.

---

## Dataset

The ORL faces dataset consists of grayscale images of faces with a resolution of 112x92 pixels. The dataset has been split into training and testing sets and is stored in a compressed `.npz` file format.

- **Training Data**: 240 images
- **Test Data**: 160 images
- **Image Dimensions**: 112x92 pixels, grayscale

---

## Model Architecture

The model architecture is based on a Convolutional Neural Network (CNN) with the following layers:
- **Input Layer**: Image input of shape `(112, 92, 1)` (grayscale image)
- **Conv2D + MaxPooling**: Three sets of convolutional layers followed by max-pooling layers
- **Flatten Layer**: To convert the 3D output into 1D
- **Dense Layers**: Three dense layers with ReLU activation and dropout
- **Output Layer**: Softmax activation to predict one of the 20 possible classes

---

## Visualizations

During training, the following visualizations are plotted:
- **Training and Validation Accuracy**: A plot showing how the accuracy changes over epochs.
- **Training and Validation Loss**: A plot showing the loss function changes during training.
- **Confusion Matrix**: A confusion matrix showing the model's performance on the validation and test data.

---

## Requirements

To run this project, the following dependencies are required:
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow (with Keras)
- scikit-learn
- mlxtend

You can install the required packages using the following command:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn mlxtend
