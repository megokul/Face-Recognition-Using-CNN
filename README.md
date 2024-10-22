# Face Recognition Using Convolutional Neural Networks (CNN)

## Project Overview
This project focuses on face recognition using Convolutional Neural Networks (CNN) to classify grayscale face images from the ORL dataset into 20 unique classes. The goal is to accurately recognize different faces from the dataset, which could be used for applications such as authentication, security, and user identification.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Project Workflow](#project-workflow)
5. [Model Performance](#model-performance)
6. [How to Use](#how-to-use)
7. [Installation](#installation)
8. [Future Improvements](#future-improvements)
9. [License](#license)

---

## Introduction
Face recognition is a widely used technique in various fields, including security, user authentication, and image classification. This project aims to recognize different faces from the **ORL face dataset** using a deep learning model based on a Convolutional Neural Network (CNN). The CNN model is designed to classify faces with high accuracy and robustness, even in grayscale images.

This project showcases the steps involved in loading the dataset, building and training the model, and evaluating its performance on unseen test data.

---

## Dataset
The **ORL Faces Dataset** consists of grayscale images of individuals, where each image has a resolution of 112x92 pixels. The dataset contains 400 images of 40 different individuals, with 10 images per individual.

Classes:
1. **Training Data**: 240 images
2. **Testing Data**: 160 images
3. **Image Dimensions**: 112x92 pixels, grayscale

Each face is resized and normalized for feeding into the CNN model for training and evaluation.

---

## Technologies Used
- **Python**: For data manipulation and machine learning model development.
- **NumPy**: Numerical operations for handling arrays and matrices.
- **Pandas**: Data manipulation and exploration.
- **Matplotlib**: Data visualization.
- **TensorFlow**: Building and training the CNN model.
- **Keras**: High-level neural network API running on TensorFlow.
- **Scikit-learn**: Model evaluation and dataset splitting.
- **Mlxtend**: Plotting confusion matrix for model evaluation.

---

## Project Workflow
1. **Data Loading & Preprocessing**: Loading the ORL dataset, reshaping the images, and normalizing pixel values for the CNN model.
2. **Data Visualization**: Visualizing a sample image from the dataset to understand the data format and content.
3. **Model Building**: Building a Convolutional Neural Network (CNN) using Keras with TensorFlow as the backend.
4. **Training the Model**: Training the CNN model on the training data while validating performance using validation data.
5. **Model Evaluation**: Evaluating the model's performance using accuracy, precision, recall, confusion matrix, and classification reports.
6. **Visualization**: Plotting the training and validation accuracy and loss across epochs to understand the model's learning progress.

---

## Model Performance
The CNN model achieved impressive results on the ORL dataset:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~93%

The best results were obtained after experimenting with different layers, activation functions, and dropout rates to reduce overfitting.

Evaluation Metrics:
- **Accuracy**: The percentage of correct predictions.
- **Precision**: Ability to identify only relevant data points (correct classifications).
- **Recall**: The percentage of actual positives correctly classified.
- **Confusion Matrix**: A detailed matrix showcasing model performance for each class.

---

## How to Use
### Clone the repository:
```bash
git clone https://github.com/megokul/Face-Recognition-Using-CNN.git
```

## Installation
Ensure you have the following dependencies installed before running the notebook. You can install them using:
```bash
pip install numpy pandas matplotlib tensorflow scikit-learn keras mlxtend
```
Alternatively, you can install all required dependencies at once using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Future Improvements
Here are a few potential improvements for future iterations of this project:

- **Data Augmentation**: Increase the robustness of the model by augmenting the dataset with techniques like rotation, flipping, and zooming.
- **Transfer Learning**: Implement transfer learning by using pre-trained models like VGG16, ResNet, etc., to further enhance accuracy.
- **Real-time Face Recognition**: Extend the project to detect and recognize faces in real-time using webcam or live video streams.
- **Additional Regularization**: Further experimentation with L2 regularization and more aggressive dropout to prevent overfitting.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code in this repository.

