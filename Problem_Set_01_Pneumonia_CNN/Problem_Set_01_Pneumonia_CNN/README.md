# Pediatric Pneumonia Detection using CNN

## Overview
This repository contains the solution for Problem Set 01. The objective of this project is to build a Convolutional Neural Network (CNN) capable of classifying pediatric chest X-ray images into two classes: **Pneumonia** and **Normal**. 

## Dataset Details
The dataset consists of 5,863 JPEG X-Ray images (anterior-posterior view) of pediatric patients (aged 1 to 5 years). The data is pre-divided into `train`, `test`, and `val` directories.

## Methodology & Approach
1. **Data Preprocessing**: Since medical datasets are sensitive to overfitting, I utilized `ImageDataGenerator` for data augmentation. Techniques like rotation, zooming, and horizontal flipping were applied to the training set. All images were resized to 150x150 pixels and normalized (pixel values scaled between 0 and 1).
2. **Model Architecture**: 
   - I designed a sequential CNN with 3 Convolutional layers to extract spatial features (like bone structures and lung opacity).
   - Each Conv2D layer is followed by a MaxPooling2D layer to reduce dimensionality.
   - The flattened output is passed through a Dense layer with 128 neurons.
   - A `Dropout` layer (40%) was intentionally added before the final output layer to prevent the model from memorizing the training data.
   - The final output layer uses a `sigmoid` activation function suitable for binary classification.
3. **Training**: The model was compiled using the `Adam` optimizer and `binary_crossentropy` loss function.

## Findings
- **Data Augmentation is Crucial**: Initial testing without augmentation led to rapid overfitting. Adding augmentation stabilized the validation accuracy.
- **Model Performance**: The CNN successfully differentiates between healthy lungs and lungs with pneumonia, achieving a reliable test accuracy. The dropout layer played a significant role in generalizing the model for unseen test data.
