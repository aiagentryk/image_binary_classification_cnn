# Binary Image Classification: Men vs. Women

A lightweight Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images into Man or Woman. It features an end-to-end computer vision pipeline from directory management to image preprocessing, augmentation, evaluation, and inference.



## Dataset Overview

The dataset consists of 1,615 total images split evenly to prevent class bias:

* Train: 800 images (400 man / 400 woman)
* Validation: 400 images (200 man / 200 woman)
* Test: 400 images (200 man / 200 woman)
* Predict: 15 unlabelled target images for final inference


## Model Architecture

The custom CNN processes 150x150 RGB images using a sequential pipeline:

* Preprocessing: Pixel rescaling from range [0, 255] to [0, 1].
* Data Augmentation: Real-time spatial transformations including horizontal flipping, random rotation, random zoom, and contrast adjustment.
* Feature Extraction: Three stacked convolutional blocks using Conv2D and MaxPooling2D layers with expanding filter sizes (32, 64, 128).
* Classification Head: A fully connected layer with 128 units, a Dropout layer set to 0.5 to prevent overfitting, and a single Sigmoid node for binary classification output.



## Pipeline Workflow

1. Data Ingestion: Images are loaded dynamically from directory paths using Keras utilities, resizing all samples to uniform dimensions and organizing them into batches of 32.
2. Model Training: The network is compiled using the Adam optimizer and binary cross-entropy loss, trained across 15 epochs while tracking validation metrics.
3. Evaluation: Performance is tested against unseen data in the test split to measure accuracy, precision, recall, and F1-score.
4. Model Export: The trained network, weights, and configuration are serialized to a single Keras format file (man_woman_cnn.keras).
5. Visual Inference: Unlabelled images from the prediction directory are passed through the model to output target labels alongside visual confidence percentages.



## Technical Stack

* Language: Python
* Deep Learning Framework: TensorFlow / Keras
* Evaluation & Metrics: Scikit-Learn
* Data Processing & Visualization: Pathlib, NumPy, Matplotlib
