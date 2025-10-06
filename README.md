👇

🩺 Pneumonia Detection using Deep Learning
📘 Overview

This project is a deep learning-based medical image analysis system designed to detect Pneumonia from chest X-ray images.
It uses Convolutional Neural Networks (CNN) for image classification and integrates Grad-CAM visualization to highlight the infected regions, improving interpretability and trust in model predictions.

🚀 Features

Detects Normal and Pneumonia cases from X-ray images.

Integrated Grad-CAM to visualize infection regions.

Includes image preprocessing pipeline using OpenCV and NumPy.

Handles false negatives using threshold-based logic and fallback prediction.

Optimized model performance with data augmentation and dropout regularization.

🧠 Tech Stack

Programming Language: Python

Libraries: TensorFlow/Keras, OpenCV, NumPy, Matplotlib

Explainability: Grad-CAM

Model Type: Convolutional Neural Network (CNN)

⚙️ Workflow

Dataset Loading: Import chest X-ray dataset and split into train/test sets.

Preprocessing: Resize, normalize, and augment images to improve generalization.

Model Training: Train CNN model with dropout and ReLU activation layers.

Evaluation: Measure accuracy, precision, recall, and F1 score.

Explainability: Generate Grad-CAM heatmaps for each prediction.

False Negative Fix: Implement logic to minimize misclassification of Pneumonia as Normal.

📊 Results

Achieved high accuracy in detecting Pneumonia vs. Normal cases.

Grad-CAM visualizations clearly highlighted infected regions in X-rays.

Reduced false negatives through enhanced preprocessing and threshold tuning.
