# 🏞️ Landmark Classification and Tagging

This project applies Convolutional Neural Networks (CNNs) to automatically classify landmarks in images. The goal is to infer the location of a photo when no metadata (like GPS tags) is available — a common challenge in photo-sharing and storage applications.

By identifying landmarks, the system can enhance features like automatic tagging, photo organization, and location suggestion, improving user experience on platforms like Google Photos or Flickr.

## 📘 Project Overview

I developed two CNN-based models to classify landmarks from a large image dataset:

1. CNN from Scratch
- Design and train a custom CNN architecture.
- Perform data preprocessing and augmentation for better generalization.
- Evaluate model performance and export the trained network using TorchScript.

2. CNN using Transfer Learning
- Leverage pre-trained models (e.g., VGG, ResNet, or AlexNet).
- Fine-tune the network for the landmark dataset.
- Compare results against the model built from scratch.

3. Model Deployment
- Deploy the best-performing model in a simple app.
- Allow users to upload an image and predict the most likely landmark.

## 🧠 Model Architectures
**CNN from Scratch**
- Residual Blocks with skip connections for deep feature learning
- Convolutional + Batch Normalization + ReLU layers for feature extraction
- Adaptive Average Pooling to reduce spatial dimensions
- Fully connected layers for classification
- Regularization with Dropout (0.2)
- Activation: ReLU
- Output: Linear layer for multi-class landmark prediction
- Loss Function: Cross Entropy Loss
- Optimizer: Adam


**Transfer Learning Model**
- Base Model: Pretrained ResNet18 from torchvision.models
- Frozen Layers: All convolutional layers (feature extractor)
- Custom Classifier Head:
- Fully connected layer replacing the original fc layer
- Input: num_ftrs (from ResNet feature extractor)
- Output: 50 classes
- Activation: ReLU (inherited from backbone)
- Loss Function: Cross Entropy Loss
- Optimizer: Adam (on final layer parameters only)

## ⚙️ Technologies Used
- Python 3
- PyTorch
- TorchVision
- NumPy, Pandas
- Matplotlib
- Pillow (PIL)
- Gradio (for deployment)

## 🚀 Results
- Achieved high classification accuracy using transfer learning.
- Demonstrated strong generalization on unseen landmark images.
- Successfully deployed a functional app for image-based landmark prediction.

## 📈 Future Improvements
- Train on a larger and more diverse landmark dataset.
- Implement multi-label tagging for images with multiple landmarks.
- Optimize the app for faster inference using model quantization.