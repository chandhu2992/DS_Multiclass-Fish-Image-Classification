# Fish Image Classification using Deep Learning

## Project Overview
This project classifies fish species from images using Deep Learning and Convolutional Neural Networks (CNNs). It compares a custom CNN model with multiple transfer learning architectures including VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0.

## Objectives
- Classify fish images into their respective species.
- Compare the performance of different CNN architectures.
- Improve classification accuracy using transfer learning.
- Evaluate models using standard classification metrics.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Dataset Structure
dataset/
├── train/
├── validation/
└── test/

Each class should have its own folder containing fish images.

## Workflow
1. Load image dataset.
2. Preprocess and normalize images.
3. Create data generators.
4. Handle class imbalance using class weights.
5. Train:
   - Custom CNN
   - VGG16
   - ResNet50
   - MobileNet
   - InceptionV3
   - EfficientNetB0
6. Evaluate models.
7. Compare results and select the best model.

## Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Key Features
- Image normalization
- Batch processing
- Transfer learning
- Early stopping
- Model checkpointing
- Class imbalance handling

## Expected Applications
- Fish species identification
- Fisheries management
- Marine biodiversity monitoring
- Automated sorting systems

## Conclusion
The project demonstrates how deep learning and transfer learning can be used to accurately classify fish species from images. The best-performing model can be deployed for real-world fish recognition applications.
