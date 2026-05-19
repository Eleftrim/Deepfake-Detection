# Deepfake Detection using CNN and Autoencoders

A deep learning project for binary classification of real vs. fake face images,
using two complementary approaches: a Convolutional Neural Network (CNN) and
an Autoencoder-based anomaly detector. Evaluated with 5-Fold Stratified
Cross-Validation across two Kaggle datasets.

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `detection.py` | Core detection script |
| `trainFold12.ipynb` | CNN training – Folds 1 & 2 |
| `trainFold34.ipynb` | CNN training – Folds 3 & 4 |
| `trainFold5.ipynb` | CNN training – Fold 5 |
| `validation.ipynb` | Validation metrics & confusion matrix |
| `test.ipynb` | Testing on 2nd dataset |

## 🗂️ Datasets

- **Dataset 1** (CNN training & evaluation):  
  `kaggle/input/deepfake-and-real-images/Dataset/Train`

- **Dataset 2** (Autoencoder training & evaluation):  
  `kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test`

## 🏗️ Models

### CNN
- Input: 150×150 RGB images
- 3 convolutional blocks (Conv2d → BatchNorm → ReLU → MaxPool)
- Fully connected layers: 512 → 128 → 1 (Sigmoid output)
- Training: Adam optimizer, BCELoss, ReduceLROnPlateau, Early Stopping

### Autoencoder
- Trained **only on real images**
- Encoder: 2× (Conv2d → ReLU → BatchNorm → MaxPool) + Linear bottleneck
- Decoder: Linear → ConvTranspose2d × 2 → Sigmoid
- Detection via reconstruction error threshold (0.3560)

## 📊 Results

### CNN – Dataset 1 (Validation, aggregated across folds)
| Metric | Score |
|--------|-------|
| Recall | 0.918 |
| Precision | 0.965 |
| F1-Score | 0.950 |
| Accuracy | 0.950 |

### CNN – Dataset 2 (Test)
| Metric | Score |
|--------|-------|
| Recall | 0.870 |
| Precision | 0.885 |
| F1-Score | 0.876 |
| Accuracy | 0.876 |

## ⚙️ Data Augmentation
Random horizontal flip, rotation (±15°), affine translation (10%),
color jitter (brightness/contrast/saturation/hue), normalization.

## 🔧 Requirements
torch
torchvision
numpy
scikit-learn
matplotlib
jupyter
