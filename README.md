# Weakly Supervised Remote Sensing Segmentation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Achieving 10.45% mIoU on DeepGlobe Satellite imagery using only 0.04% pixel supervision (100 points per class).**

## 📌 Overview

This project tackles **semantic segmentation under weak supervision** using sparse point annotations instead of full pixel-wise labels.

I implement a **Partial Cross Entropy Loss** that enables training segmentation models using only a small subset of labeled pixels.

Labeling satellite imagery at the pixel level is prohibitively expensive. This project implements a **Point-Supervised Semi-Supervised Learning (SSL)** framework to perform semantic segmentation with extreme label scarcity. 

The project explores:

* Training with **sparse labels (point supervision)**
* Comparing **CNN vs Transformer architectures**
* Understanding the impact of **label sparsity on performance**

---

## 🚀 Key Features

* ✅ Custom **Partial Cross Entropy Loss**
* ✅ Sparse label simulation from full masks
* ✅ Modular PyTorch project structure
* ✅ Support for multiple models:

  * U-Net (CNN-based)
  * SegFormer (Transformer-based)
* ✅ Experiment tracking (different number of points)
* ✅ Evaluation using mIoU and class-wise IoU
* ✅ Visualization of predictions vs sparse labels

## 🧠 Problem Statement

In real-world scenarios, dense segmentation labels are expensive.

👉 This project simulates a more realistic setup:

* Only a few pixels per image are labeled
* The model must learn segmentation from limited supervision

---

## 📊 Dataset

I use the DeepGlobe Land Cover Dataset.

* High-resolution satellite imagery
* 7 land cover classes:

  * Urban
  * Agriculture
  * Rangeland
  * Forest
  * Water
  * Barren
  * unknown

---

## ⚙️ Method

### 1. Sparse Label Generation

Full segmentation masks are converted into **point annotations** by randomly sampling a fixed number of pixels.

### 2. Partial Cross Entropy

Loss is computed only on labeled pixels:

* Unlabeled pixels are ignored
* Enables training with incomplete supervision

### 3. Models

* **U-Net** with pretrained encoder
* **SegFormer** for global context modeling

---

## 🧪 Experiments

We evaluate:

### 🔹 Effect of label sparsity

* 100 points per image
* 200 points per image

### 🔹 Model comparison

* U-Net vs SegFormer

---

## 📈 Results

| Model     | Points | mIoU       |
| --------- | ------ | ---------- |
| U-Net     | 100    | 0.0545     |
| U-Net     | 200    | 0.0483     |
| SegFormer | 100    | **0.1045** |
| SegFormer | 200    | 0.0927     |

---

## 🔍 Key Insights

* Transformer-based models outperform CNNs under sparse supervision
* Increasing labeled points does not always improve performance
* Class imbalance significantly affects results
* Sparse supervision introduces training instability

---

## 🏗️ Project Structure

```
LandVisor_Point_Supervised_Segmentation_project/
│
├── configs/
│   ├── model/
│   │   └── model_config.yaml
│   └── base_config.yaml
├── data/
├── src/
├── tests/
├── scripts/
├── report/
├── main.py
└── README.md
```

---

## 🛠️ Technical Stack
* **Core:** PyTorch, Torchvision
* **Models:** Segmentation Models PyTorch (SMP), HuggingFace Transformers
* **Augmentations:** Albumentations
* **Visualization:** Matplotlib, OpenCV

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/yourusername/LandVisor.git](https://github.com/yourusername/LandVisor.git)
cd LandVisor
pip install -r requirements.txt
```
### 2. Data Preparation
Place your DeepGlobe images in data/train/images and masks in data/train/masks. Run the split script:

```bash
python scripts/prepare_data.py
```

###  3. Training
To train the model:

```bash
python scripts/train.py 
```

###  4. evaluating
To evaluate the model:

```bash
python scripts/evaluate.py 
```

### Engineering Insights
1. Global vs. Local Receptive Fields
CNNs (U-Net) failed to bridge the gap between sparse points due to their local receptive fields. SegFormer’s self-attention allowed the model to correlate distant pixels with identical spectral signatures, effectively "healing" the gaps between points.

2. The Semi-Supervised Advantage
By applying a consistency constraint on unlabeled pixels, the model learned spatial hierarchies (like the shape of a city or the flow of a river) that were not explicitly labeled, proving that SSL is a requirement for point-supervised tasks.

---

## 📊 Evaluation

* Metric: **Mean Intersection over Union (mIoU)**
* Evaluation is performed on **full ground truth masks**

---

## 📌 Future Improvements

* Boundary-aware point sampling
* Dice loss integration
* Longer training schedules
* Class-balanced loss
---

## 💡 Why This Project Matters

This project reflects real-world challenges:

* Limited labeled data
* Need for scalable annotation strategies
* Robust model design under constraints

---

## 👨‍💻 Author

**Sandy Hedia**

AI Engineer (Computer Vision & NLP)

