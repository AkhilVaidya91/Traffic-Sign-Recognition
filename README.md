# Traffic Sign Recognition System with Hybrid CNN and Random Forest Models for Autonomous Driving Applications

[![IEEE Publication](https://img.shields.io/badge/IEEE-Published-blue)](https://ieeexplore.ieee.org/document/10795884)

This repository contains the official implementation for our paper ["Traffic Sign Recognition System with Hybrid CNN and Random Forest Models for Autonomous Driving Applications"](https://ieeexplore.ieee.org/document/10795884) published at the 2024 First International Conference on Software, Systems and Information Technology (SSITCON).

## Abstract
This research presents a novel hybrid approach for traffic sign recognition by combining Convolutional Neural Networks (CNN) and Random Forest models. Our system achieves state-of-the-art performance for autonomous driving applications with a two-stage pipeline: detection using YOLOv8 followed by classification using VGG16 embeddings with Random Forest.

## Architecture
![Traffic Sign Detection and Classification Pipeline](path/to/architecture_image.png)

Our system employs a multi-pathway architecture:
1. YOLOv8 for initial traffic sign detection
2. VGG16 for deep feature extraction/embedding
3. Random Forest classifier for final classification
4. Special handling for speed limit signs using Hough Circle Transform

## Key Results

### Detection Results (Table I)
| Model    | Precision | Recall | mAP50  | mAP50-95 |
|----------|-----------|--------|--------|----------|
| YOLOv8m  | 0.98264   | 1      | 0.99348| 0.86228  |
| YOLOv5m  | 0.827     | 0.748  | 0.828  | 0.633    |

### Classification Results (Table II)
| Model        | Accuracy | Precision | Recall  | F1 Score | ROC-AUC Score |
|--------------|----------|-----------|---------|----------|---------------|
| SGD          | 97.21%   | 97.35%    | 97.19%  | 97.16%   | 99.98%        |
| SVM          | 93.35%   | 94.06%    | 93.37%  | 93.61%   | 99.85%        |
| KNN          | 97.54%   | 97.52%    | 97.49%  | 97.49%   | 99.81%        |
| Random Forest| 98.34%   | 98.65%    | 98.60%  | 98.61%   | 99.97%        |
| ANN          | 98.20%   | 97.91%    | 97.84%  | 97.85%   | 99.96%        |
| CNN          | 97.47%   | 97.63%    | 97.59%  | 97.60%   | 99.87%        |

## Features
- Demo application for real-time traffic sign detection and classification
- Exploratory Data Analysis (EDA) of traffic sign datasets
- Data preprocessing and augmentation
- Comparison of various embedding models
- 3D visualization of embedding clusters using t-SNE
- Evaluation and comparison of multiple classification models
- YOLOv8-based traffic sign detection
- Integration of Hough Circle Transform for speed limit sign detection
- Optical Character Recognition (OCR) using TrOCR for speed limit value extraction

## Project Structure
```
TSR/
│
├── data/
│   ├── classification/
│   └── detection-YOLO/
│
├── models/
│   ├── detection/
│   ├── classification/
│   └── embeddings/
│
├── notebooks/
|   ├── detection.ipynb
│   ├── EDA.ipynb
│   ├── data_preprocessing.ipynb
│   ├── embedding.ipynb
│   ├── classification_training.ipynb
│   ├── classification_evaluation.ipynb
│   └── CNN_CBAM_EXPERIMENTAL.ipynb
│
├── src/
|   ├── app.py
|   └── utils/
│
├── requirements.txt
│
└── README.md
```

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/AkhilVaidya91/Signumum-Revelio.git
   cd Signumum-Revelio
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the main application:
   ```
   streamlit run app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000` to access the landing page and demo application.
3. To explore the individual components of the project, refer to the Jupyter notebooks in the `notebooks/` directory.
