# Hyperspectral Image Anomaly Detection
A Python implementation comparing multiple machine learning approaches for detecting anomalies in hyperspectral imagery using synthetic data.

## Overview
This project demonstrates three different anomaly detection techniques applied to hyperspectral data:
- **One-Class SVM** (Unsupervised)
- **Isolation Forest** (Unsupervised)
- **Random Forest Classifier** (Supervised)

Hyperspectral imaging captures data across hundreds of spectral bands, making it valuable for applications like environmental monitoring, mineral exploration, and target detection.

## Features
- Synthetic hyperspectral data generation with realistic spectral signatures
- Dimensionality reduction using PCA
- Three anomaly detection algorithms with performance comparison
- Comprehensive visualization of results
- Confusion matrices and classification metrics

## Requirements
```bash
scikit-learn
numpy
matplotlib
seaborn
```

Install dependencies:
```bash
pip install scikit-learn numpy matplotlib seaborn
```

The script will:
1. Generate synthetic hyperspectral data (5000 samples, 200 spectral bands)
2. Preprocess and split data into training/test sets
3. Apply PCA for dimensionality reduction
4. Train and evaluate three anomaly detection models
5. Generate visualizations and save to `hyperspectral.png`
6. Print comprehensive performance metrics

## Data Description
**Synthetic Dataset:**
- 5000 total samples
- 200 spectral bands per sample
- 95% normal samples (background)
- 5% anomalous samples (targets with distinctive spectral peaks)
- Train/test split: 70/30

## Models
### 1. One-Class SVM
- Unsupervised learning on normal samples only
- RBF kernel with auto gamma
- Nu parameter: 0.1

### 2. Isolation Forest
- Unsupervised anomaly detection
- Contamination rate: 0.05
- Random state: 42

### 3. Random Forest Classifier
- Supervised learning with labeled data
- 100 estimators
- Max depth: 10

## Output
The script generates:
- **Console output:** Classification reports and performance metrics
- **Visualization:** 6-panel figure including:
  - Sample spectral signatures
  - PCA visualization (ground truth)
  - Model predictions in PCA space
  - Confusion matrices for all three models

## Performance Metrics
Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Key Findings
- Random Forest achieves the best performance with labeled training data
- Isolation Forest is effective for initial exploration without labels
- One-Class SVM provides robust unsupervised detection
- PCA reduces computational cost while retaining 95% of variance

## Recommendations
- **Use Random Forest** when you have labeled training data
- **Use Isolation Forest** for initial exploration without labels
- **Apply PCA** to reduce computational cost in high-dimensional data
- **Consider ensemble methods** combining multiple approaches for production systems

## File Structure
```
.
├── hyperspectral.py          # Main script
├── hyperspectral.png          # Generated visualization
└── README.md                  # This file
```
- Real hyperspectral data can be obtained from sources like AVIRIS, Hyperion, or PRISMA
- The synthetic data generator can be modified to match specific spectral characteristics
- Model hyperparameters should be tuned for specific applications
