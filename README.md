
# 🎗️ Calcification Risk Classification Framework

> **A Hybrid Deep Learning System combining DenseNet121 & XGBoost for High-Performance Mammography Analysis.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20MONAI-orange?style=flat&logo=pytorch&logoColor=white)](https://monai.io/)
[![API](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📖 Introduction

This project implements a production-ready pipeline for assessing calcification risks in mammography images. Unlike traditional end-to-end Deep Learning approaches, we utilize a **Hybrid Architecture**:

1.  **Feature Extractor:** Uses a frozen **DenseNet121** (pretrained on ImageNet/adapted via MONAI) to extract high-level visual embeddings.
2.  **Classifier:** A lightweight **XGBoost** model performs the final classification, offering superior performance on tabular representations of image features.

This approach was designed to balance **high accuracy** with **low computational cost**, suitable for deployment in resource-constrained environments (Edge/Fog nodes).

## 📊 Key Results & Performance

Based on the CBIS-DDSM dataset, the framework achieves state-of-the-art efficiency compared to full fine-tuning methods:

| Metric | Performance | Notes |
| :--- | :--- | :--- |
| **Sensitivity** | **93.6%** | Critical for minimizing false negatives in medical diagnosis |
| **Precision** | **86.1%** | Balanced false positive rate |
| **Inference Time**| **< 150ms** | Per image latency (via FastAPI) |
| **Training Time** | **< 2 mins** | Drastic reduction vs. hours for end-to-end CNN training |

📄 *For a detailed methodology, hypothesis, and ablation study, please refer to the [**Technical Paper**](docs/paper.pdf).*

## 🏗️ System Architecture

<img width="1109" height="627" alt="image" src="https://github.com/user-attachments/assets/69c2490f-42e9-430b-9651-88d0948ced57" />


## 🗄️ Project Structure
```
/
├── api/
│   └── main.py            # FastAPI application entry point
├── src/
│   ├── classifier.py      # Classifier wrappers (XGBoost, MLP)
│   ├── config.py          # Configuration and constants
│   ├── dataset.py         # Dataset class (BiomedCLIP/MONAI support)
│   ├── evaluation.py      # Evaluation metrics and CV logic
│   ├── feature_extractor.py # Feature extractor (BiomedCLIP/DenseNet)
│   ├── generate_features.py # CLI for feature extraction
│   ├── inference.py       # Inference pipeline service
│   ├── run_evaluation.py  # CLI for model evaluation
│   ├── train.py           # CLI for training final model
│   └── utils.py           # Utility functions
├── data/                  # Data storage
├── models/                # Saved models storage
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## ⚙️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/risk-classify-mammography.git
    cd risk-classify-mammography
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Running the API
To start the web server:

```bash
uvicorn api.main:app --reload
```
The API will be available at `http://localhost:8000`.
Access the interactive documentation at `http://localhost:8000/docs`.

### Pipeline (CLI)
The project now uses command-line scripts for the ML pipeline:

1.  **Generate Features**:
    Extract features from your image directory (supports BiomedCLIP, DenseNet).
    ```bash
    python -m src.generate_features --data_dir "path/to/images" --output_name "features_biomedclip"
    ```

2.  **Evaluate Performance (Cross-Validation)**:
    Run KFold or LOGO to check model accuracy metrics.
    ```bash
    python -m src.run_evaluation --features "data/features/features_biomedclip.npy" --labels "data/features/labels.npy" --model xgboost --cv kfold
    ```

3.  **Train Final Model**:
    Train the model on the full dataset and save it for the API.
    ```bash
    python -m src.train --features "data/features/features_biomedclip.npy" --labels "data/features/labels.npy" --output "models/xgb_model.pkl"
    ```

## 🛠️ Components

- **Feature Extractor**: Uses `densenet121` from MONAI, identifying patterns in mammograms.
- **Classifier**: XGBoost model optimized for binary/multi-class risk classification.
- **API**: Built with FastAPI for high performance and auto-generated Swagger UI.

## 📝 Notes
- Ensure you have the trained XGBoost model placed in `models/xgb_model.pkl` for the inference API to work fully.
- Dataset is available at [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data).

## 📫 Contact
- Email: phuong.tranolive@hcmut.edu.vn
- GitHub: [phuongolive](https://github.com/phuongolive)
