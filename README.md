
# 🎗️ Calcification Risk Classification API

A professional deep learning framework for classifying breast cancer risk from mammography images, exposing a robust REST API.

## 📚 Project Overview
This project processes mammogram images to detect and classify calcification risks. It utilizes a hybrid approach:
1.  **Feature Extraction**: Using a pre-trained **DenseNet121** (via MONAI).
2.  **Classification**: Using **XGBoost** for high-performance tabular classification on extracted features.

The project has been restructured for production-grade deployment with a FastAPI backend.

## � Research Paper
For a detailed explanation of the methodology, hypothesis, and results, please refer to the [Technical Paper](docs/paper.pdf).

## �🗄️ Project Structure
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
