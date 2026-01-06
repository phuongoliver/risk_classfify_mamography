
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image processing
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 4  # Adjust based on system

# Model configuration
FEATURE_EXTRACTOR_NAME = "microsoft/BiomedCLIP-PubMedCLIP-ViT-B-16"
CLASSIFIER_NAME = "xgb_classifier"
use_cuda = True  # Can be made dynamic
