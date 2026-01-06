
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

def is_roi_image(image_path: str, black_ratio_threshold: float = 0.8) -> bool:
    """
    Check if an image is a Region of Interest (ROI) based on black pixel ratio.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    unique_vals = np.unique(binary)
    if not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [0, 255]):
        return False

    total_pixels = binary.size
    black_pixels = np.sum(binary == 0)
    black_ratio = black_pixels / total_pixels

    return black_ratio >= black_ratio_threshold


