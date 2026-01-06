
import argparse
import numpy as np
import os
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from .classifier import RiskClassifier
from .config import FEATURE_EXTRACTOR_NAME, FEATURES_DIR, MODELS_DIR

def train_model(features_path, labels_path, output_path, model_type="xgboost"):
    print(f"Loading data from {features_path}...")
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Error: Features or Labels file not found.")
        return

    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"Training {model_type} on {len(X)} samples...")
    
    # Select inner model
    if model_type == "xgboost":
        # Params matching those used in evaluation/notebooks usually
        inner_model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42
        )
    elif model_type == "mlp":
        inner_model = MLPClassifier(
            hidden_layer_sizes=(128,), 
            random_state=42, 
            max_iter=500
        )
    else:
        raise ValueError("Unsupported model type")

    # Wrap in our RiskClassifier structure
    classifier = RiskClassifier(model_instance=inner_model)
    classifier.train(X, y)
    
    # Save
    print(f"Saving model to {output_path}...")
    classifier.save(output_path)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Train and Save Final Model")
    parser.add_argument("--features", type=str, default=str(FEATURES_DIR / "features_swin.npy"), help="Path to features .npy file")
    parser.add_argument("--labels", type=str, default=str(FEATURES_DIR / "labels.npy"), help="Path to labels .npy file")
    parser.add_argument("--output", type=str, default=str(MODELS_DIR / "xgb_model.pkl"), help="Path to save model")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "mlp"], help="Model type")
    
    args = parser.parse_args()
    
    train_model(args.features, args.labels, args.output, args.model)

if __name__ == "__main__":
    main()
