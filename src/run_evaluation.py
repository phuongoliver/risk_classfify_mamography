
import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from .evaluation import evaluation
from .config import MODELS_DIR, FEATURES_DIR

def main():
    parser = argparse.ArgumentParser(description="Run Risk Classification Evaluation")
    parser.add_argument("--features", type=str, default=str(FEATURES_DIR / "features_swin.npy"), help="Path to features file")
    parser.add_argument("--labels", type=str, default=str(FEATURES_DIR / "labels.npy"), help="Path to labels file")
    parser.add_argument("--groups", type=str, default=None, help="Path to groups file (patient IDs)")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "xgboost"], help="Model type")
    parser.add_argument("--cv", type=str, default="kfold", choices=["kfold", "logo"], help="CV strategy")
    
    args = parser.parse_args()
    
    # Setup Model
    if args.model == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128,), random_state=42, max_iter=500)
    elif args.model == "xgboost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
    # Setup CV
    if args.cv == "logo":
        if args.groups is None:
            raise ValueError("LeaveOneGroupOut requires --groups argument")
        cv = LeaveOneGroupOut()
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
    # Verify files exist
    if not os.path.exists(args.features):
        print(f"Features file not found: {args.features}")
        return
    if not os.path.exists(args.labels):
        print(f"Labels file not found: {args.labels}")
        return
        
    print(f"Running evaluation with {args.model} using {args.cv}...")
    
    results = evaluation(
        model=model,
        features_path=args.features,
        labels_path=args.labels,
        groups_path=args.groups,
        cv_strategy=cv,
        label_name="Risk Class"
    )
    
    # Save results
    output_path = MODELS_DIR / f"evaluation_{args.model}_{args.cv}.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
