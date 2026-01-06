
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from pathlib import Path

class RiskClassifier:
    def __init__(self, model_path: str = None, model_type: str = "xgboost", model_instance=None):
        if model_instance:
             self.model = model_instance
        elif model_type == "xgboost":
            self.model = XGBClassifier()
        elif model_type == "mlp":
            self.model = MLPClassifier()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, X_train, y_train, **kwargs):
        """
        Train the classifier.
        kwargs can be passed to the underlying model's set_params or fit method.
        """
        # Distinguish between fit params and model init params if necessary
        # For simple usage, we assume kwargs are for set_params
        try:
            self.model.set_params(**kwargs)
        except Exception as e:
            print(f"Warning: Could not set params: {e}")
        
        self.model.fit(X_train, y_train)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 4:
            # If features are (N, C, H, W) and need pooling
            features = features.mean(axis=(2, 3))
        
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 4:
            features = features.mean(axis=(2, 3))
        
        # Check if the model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[:, 1] # Return probability of positive class
        else:
            raise NotImplementedError("Model does not support predict_proba")

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
