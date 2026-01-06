
import numpy as np
import torch
from PIL import Image
from monai.transforms import Compose, Resize, ScaleIntensity

from .feature_extractor import FeatureExtractor
from .classifier import RiskClassifier
from .config import MODELS_DIR, FEATURE_EXTRACTOR_NAME

class InferenceService:
    def __init__(self, classifier_path: str = None):
        if classifier_path is None:
            classifier_path = MODELS_DIR / "xgb_model.pkl"
            
        print("Initializing Feature Extractor...")
        self.feature_extractor = FeatureExtractor(model_name=FEATURE_EXTRACTOR_NAME)
        
        print("Initializing Classifier...")
        self.classifier = RiskClassifier(model_path=classifier_path)
        
        self.transform = Compose([
            Resize((224, 224)),
            ScaleIntensity(),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # (1, H, W)
        
        # Transform expects numpy or tensor
        img_transformed = self.transform(img_array)
        
        # Convert to tensor and add batch dim -> (1, 1, H, W)
        if not isinstance(img_transformed, torch.Tensor):
            img_tensor = torch.tensor(img_transformed)
        else:
            img_tensor = img_transformed
            
        return img_tensor.float().unsqueeze(0)

    def predict(self, image: Image.Image):
        # 1. Preprocess
        img_tensor = self.preprocess(image)
        
        # 2. Extract Features
        features = self.feature_extractor.extract(img_tensor)
        
        # 3. Classify
        # Note: Features might be (1, 1024, 7, 7) or similar depending on DenseNet layer
        # The classifier handles pooling if needed
        prediction = self.classifier.predict(features)
        probability = self.classifier.predict_proba(features)
        
        return {
            "prediction": int(prediction[0]),
            "risk_probability": float(probability[0])
        }
