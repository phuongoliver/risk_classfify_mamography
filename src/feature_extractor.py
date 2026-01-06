
import torch
import torch.nn as nn
from monai.networks.nets import densenet121
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self, model_name: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_type = "monai_densenet" 
        
        if self.model_name and "biomedclip" in self.model_name.lower():
            self.model_type = "biomedclip"
            
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
        
        if self.model_type == "biomedclip":
             self.processor = CLIPImageProcessor.from_pretrained(self.model_name)

    def _build_model(self):
        if self.model_type == "biomedclip":
            print(f"Loading BiomedCLIP model: {self.model_name}")
            try:
                model = CLIPVisionModel.from_pretrained(self.model_name)
                return model
            except Exception as e:
                print(f"Error loading BiomedCLIP model: {e}")
                print("Falling back to DenseNet121...")
                self.model_type = "monai_densenet"
                
        # Default/Fallback to DenseNet121
        # Load MONAI DenseNet121 pretrained on medical images
        # spatial_dims=2 for 2D images
        model = densenet121(spatial_dims=2, in_channels=1, out_channels=2, pretrained=True)
        
        # Remove final classifier layer to get features
        # DenseNet121 from monai usually has 'class_layers' as the final block
        model.class_layers = nn.Identity()
        
        return model

    def extract(self, images) -> np.ndarray:
        """
        Extract features from a batch of images.
        Args:
            images: 
                - If BiomedCLIP: PIL Images or list of PIL Images, or pre-processed tensors
                - If DenseNet: torch.Tensor of shape (B, C, H, W)
        Returns:
            np.ndarray: Features of shape (B, Features)
        """
        
        if self.model_type == "biomedclip":
            # Handle PIL inputs regarding processor
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.pooler_output
            return features.cpu().numpy()
            
        else:
            # DenseNet expects tensors
            if not isinstance(images, torch.Tensor):
                 raise ValueError("DenseNet extractor expects input of type torch.Tensor")
                 
            images = images.to(self.device)
            with torch.no_grad():
                features = self.model(images)
            return features.cpu().numpy()
