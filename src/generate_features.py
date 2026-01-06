
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from .feature_extractor import FeatureExtractor
from .config import DATA_DIR, FEATURES_DIR, FEATURE_EXTRACTOR_NAME

def extract_features(data_dir: str, output_name: str = "features"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    extractor = FeatureExtractor(model_name=FEATURE_EXTRACTOR_NAME, device=device)
    
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    
    # Recursively find images or just flat directory? 
    # Notebook did flat directory. Let's assume flat for now or walk.
    data_path = Path(data_dir)
    image_files = [
        f for f in sorted(os.listdir(data_path)) 
        if f.lower().endswith(tuple(image_extensions))
    ]
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return

    features_list = []
    filenames = []
    
    print(f"Extracting features from {len(image_files)} images...")
    
    for fname in tqdm(image_files):
        img_path = data_path / fname
        try:
            # We open as RGB for BiomedCLIP or let extractor handle it?
            # FeatureExtractor expects PIL for biomedclip, or Tensor for dense.
            # Our dataset.py logic suggests we should stick to a standard. 
            # Here we just open as PIL and let FeatureExtractor handle internal logic.
            img = Image.open(img_path).convert("RGB")
            
            # Extract expects batch, so we might need to modify FeatureExtractor or just pass one
            # FeatureExtractor.extract expects 'images' which usually is batch.
            # But the processor can handle list of images.
            
            feat = extractor.extract(images=[img]) # Pass as list to emulate batch of 1
            features_list.append(feat.flatten()) # Flatten if (1, D) -> (D,)
            filenames.append(fname)
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            
    features = np.array(features_list)
    
    output_path = FEATURES_DIR / f"{output_name}.npy"
    names_path = FEATURES_DIR / f"{output_name}_filenames.npy"
    
    np.save(output_path, features)
    np.save(names_path, np.array(filenames))
    
    print(f"Saved features shape {features.shape} to {output_path}")
    print(f"Saved filenames to {names_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Features using Pretrained Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_name", type=str, default="features", help="Base name for output files")
    
    args = parser.parse_args()
    
    extract_features(args.data_dir, args.output_name)

if __name__ == "__main__":
    main()
