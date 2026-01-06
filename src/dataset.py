
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from monai.transforms import Compose, Resize, ScaleIntensity
from pathlib import Path
from typing import List, Tuple

class MammogramDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_filenames = [
            f for f in sorted(os.listdir(image_dir)) 
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        if transform:
            self.transform = transform
        else:
            # Default transform mimicking the notebook
            self.transform = Compose([
                Resize((224, 224)),
                ScaleIntensity(),
            ])

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> Tuple[object, str]:
        img_name = self.image_filenames[idx]
        img_path = self.image_dir / img_name
        
        try:
            # Check desired format based on usage context ideally, 
            # but here we allow raw PIL return if self.transform is explicitly False/None for some use cases
            # However, standard PyTorch datasets usually return Tensors.
            # We will default to RGB if using something like CLIP.
            
            img_pil = Image.open(img_path)
            
            if self.transform:
                # If using MONAI transforms, usually wants channel-first numpy/tensor
                img = img_pil.convert("L")
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                img = self.transform(img)
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img)
                return img.float(), img_name
            else:
                # Return raw PIL image (useful for HuggingFace processors)
                img_pil = img_pil.convert("RGB") # CLIP models usually expect RGB
                return img_pil, img_name
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle error appropriately. 
            # For strict training, raising error is better.
            raise e
