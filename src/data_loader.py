import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from src.utils import apply_histogram_matching
import os
import glob
from albumentations.pytorch import ToTensorV2
import albumentations as A

class RemoteSensingDataset(Dataset):
    def __init__(self, image_dir, reference_path, transform=None, points_per_class=10,mask_dir=None):
        self.image_dir = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_dir = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        self.reference_path = reference_path # Path to your reference image
        self.points_per_class = points_per_class

        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_dir)

    def __getitem__(self, idx):
        # 1. Load Image
        image = cv2.imread(self.image_dir[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If we have masks, process them
        if len(self.mask_dir) > 0:
            rgb_mask = cv2.imread(self.mask_dir[idx])
            rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
            
            # Map RGB to 0-6 Indices
            full_mask = self.rgb_to_masks(rgb_mask)
            sparse_mask = self._generate_points(full_mask)
        else:
            # Inference Mode: Return dummy masks so the DataLoader doesn't break
            h, w = image.shape[:2]
            full_mask = np.full((h, w), -1, dtype=np.int64)
            sparse_mask = np.full((h, w), -1, dtype=np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=sparse_mask, full_mask=full_mask)
            image = augmented['image']
            sparse_mask = augmented['mask']
            full_mask = augmented['full_mask']
            
        else:
            # If no transform/ToTensorV2, manual conversion is needed:
            # HWC -> CHW and normalize to [0, 1]
            image = np.transpose(image, (2, 0, 1)) / 255.0

        # Safety: Ensure everything is a proper Torch Tensor with the right type
        image_tensor = torch.as_tensor(image).float()
        sparse_tensor = torch.as_tensor(sparse_mask).long()
        full_tensor = torch.as_tensor(full_mask).long()

        return image_tensor, sparse_tensor, full_tensor

    def rgb_to_masks(self,rgb_mask):
        # The palette
        palette = {
            (0, 255, 255): 0,   # urban_land
            (255, 255, 0): 1,   # agriculture_land
            (255, 0, 255): 2,   # rangeland
            (0, 255, 0): 3,     # forest_land
            (0, 0, 255): 4,     # water
            (255, 255, 255): 5, # barren_land
            (0, 0, 0): 6        # unknown
        }
        
        h, w = rgb_mask.shape[:2]
        mask = np.zeros((h, w), dtype=np.int64)
        
        for rgb, idx in palette.items():
            # Find pixels that match this RGB color
            match = np.all(rgb_mask == rgb, axis=-1)
            mask[match] = idx
            
        return mask
    
    def _generate_points(self, mask):
        # Ensure mask is a numpy array
        mask_np = np.array(mask)
        sparse = np.full(mask_np.shape, -1, dtype=np.int64) 
        
        classes = np.unique(mask_np)
        for cls in classes:
            if cls == -1: # Skip ignore index
                continue
                
            # Get coordinates where mask equals current class
            coords = np.argwhere(mask_np == cls) # Returns shape (N, 2)
            
            if len(coords) > 0:
                n_points = min(self.points_per_class, len(coords))
                # Randomly pick indices
                indices = np.random.choice(len(coords), n_points, replace=False)
                
                # Select the coordinate pairs
                chosen_coords = coords[indices]
                
                # Iterate and assign
                for point in chosen_coords:
                    r, c = point[0], point[1]
                    sparse[r, c] = cls
                    
        return sparse
    
    # ==================== DATA AUGMENTATION ====================
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AugmentationFactory:
    @staticmethod
    def get_train_transform(image_size):
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.9, 1.1),
                rotate=(-45, 45),
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ], p=0.3),
            ToTensorV2(),
        ]
        )

    @staticmethod
    def get_val_transform(image_size):
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            ToTensorV2(),
        ],
        additional_targets={'full_mask': 'mask'}
        )
    