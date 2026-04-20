import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from mask_generator import combined_mask, mask_to_tensor, apply_mask

class InpaintingDataset(Dataset):
    def __init__(self, image_dir, patch_size = 256, augment = True, grayscale = True):
        super().__init__()
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.augment = augment
        self.grayscale  = grayscale

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(exts)
        ]
        if not self.image_paths:
            raise RuntimeError(f"No images found in '{image_dir}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load clean image
        img = Image.open(self.image_paths[idx])
        img = img.convert('L') if self.grayscale else img.convert('RGB')
        img = TF.to_tensor(img)

        # Random crop
        if self.patch_size:
            img = self._random_crop(img, self.patch_size)

        # Augmentation
        if self.augment:
            img = self._augment(img)

        c, h, w = img.shape

        # Generate random mask (fresh every step)
        mask_pil = combined_mask(h, w)
        mask = mask_to_tensor(mask_pil)

        # Apply mask to image (fill holes with 0)
        masked_img = apply_mask(img, mask, fill_value=0.0)

        return masked_img, mask, img # (input, mask, target)

    def _random_crop(self, img, size):
        _, h, w = img.shape

        if h < size or w < size:
            img = TF.resize(img, (max(h, size), max(w, size)))
            _, h, w = img.shape

        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        return img[:, top:top+size, left:left+size]

    def _augment(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)

        if random.random() > 0.5:
            img = TF.vflip(img)

        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=[1, 2])
        return img


def get_inpainting_dataloaders(train_dir, val_dir, batch_size = 8, patch_size = 256, num_workers = 4):
    train_ds = InpaintingDataset(train_dir, patch_size = patch_size, augment = True)
    val_ds = InpaintingDataset(val_dir, patch_size = None, augment = False)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = 1, shuffle = False, num_workers = num_workers)
    return train_loader, val_loader