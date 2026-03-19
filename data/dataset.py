import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class DenoisingDataset(Dataset):
    def __init__(self, image_dir, patch_size = 256, noise_sigma = 25, augment = True, grayscale = True):
        super(DenoisingDataset, self).__init__()
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma
        self.augment = augment
        self.grayscale = grayscale

        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(exts)
        ]
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in '{image_dir}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx])

        if self.grayscale:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img = TF.to_tensor(img)

        # Random crop
        if self.patch_size is not None:
            img = self._random_crop(img, self.patch_size)

        # Augmentation
        if self.augment:
            img = self._augment(img)

        # Synthesize noise
        sigma = self.noise_sigma if self.noise_sigma else random.uniform(10, 75)
        noise = torch.randn_like(img) * (sigma / 255.0)
        noisy = torch.clamp(img + noise, 0.0, 1.0)

        return noisy, img   # (input, target)

    def _random_crop(self, img, size):
        c, h, w = img.shape

        if h < size or w < size:
            img = TF.resize(img, (max(h, size), max(w, size)))
            c, h, w = img.shape
        
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


def get_dataloaders(train_dir, val_dir, batch_size = 16, patch_size = 256, noise_sigma = 25, num_workers = 4):
    train_ds = DenoisingDataset(train_dir, patch_size = patch_size, noise_sigma = noise_sigma, augment = True)
    val_ds = DenoisingDataset(val_dir, patch_size = None, noise_sigma = noise_sigma, augment = False)
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    val_loader = DataLoader(val_ds, batch_size = 1, shuffle = False, num_workers = num_workers)
    return train_loader, val_loader

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <image_folder>")
        sys.exit(1)

    ds = DenoisingDataset(sys.argv[1], patch_size=256, noise_sigma=25)
    noisy, clean = ds[0]
    print(f"Dataset size: {len(ds)}")
    print(f"Noisy shape: {noisy.shape}  min = {noisy.min():.3f}  max = {noisy.max():.3f}")
    print(f"Clean shape: {clean.shape}  min = {clean.min():.3f}  max = {clean.max():.3f}")