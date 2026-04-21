# Artifact Restoration using Neural Networks

> NOTES:
> The purpose of this project is to use existing literature to learn
> machine learning image algorithms and techniques.
> 

# Google Colab
It is recommended to run the project via [Google Colab](https://colab.research.google.com/drive/1yqeMhJ31ITy2BF0HT7OHinNFfQFOTX-f?usp=sharing) as it will result in faster training times. Just make sure you follow the instructions at the top of the notebook to set the proper hardware and add your GitHub token (necessary to clone this private repository).

# Setup
1. Open CMD within the project root and run:
```bash
pip install -r requirements.txt
```
  This will install all required python components for the project.

2. Download the trained denoising model from [here](https://drive.google.com/file/d/1feZ_9RPZIrjEvDFaOQiKefmfGDQrI7Np/view?usp=sharing). If you would like to train your own model, run the command below (warning: this can take up to a day based on how powerful your CPU is unless trained via Google Collab).
```bash
!python train.py --train_dir data/images/train --val_dir data/images/test --noise_sigma 25 --epochs 100 --batch_size 8 --patch_size 256 --num_workers 2 --output_dir checkpoints
```
3. To test/evaluate the denoising model, run this command:
```bash
!python infer.py --model checkpoints/best_model.pth --input data/images/test --add_noise 25 --output results_denoising --visualize
```
   This will denoise the images (in black and white) and show a comparison for one of the images.

4. Train an inpainting model by running the command below (warning: this can take up to a day based on how powerful your CPU is unless trained via Google Colab).
```bash
!python train_inpainting.py --train_dir data/images/train --val_dir data/images/test --epochs 100 --batch_size 8 --patch_size 256 --num_workers 2 --output_dir checkpoints_inpainting
```
5. To test/evaluate the denoising model, run this command:
```bash
!python infer_inpainting.py --model checkpoints_inpainting/best_inpainting_model.pth --input data/images/test --output results_inpainting --visualize
```

6. To get a comparison with both models apply and how the affect images, run the following script (already a cell in our Google Colab notebook):
```bash
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import sys, os

sys.path.insert(0, '.')
sys.path.insert(0, 'inpainting')
sys.path.insert(0, 'inpainting/data')

from models.unet import UNet
from models.inpainting_unet import InpaintingUNet
from data.mask_generator import combined_mask, mask_to_tensor, apply_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load both models
denoiser = UNet(in_channels = 1, out_channels = 1).to(device)
inpainter = InpaintingUNet(in_channels = 1, out_channels = 1).to(device)

denoiser.load_state_dict(torch.load('checkpoints/best_model.pth', map_location = device, weights_only = True)['model'])
inpainter.load_state_dict(torch.load('checkpoints_inpainting/best_inpainting_model.pth', map_location = device, weights_only = True)['model'])

denoiser.eval()
inpainter.eval()

# Pick a test image
test_images = [f for f in os.listdir('data/images/test') if f.endswith('.jpg')]
img_path = os.path.join('data/images/test', test_images[0])
clean = TF.to_tensor(Image.open(img_path).convert('L'))
_, h, w = clean.shape

# Denoising: add noise
noisy = (clean + torch.randn_like(clean) * (25/255)).clamp(0, 1)

# Inpainting: apply mask
mask_pil = combined_mask(h, w)
mask = mask_to_tensor(mask_pil)
masked = apply_mask(clean, mask, fill_value=0.0)

with torch.no_grad():
    denoised = denoiser(noisy.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
    inpainted = inpainter(masked.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)

# Plot all 5 panels
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
panels = [
    (clean, 'Original Clean'),
    (noisy, 'Noisy Input (σ=25)'),
    (denoised, 'Denoised Output'),
    (masked, 'Masked Input'),
    (inpainted, 'Inpainted Output'),
]

for ax, (img, title) in zip(axes, panels):
    ax.imshow(TF.to_pil_image(img), cmap = 'gray')
    ax.set_title(title, fontsize = 11, fontweight = 'bold')
    ax.axis('off')

plt.suptitle('Artifact Restoration: Denoising & Inpainting Results', fontsize = 13)
plt.tight_layout()
plt.savefig('combined_results.png', dpi = 150, bbox_inches = 'tight')
plt.show()
print('Saved combined_results.png')
```
