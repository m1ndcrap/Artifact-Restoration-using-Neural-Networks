import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
from models.inpainting_unet import InpaintingUNet
from data.mask_generator import combined_mask, mask_to_tensor, apply_mask

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from evaluate import evaluate_batch
except ImportError:
    import math, torch.nn.functional as F
    def evaluate_batch(pred, target):
        mse = F.mse_loss(pred.clamp(0,1), target.clamp(0,1)).item()
        return {'psnr': 10 * math.log10(1.0 / (mse + 1e-8)), 'ssim': 0.0}

def load_model(path, device, grayscale = True):
    channels = 1 if grayscale else 3
    model = InpaintingUNet(in_channels = channels, out_channels = channels).to(device)
    ckpt = torch.load(path, map_location=device, weights_only = True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model (epoch {ckpt.get('epoch','?')}, "f"best PSNR {ckpt.get('best_psnr',0):.2f} dB)")
    return model

def inpaint_image(model, img_path, device, grayscale = True, mask_path = None, save_path = None):
    img = Image.open(img_path)
    img = img.convert('L') if grayscale else img.convert('RGB')
    clean = TF.to_tensor(img)
    _, h, w = clean.shape

    # Load or generate mask
    if mask_path:
        mask_pil = Image.open(mask_path).convert('L').resize((w, h))
    else:
        mask_pil = combined_mask(h, w)

    mask = mask_to_tensor(mask_pil)
    masked_img = apply_mask(clean, mask, fill_value = 0.0)

    # Run model
    with torch.no_grad():
        inp = masked_img.unsqueeze(0).to(device)
        msk = mask.unsqueeze(0).to(device)
        output = model(inp, msk).squeeze(0).cpu().clamp(0.0, 1.0)

    metrics = evaluate_batch(output.unsqueeze(0), clean.unsqueeze(0))

    masked_pil = TF.to_pil_image(masked_img.clamp(0, 1))
    inpainted_pil = TF.to_pil_image(output)
    clean_pil = TF.to_pil_image(clean)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok = True)
        inpainted_pil.save(save_path)

    return masked_pil, inpainted_pil, clean_pil, metrics

def visualize(masked, inpainted, clean, title = ""):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))

        for ax, img, label in zip(axes, [masked, inpainted, clean], ['Masked Input (damaged)', 'Inpainted Output', 'Ground Truth']):
            ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
            ax.set_title(label, fontsize = 13)
            ax.axis('off')

        if title:
            fig.suptitle(title, fontsize = 14)

        plt.tight_layout()
        plt.savefig('inpaint_preview.png', dpi = 150, bbox_inches = 'tight')
        plt.show()
        print("Saved inpaint_preview.png")
    except ImportError:
        print("skipping visualization since matplotlib is not installed.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required = True)
    p.add_argument('--input', required = True, help = 'Image or folder')
    p.add_argument('--mask', default = None, help = 'Optional mask image')
    p.add_argument('--output', default = 'results_inpainting')
    p.add_argument('--grayscale', action = 'store_true', default = True)
    p.add_argument('--visualize', action = 'store_true')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device, args.grayscale)

    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(exts)
        ]

    print(f"Processing {len(image_paths)} image(s)...\n")

    all_psnr, all_ssim = [], []

    for i, path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(path))[0]
        save_path = os.path.join(args.output, f"{fname}_inpainted.png")
        masked, inpainted, clean, metrics = inpaint_image(model, path, device, grayscale = args.grayscale, mask_path = args.mask, save_path = save_path)
        all_psnr.append(metrics['psnr'])
        all_ssim.append(metrics['ssim'])
        print(f"[{i+1}/{len(image_paths)}] {fname}: "f"PSNR = {metrics['psnr']:.2f} dB  SSIM = {metrics['ssim']:.4f}")

        if args.visualize and i == 0:
            visualize(masked, inpainted, clean, title = fname)

    if all_psnr:
        print(f"\n{'='*40}")
        print(f"Average PSNR: {sum(all_psnr)/len(all_psnr):.2f} dB")
        print(f"Average SSIM: {sum(all_ssim)/len(all_ssim):.4f}")


if __name__ == '__main__':
    main()