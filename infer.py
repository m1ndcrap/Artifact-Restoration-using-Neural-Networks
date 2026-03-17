import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from models.unet import UNet
from evaluate import evaluate_batch

def load_model(checkpoint_path, device, grayscale=True):
    channels = 1 if grayscale else 3
    model = UNet(in_channels = channels, out_channels = channels).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"Loaded model from '{checkpoint_path}' "
          f"(trained to epoch {ckpt.get('epoch','?')}, "
          f"best PSNR {ckpt.get('best_psnr', 0):.2f} dB)")
    
    return model

def process_image(model, img_path, device, grayscale = True, add_noise_sigma = None, save_path = None):
    img = Image.open(img_path)
    img = img.convert('L') if grayscale else img.convert('RGB')
    tensor = TF.to_tensor(img)

    if add_noise_sigma is not None:
        noise = torch.randn_like(tensor) * (add_noise_sigma / 255.0)
        noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
        clean_tensor = tensor
    else:
        noisy_tensor = tensor
        clean_tensor = None

    # Run model
    with torch.no_grad():
        inp = noisy_tensor.unsqueeze(0).to(device)
        out = model(inp).squeeze(0).cpu().clamp(0.0, 1.0)

    # Compute metrics if we have ground truth
    metrics = None

    if clean_tensor is not None:
        metrics = evaluate_batch(
            out.unsqueeze(0), clean_tensor.unsqueeze(0))

    # Convert to PIL images
    noisy_pil = TF.to_pil_image(noisy_tensor)
    denoised_pil = TF.to_pil_image(out)

    # Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok = True)
        denoised_pil.save(save_path)

    return noisy_pil, denoised_pil, metrics


def visualize_result(noisy, denoised, clean = None, title = ""):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping visualization.")
        return

    imgs = [noisy, denoised]
    labels = ['Noisy Input', 'Denoised Output']

    if clean is not None:
        imgs.append(clean)
        labels.append('Ground Truth')

    fig, axes = plt.subplots(1, len(imgs), figsize = (5 * len(imgs), 5))

    for ax, img, label in zip(axes, imgs, labels):
        ax.imshow(img, cmap='gray' if img.mode == 'L' else None)
        ax.set_title(label)
        ax.axis('off')

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig('result_preview.png', dpi = 150, bbox_inches = 'tight')
    plt.show()
    print("Saved preview to result_preview.png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required = True, help = 'Path to .pth checkpoint')
    p.add_argument('--input', required = True, help = 'Input image or folder')
    p.add_argument('--output', default = 'results', help = 'Output folder')
    p.add_argument('--add_noise', type = float, default = None, help = 'Add Gaussian noise of given sigma before denoising')
    p.add_argument('--grayscale', action = 'store_true', default = True)
    p.add_argument('--visualize', action = 'store_true', help = 'Show matplotlib preview for first image')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device, grayscale=args.grayscale)

    # Collect image paths
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
        save_path = os.path.join(args.output, f"{fname}_denoised.png")

        noisy, denoised, metrics = process_image(
            model, path, device,
            grayscale = args.grayscale,
            add_noise_sigma = args.add_noise,
            save_path = save_path
        )

        if metrics:
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
            print(f"[{i+1}/{len(image_paths)}] {fname}: "
                  f"PSNR = {metrics['psnr']:.2f} dB  SSIM = {metrics['ssim']:.4f}")
        else:
            print(f"[{i+1}/{len(image_paths)}] {fname}: saved to {save_path}")

        if args.visualize and i == 0:
            visualize_result(noisy, denoised, title=fname)

    if all_psnr:
        print(f"\n{'='*40}")
        print(f"Average PSNR: {sum(all_psnr)/len(all_psnr):.2f} dB")
        print(f"Average SSIM: {sum(all_ssim)/len(all_ssim):.4f}")


if __name__ == '__main__':
    main()