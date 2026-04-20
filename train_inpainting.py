import argparse
import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Allow imports from subdirectories
sys.path.insert(0, os.path.dirname(__file__))
from models.inpainting_unet import InpaintingUNet
from models.inpainting_losses import InpaintingLoss
from data.inpainting_dataset import get_inpainting_dataloaders

# Reuse PSNR/SSIM from denoising project
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from evaluate import evaluate_batch
except ImportError:
    # Fallback inline PSNR
    import math, torch.nn.functional as F
    def evaluate_batch(pred, target):
        mse = F.mse_loss(pred.clamp(0,1), target.clamp(0,1)).item()
        psnr = 10 * math.log10(1.0 / mse) if mse > 0 else 100.0
        return {'psnr': psnr, 'ssim': 0.0}


def parse_args():
    p = argparse.ArgumentParser(description = "Train Inpainting U-Net")
    p.add_argument('--train_dir', required = True)
    p.add_argument('--val_dir', required = True)
    p.add_argument('--output_dir', default = 'checkpoints_inpainting')
    p.add_argument('--epochs', type=int, default = 100)
    p.add_argument('--batch_size', type=int, default = 8)
    p.add_argument('--patch_size', type=int, default = 256)
    p.add_argument('--lr', type=float, default = 2e-4)
    p.add_argument('--grayscale', action='store_true', default = True)
    p.add_argument('--no_perceptual', action='store_true', help = 'Disable perceptual loss (faster but lower quality)')
    p.add_argument('--resume', default = None)
    p.add_argument('--num_workers', type = int, default = 4)
    return p.parse_args()

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok = True)
    torch.save(state, path)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (masked_img, mask, clean) in enumerate(loader):
        masked_img = masked_img.to(device)
        mask = mask.to(device)
        clean = clean.to(device)
        optimizer.zero_grad()
        output = model(masked_img, mask)
        loss = criterion(output, clean, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "f"Loss: {loss.item():.5f} | {elapsed:.1f}s")

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = total_ssim = 0.0

    for masked_img, mask, clean in loader:
        masked_img = masked_img.to(device)
        mask = mask.to(device)
        clean = clean.to(device)
        output = model(masked_img, mask)
        metrics = evaluate_batch(output, clean)
        total_psnr += metrics['psnr']
        total_ssim += metrics['ssim']

    n = len(loader)
    return {'psnr': total_psnr / n, 'ssim': total_ssim / n}

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    train_loader, val_loader = get_inpainting_dataloaders(args.train_dir, args.val_dir, batch_size = args.batch_size, patch_size = args.patch_size, num_workers = args.num_workers)
    channels = 1 if args.grayscale else 3
    model = InpaintingUNet(in_channels = channels, out_channels = channels).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6)
    criterion = InpaintingLoss(use_perceptual = not args.no_perceptual)
    start_epoch, best_psnr = 0, 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location = 'cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
    print(f"\n{'='*60}")
    print(f"Training inpainting model for {args.epochs} epochs")
    print(f"Perceptual loss: {'OFF' if args.no_perceptual else 'ON'}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "f"Loss: {train_loss:.5f} | "f"PSNR: {val_metrics['psnr']:.2f} dB | "f"SSIM: {val_metrics['ssim']:.4f}")

        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr}, os.path.join(args.output_dir, 'best_inpainting_model.pth'))
            print(f" Best model saved (PSNR: {best_psnr:.2f} dB)")

        if epoch % 10 == 0:
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_psnr': best_psnr}, os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth'))

    with open(os.path.join(args.output_dir, 'history_inpainting.json'), 'w') as f:
        json.dump(history, f, indent = 2)

    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB")

if __name__ == '__main__':
    main()