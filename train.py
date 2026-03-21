import argparse
import os
import time
import json
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.unet import UNet
from models.losses import MSELoss
from data.dataset import get_dataloaders
from evaluate import evaluate_batch

def parse_args():
    p = argparse.ArgumentParser(description = "Train U-Net denoiser")
    p.add_argument('--train_dir', required = True, help = 'Path to training images')
    p.add_argument('--val_dir', required = True, help = 'Path to validation images')
    p.add_argument('--output_dir', default = 'checkpoints', help = 'Where to save models')
    p.add_argument('--epochs', type = int, default = 100)
    p.add_argument('--batch_size', type = int, default = 16)
    p.add_argument('--patch_size', type = int, default = 256)
    p.add_argument('--lr', type = float, default = 1e-3)
    p.add_argument('--noise_sigma', type = float, default = 25, help = 'Noise level (0-255). Use 0 for blind denoising (random 10-75).')
    p.add_argument('--grayscale', action = 'store_true', default = True)
    p.add_argument('--resume', default = None, help = 'Path to checkpoint to resume from')
    p.add_argument('--num_workers', type = int, default = 4)
    return p.parse_args()

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    torch.save(state, path)

def load_checkpoint(path, model, optimizer = None, scheduler = None):
    ckpt = torch.load(path, map_location = 'cpu')
    model.load_state_dict(ckpt['model'])

    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    
    return ckpt.get('epoch', 0), ckpt.get('best_psnr', 0.0)

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "
                  f"Loss: {loss.item():.6f} | {elapsed:.1f}s elapsed")

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        metrics = evaluate_batch(output, clean)
        total_psnr += metrics['psnr']
        total_ssim += metrics['ssim']

    n = len(loader)
    return {'psnr': total_psnr / n, 'ssim': total_ssim / n}

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    noise_sigma = args.noise_sigma if args.noise_sigma > 0 else None
    train_loader, val_loader = get_dataloaders(
        train_dir = args.train_dir,
        val_dir = args.val_dir,
        batch_size = args.batch_size,
        patch_size = args.patch_size,
        noise_sigma = noise_sigma,
        num_workers = args.num_workers,
    )

    # Model
    channels = 1 if args.grayscale else 3
    model = UNet(in_channels = channels, out_channels = channels).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6)
    criterion = MSELoss()

    # Resume from checkpoint if provided
    start_epoch = 0
    best_psnr = 0.0
    if args.resume:
        start_epoch, best_psnr = load_checkpoint(
            args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f} dB")

    # History log
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"Noise sigma: {noise_sigma if noise_sigma else 'blind (10-75)'}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)

        # Validate
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['val_psnr'].append(val_metrics['psnr'])
        history['val_ssim'].append(val_metrics['ssim'])
        print(f"\nEpoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
              f"Val SSIM: {val_metrics['ssim']:.4f}")

        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"* New best model saved (PSNR: {best_psnr:.2f} dB) *")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth'))

    # Save training history
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent = 2)

    print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")

if __name__ == '__main__':
    main()