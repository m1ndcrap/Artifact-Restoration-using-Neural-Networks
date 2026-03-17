import torch
import torch.nn.functional as F
import math

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(pred, target, reduction = 'mean').item()

    if mse == 0:
        return float('inf')
    
    return 10 * math.log10((max_val ** 2) / mse)

def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    channels = pred.shape[1]

    # Gaussian window
    def gaussian_kernel(size, sigma = 1.5):
        coords = torch.arange(size, dtype = torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel = g.outer(g)
        return kernel.unsqueeze(0).unsqueeze(0)

    kernel = gaussian_kernel(window_size).to(pred.device)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=channels)

    mu_x = conv(pred)
    mu_y = conv(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = conv(pred * pred)   - mu_x2
    sigma_y2 = conv(target * target) - mu_y2
    sigma_xy = conv(pred * target) - mu_xy
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (numerator / denominator).mean().item()

def evaluate_batch(pred: torch.Tensor, target: torch.Tensor):
    pred_clamp = pred.clamp(0.0, 1.0)
    target_clamp = target.clamp(0.0, 1.0)
    return {
        'psnr': psnr(pred_clamp, target_clamp),
        'ssim': ssim(pred_clamp, target_clamp),
    }