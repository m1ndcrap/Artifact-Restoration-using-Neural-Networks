import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def forward(self, pred, target):
        return F.mse_loss(pred, target)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(pretrained = False)
    
            # Use first 9 layers (relu2_2) for feature extraction
            self.features = nn.Sequential(*list(vgg.features.children())[:9]).eval()
        
            for p in self.features.parameters():
                p.requires_grad = False
          
            self.available = True
        except ImportError:
            print("torchvision not found — PerceptualLoss will fall back to MSE.")
            self.available = False

    def forward(self, pred, target):
        if not self.available:
            return F.mse_loss(pred, target)
        
        # VGG expects 3-channel input
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.mse_loss(pred_feat, target_feat)


class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.mse = MSELoss()
        self.perceptual = PerceptualLoss()
        self.w = perceptual_weight

    def forward(self, pred, target):
        return self.mse(pred, target) + self.w * self.perceptual(pred, target)