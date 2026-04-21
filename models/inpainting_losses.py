import torch
import torch.nn as nn
import torch.nn.functional as F

class InpaintingLoss(nn.Module):
    def __init__(self, use_perceptual = True):
        super().__init__()
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.vgg = VGGFeatureExtractor()

    def forward(self, output, target, mask):
        if self.use_perceptual:
            self.vgg = self.vgg.to(output.device)

        # L1 on valid pixels
        l1_valid = F.l1_loss(mask * output, mask * target)

        # L1 on hole pixels (weighted higher)
        l1_hole = F.l1_loss((1 - mask) * output, (1 - mask) * target)

        total = l1_valid + 6.0 * l1_hole

        if self.use_perceptual:
            # Expand to 3 channels for VGG
            if output.shape[1] == 1:
                out3 = output.repeat(1, 3, 1, 1)
                tgt3 = target.repeat(1, 3, 1, 1)
            else:
                out3, tgt3 = output, target

            out_feats = self.vgg(out3)
            tgt_feats = self.vgg(tgt3)

            # Perceptual loss
            l_perceptual = sum(F.l1_loss(o, t) for o, t in zip(out_feats, tgt_feats))

            # Style loss (Gram matrices)
            l_style = sum(F.l1_loss(gram(o), gram(t)) for o, t in zip(out_feats, tgt_feats))

            total = total + 0.05 * l_perceptual + 120.0 * l_style

        # Total variation loss (smoothness in predicted holes)
        l_tv = total_variation(output * (1 - mask))
        total = total + 0.1 * l_tv

        return total


def gram(x):
    b, c, h, w = x.shape
    feat = x.view(b, c, -1)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


def total_variation(x):
    return (torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean())


class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            from torchvision import models
            vgg = models.vgg16(pretrained = False)
            layers = list(vgg.features.children())

            # Use features at 3 scales
            self.slice1 = nn.Sequential(*layers[:5]).eval() # relu1_2
            self.slice2 = nn.Sequential(*layers[5:10]).eval() # relu2_2
            self.slice3 = nn.Sequential(*layers[10:17]).eval() # relu3_3

            for p in self.parameters():
                p.requires_grad = False

            self.available = True
        except Exception:
            self.available = False

    def forward(self, x):
        if not self.available:
            return [x]
        
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]