import torch
import torch.nn as nn
import torch.nn.functional as F

# Partial Convolution
class PartialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, bias = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, bias = bias)
        
        # Fixed all-ones kernel for mask updating (not trained)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride = stride, padding = padding, bias = False)
        nn.init.constant_(self.mask_conv.weight, 1.0)

        for p in self.mask_conv.parameters():
            p.requires_grad = False

        self.kernel_size = kernel_size
        self.in_channels = in_channels

    def forward(self, x, mask):
        # Scale input by mask so holes contribute zero
        x_masked = x * mask

        # Apply convolution on masked input
        out = self.conv(x_masked)

        # Compute mask sum for normalization
        with torch.no_grad():
            mask_sum = self.mask_conv(mask)

        # Normalize: divide by number of valid pixels in each window
        total = self.kernel_size * self.kernel_size * self.in_channels
        scale = total / (mask_sum + 1e-8)

        # Only apply scale where there was at least one valid pixel
        scale = torch.clamp(scale, max = total)
        out = out * scale

        # Add bias after scaling
        if self.conv.bias is not None:
            bias = self.conv.bias.view(1, -1, 1, 1)
            out = out + bias * (1 - torch.clamp(mask_sum, 0, 1))

        # Update mask
        with torch.no_grad():
            new_mask = torch.clamp(mask_sum, 0, 1)

        return out, new_mask


# Building Blocks
class PartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, activation = 'relu'):
        super().__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()

        if activation == 'relu':
            self.act = nn.ReLU(inplace = True)
        elif activation == 'leaky':
            self.act = nn.LeakyReLU(0.2, inplace = True)
        else:
            self.act = nn.Identity()

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.act(self.bn(x))
        return x, mask

class DoublePartialConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True):
        super().__init__()
        self.block1 = PartialConvBlock(in_channels, out_channels, bn = bn)
        self.block2 = PartialConvBlock(out_channels, out_channels, bn = bn)

    def forward(self, x, mask):
        x, mask = self.block1(x, mask)
        x, mask = self.block2(x, mask)
        return x, mask


# Inpainting U-Net
class InpaintingUNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, features = [64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.enc1 = DoublePartialConvBlock(in_channels, features[0])
        self.enc2 = DoublePartialConvBlock(features[0], features[1])
        self.enc3 = DoublePartialConvBlock(features[1], features[2])
        self.enc4 = DoublePartialConvBlock(features[2], features[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoublePartialConvBlock(features[3], features[3] * 2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride = 2)
        self.dec4 = DoublePartialConvBlock(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride = 2)
        self.dec3 = DoublePartialConvBlock(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride = 2)
        self.dec2 = DoublePartialConvBlock(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride = 2)
        self.dec1 = DoublePartialConvBlock(features[0] * 2, features[0])

        # Output
        self.final = nn.Conv2d(features[0], out_channels, kernel_size = 1)
        self.sigmoid = nn.Sigmoid()

    def _pool_mask(self, mask):
        return F.max_pool2d(mask, 2)

    def _pad(self, x, skip):
        dH = skip.size(2) - x.size(2)
        dW = skip.size(3) - x.size(3)
        return F.pad(x, [dW // 2, dW - dW // 2, dH // 2, dH - dH // 2])

    def forward(self, x, mask):
        # Encoder path
        s1, m1 = self.enc1(x, mask)
        s2, m2 = self.enc2(self._pool(s1), self._pool_mask(m1))
        s3, m3 = self.enc3(self._pool(s2), self._pool_mask(m2))
        s4, m4 = self.enc4(self._pool(s3), self._pool_mask(m3))

        # Bottleneck
        b, mb = self.bottleneck(self._pool(s4), self._pool_mask(m4))

        # Decoder path
        d4 = self._pad(self.up4(b), s4)
        md4 = self._pad(F.interpolate(mb, scale_factor = 2, mode = 'nearest'), m4)
        d4, _ = self.dec4(torch.cat([d4, s4], 1), torch.cat([md4, m4], 1)[:, :1])
        d3 = self._pad(self.up3(d4), s3)
        md3 = self._pad(F.interpolate(md4, scale_factor = 2, mode = 'nearest'), m3)
        d3, _ = self.dec3(torch.cat([d3, s3], 1), torch.cat([md3, m3], 1)[:, :1])
        d2 = self._pad(self.up2(d3), s2)
        md2 = self._pad(F.interpolate(md3, scale_factor = 2, mode = 'nearest'), m2)
        d2, _ = self.dec2(torch.cat([d2, s2], 1), torch.cat([md2, m2], 1)[:, :1])
        d1 = self._pad(self.up1(d2), s1)
        md1 = self._pad(F.interpolate(md2, scale_factor = 2, mode = 'nearest'), m1)
        d1, _ = self.dec1(torch.cat([d1, s1], 1), torch.cat([md1, m1], 1)[:, :1])
        out = self.sigmoid(self.final(d1))

        # Composite: keep known pixels from input, use model output for holes
        out = mask * x + (1 - mask) * out
        return out

    def _pool(self, x):
        return self.pool(x)

# Checking the model with a simple test case
if __name__ == "__main__":
    model = InpaintingUNet(in_channels = 1, out_channels = 1)
    x = torch.randn(2, 1, 256, 256)
    mask = torch.ones(2, 1, 256, 256)
    mask[:, :, 80:160, 80:160] = 0 # simulate a square hole
    out = model(x, mask)
    print(f"Input: {x.shape}")
    print(f"Mask: {mask.shape}  (0 = hole, 1 = known)")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")