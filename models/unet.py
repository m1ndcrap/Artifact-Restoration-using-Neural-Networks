import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch from odd dimensions
        diffH = skip.size(2) - x.size(2)
        diffW = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
        x = torch.cat([skip, x], dim = 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, features[0])
        self.enc2 = DownBlock(features[0], features[1])
        self.enc3 = DownBlock(features[1], features[2])
        self.enc4 = DownBlock(features[2], features[3])

        # Bottleneck
        self.bottleneck = DownBlock(features[3], features[3] * 2)

        # Decoder
        self.dec4 = UpBlock(features[3] * 2, features[3])
        self.dec3 = UpBlock(features[3], features[2])
        self.dec2 = UpBlock(features[2], features[1])
        self.dec1 = UpBlock(features[1], features[0])

        # Final 1x1 conv
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder path with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Residual learning: predict noise, subtract from input
        noise_pred = self.final(d1)
        return x - noise_pred


if __name__ == "__main__":
    model = UNet(in_channels = 1, out_channels = 1)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")