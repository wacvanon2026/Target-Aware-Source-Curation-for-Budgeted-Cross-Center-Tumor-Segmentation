import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDecoder(nn.Module):

    def __init__(self, in_channels, out_channels=128, upsample_scale=32):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.upsample_scale = upsample_scale

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[-1]
        x = self.conv_block(x)
        x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return x
