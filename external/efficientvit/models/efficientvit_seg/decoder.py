import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDecoder(nn.Module):
    """
    通用上采样解码器：
    接收 EfficientViT backbone 的最后一层特征（通常是 1/32 尺度）
    自动上采样回原图大小。
    """

    def __init__(self, in_channels, out_channels=128, upsample_scale=32):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 最后上采样到原图分辨率
        self.upsample_scale = upsample_scale

    def forward(self, x):
        # x: (B, in_channels, H/32, W/32)
        if isinstance(x, dict):  # 万一输入是 dict
            x = list(x.values())[-1]

        x = self.conv_block(x)
        x = F.interpolate(x, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
        return x
