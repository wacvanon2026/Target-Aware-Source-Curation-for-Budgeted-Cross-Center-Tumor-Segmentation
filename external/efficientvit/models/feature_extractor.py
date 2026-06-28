import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    Feature extractor for EfficientViT_Seg
    ✅ Works perfectly with your current EfficientViT_Seg definition.
    It returns the backbone feature maps (before decoder).
    """

    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, x):
        with torch.no_grad():
            # 提取 EfficientViT 主干特征
            feats = self.seg_model.model.backbone(x)

            # 兼容返回 dict 的情况（EfficientViT 有时输出多层特征）
            if isinstance(feats, dict):
                feats = list(feats.values())[-1]

        return feats
