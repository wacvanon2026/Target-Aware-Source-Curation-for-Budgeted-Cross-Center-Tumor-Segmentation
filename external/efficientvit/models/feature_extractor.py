import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):

    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, x):
        with torch.no_grad():
            feats = self.seg_model.model.backbone(x)
            if isinstance(feats, dict):
                feats = list(feats.values())[-1]
        return feats
