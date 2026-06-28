import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientvit.seg_model_zoo import create_efficientvit_seg_model
from .decoder import SimpleDecoder


class EfficientViT_Seg(nn.Module):
    """
    EfficientViT segmentation model for BraTS MRI segmentation (4-channel input, 4-class output)
    """

    def __init__(self, backbone="efficientvit_l1", num_classes=4, in_channels=4, pretrained=True):
        super().__init__()

        # === 1️⃣ 加载 ADE20K pretrained EfficientViT ===
        # print(f"🧠 Loading EfficientViT backbone: {backbone} (pretrained={pretrained})")
        try:
            self.model = create_efficientvit_seg_model(f"efficientvit-seg-l1-ade20k", pretrained=pretrained)
        except FileNotFoundError:
            if not pretrained:
                raise
            print("EfficientViT pretrained checkpoint not found; initializing segmentation backbone without pretrained weights.")
            self.model = create_efficientvit_seg_model(f"efficientvit-seg-l1-ade20k", pretrained=False)

        # === 2️⃣ 替换输入层（自适应通道数） ===
        old_conv = self.model.backbone.stages[0].op_list[0].conv  # 第一层 Conv2d

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if in_channels == 3:
                # ✅ LiTS / ATLAS / RGB-like
                new_conv.weight.copy_(old_conv.weight)

            elif in_channels > 3:
                # ✅ BraTS / multi-modal
                new_conv.weight[:, :3, :, :] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c, :, :] = old_conv.weight.mean(dim=1)

            else:
                raise ValueError(f"Unsupported in_channels={in_channels}")

        self.model.backbone.stages[0].op_list[0].conv = new_conv


        # === 3️⃣ 获取 backbone 输出通道 ===
        last_block = list(self.model.backbone.stages[-1].op_list)[-1]
        backbone_out_ch = last_block.context_module.main.proj.conv.out_channels
        # print(f"🔍 Backbone output channels: {backbone_out_ch}")


        # === 4️⃣ 自定义 Decoder（上采样 ×32）===
        self.decoder = SimpleDecoder(in_channels=backbone_out_ch, out_channels=128, upsample_scale=32)

        # === 5️⃣ 输出分类层 ===
        self.final_head = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = self.model.backbone(x)

        # 取最后一层特征
        if isinstance(feats, dict):
            feats = list(feats.values())[-1]

        out = self.decoder(feats)
        out = self.final_head(out)

        # 保证输出尺寸与输入一致
        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        return out
