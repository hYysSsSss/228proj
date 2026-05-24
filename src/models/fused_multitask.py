import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation import DoubleConv


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class FeatureUNetSegmenter(nn.Module):
    """U-Net segmentation model that exposes encoder features for fusion."""

    def __init__(self, in_channels=6, num_classes=5, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.channels = channels
        self.pool = nn.MaxPool2d(2)
        self.down1 = DoubleConv(in_channels, c1)
        self.down2 = DoubleConv(c1, c2)
        self.down3 = DoubleConv(c2, c3)
        self.down4 = DoubleConv(c3, c4)
        self.mid = DoubleConv(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, 2, 2)
        self.conv4 = DoubleConv(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.conv3 = DoubleConv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.conv2 = DoubleConv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.conv1 = DoubleConv(c1 + c1, c1)
        self.out = nn.Conv2d(c1, num_classes, 1)

    def encode(self, x):
        f1 = self.down1(x)
        f2 = self.down2(self.pool(f1))
        f3 = self.down3(self.pool(f2))
        f4 = self.down4(self.pool(f3))
        f5 = self.mid(self.pool(f4))
        return [f1, f2, f3, f4, f5]

    def decode(self, feats):
        f1, f2, f3, f4, f5 = feats
        x = self.conv4(torch.cat([self.up4(f5), f4], dim=1))
        x = self.conv3(torch.cat([self.up3(x), f3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), f2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), f1], dim=1))
        return self.out(x)

    def forward(self, x, return_features=False):
        feats = self.encode(x)
        logits = self.decode(feats)
        if return_features:
            return logits, feats
        return logits


class ResNetStageClassifier(nn.Module):
    """ResNet-like classifier with stages aligned to the U-Net encoder."""

    def __init__(self, in_channels=6, num_classes=4, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.channels = channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            ResidualBlock(c1, c1),
        )
        self.stage2 = nn.Sequential(ResidualBlock(c1, c2, stride=2), ResidualBlock(c2, c2))
        self.stage3 = nn.Sequential(ResidualBlock(c2, c3, stride=2), ResidualBlock(c3, c3))
        self.stage4 = nn.Sequential(ResidualBlock(c3, c4, stride=2), ResidualBlock(c4, c4))
        self.stage5 = nn.Sequential(ResidualBlock(c4, c5, stride=2), ResidualBlock(c5, c5))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c5, num_classes)

    def forward(self, x, seg_features=None, adapters=None, detach_seg=False, return_features=False):
        cls_features = []
        out = x
        for i, stage in enumerate([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5]):
            out = stage(out)
            if seg_features is not None and adapters is not None:
                guide = seg_features[i].detach() if detach_seg else seg_features[i]
                guide = adapters[i](guide)
                if guide.shape[-2:] != out.shape[-2:]:
                    guide = F.interpolate(guide, size=out.shape[-2:], mode="bilinear", align_corners=False)
                out = out + guide
            cls_features.append(out)
        logits = self.head(torch.flatten(self.pool(out), 1))
        if return_features:
            return logits, cls_features
        return logits


class SegGuidedClassifier(nn.Module):
    """Two-branch model: segmentation features guide classification through 1x1 adapters."""

    def __init__(self, in_channels=6, seg_classes=5, cls_classes=4, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        self.segmenter = FeatureUNetSegmenter(in_channels, seg_classes, channels)
        self.classifier = ResNetStageClassifier(in_channels, cls_classes, channels)
        self.adapters = nn.ModuleList([nn.Conv2d(ch, ch, 1, bias=False) for ch in channels])

    def forward(self, x, detach_seg_for_cls=False):
        seg_logits, seg_features = self.segmenter(x, return_features=True)
        cls_logits = self.classifier(
            x,
            seg_features=seg_features,
            adapters=self.adapters,
            detach_seg=detach_seg_for_cls,
        )
        return {"seg": seg_logits, "cls": cls_logits}

    def segmentation_parameters(self):
        return self.segmenter.parameters()

    def classification_fusion_parameters(self):
        for module in [self.classifier, self.adapters]:
            yield from module.parameters()
