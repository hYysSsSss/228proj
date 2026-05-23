import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base=32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.conv1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        m = self.mid(self.pool(d4))
        x = self.conv4(torch.cat([self.up4(m), d4], dim=1))
        x = self.conv3(torch.cat([self.up3(x), d3], dim=1))
        x = self.conv2(torch.cat([self.up2(x), d2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), d1], dim=1))
        return self.out(x)


def _replace_first_conv(model, in_channels):
    conv = model.backbone.conv1
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == 6:
            new.weight[:, :3].copy_(conv.weight)
            new.weight[:, 3:].copy_(conv.weight)
            new.weight.mul_(0.5)
    model.backbone.conv1 = new
    return model


def build_segmentation_model(name, num_classes, in_channels=6, pretrained=False):
    name = name.lower()
    if name in {"unet", "change_unet"}:
        return UNet(in_channels=in_channels, num_classes=num_classes)
    weights = None
    weights_backbone = models.ResNet50_Weights.DEFAULT if pretrained else None
    if name == "deeplabv3_resnet50":
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained and num_classes == 21 else None
        weights_backbone = None if weights is not None else weights_backbone
        model = models.segmentation.deeplabv3_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
        )
        if model.classifier[-1].out_channels != num_classes:
            model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    elif name == "fcn_resnet50":
        weights = models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained and num_classes == 21 else None
        weights_backbone = None if weights is not None else weights_backbone
        model = models.segmentation.fcn_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
        )
        if model.classifier[-1].out_channels != num_classes:
            model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        raise ValueError(f"Unknown segmentation model: {name}")
    if in_channels != 3:
        model = _replace_first_conv(model, in_channels)
    return model


def segmentation_logits(output):
    if isinstance(output, dict):
        return output["out"]
    return output
