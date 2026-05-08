import os
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, freeze=True, pretrained_path=None):
        super().__init__()
        if name == 'resnet34':
            m = models.resnet34(weights=None)
            if pretrained and pretrained_path and os.path.isfile(pretrained_path):
                state = torch.load(pretrained_path, map_location='cpu')
                m.load_state_dict(state, strict=False)
            c8, c16, c32 = 128, 256, 512
        else:
            m = models.resnet50(weights=None)
            if pretrained and pretrained_path and os.path.isfile(pretrained_path):
                state = torch.load(pretrained_path, map_location='cpu')
                m.load_state_dict(state, strict=True)
            c8, c16, c32 = 512, 1024, 2048
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_channels = {"stride8": c8, "stride16": c16, "stride32": c32}
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return {
            "stride8": c2,
            "stride16": c3,
            "stride32": c4
        }
