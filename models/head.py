import torch
import torch.nn as nn
from configs.config import Config

class HeatmapHead(nn.Module):
    def __init__(self, in_channels=256, out_channels=8):
        super(HeatmapHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        # Heatmap prediction
        self.heatmap = nn.Conv2d(64, out_channels, kernel_size=1)
        # Offset prediction (optional, for sub-pixel accuracy)
        self.offset = nn.Conv2d(64, out_channels * 2, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H_bev, W_bev)
        """
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        hm = torch.sigmoid(self.heatmap(x))
        off = self.offset(x)
        
        return hm, off
