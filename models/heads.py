import torch
import torch.nn as nn

class HeatmapHead(nn.Module):
    def __init__(self, in_channels=256, mid_channels=96, num_container_points=4, num_spreader_points=4):
        super().__init__()
        self.container_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_container_points, kernel_size=1)
        )
        self.spreader_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_spreader_points, kernel_size=1)
        )

    def forward(self, x):
        c = torch.sigmoid(self.container_branch(x))
        s = torch.sigmoid(self.spreader_branch(x))
        return c, s
