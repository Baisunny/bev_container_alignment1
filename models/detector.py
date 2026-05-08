import torch
import torch.nn as nn
from models.backbone import Backbone
from models.bevformer_layer import BEVFormerLayer
from models.head import HeatmapHead
from configs.config import Config

class BEVFormerDetector(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3):
        super(BEVFormerDetector, self).__init__()
        self.backbone = Backbone()
        self.neck = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        # Learnable BEV queries
        self.bev_queries = nn.Parameter(torch.randn(Config.bev_h * Config.bev_w, embed_dim))
        
        # BEVFormer layers
        self.layers = nn.ModuleList([
            BEVFormerLayer(embed_dim) for _ in range(num_layers)
        ])
        
        # Heatmap prediction head
        self.head = HeatmapHead(in_channels=embed_dim, out_channels=Config.num_classes)

    def forward(self, img, calibs):
        """
        img: (B, num_cams, 3, H_img, W_img)
        calibs: dict with 'front' and 'rear' calibration
        """
        B, N_c, C, H, W = img.shape
        
        # 1. Extract multi-view image features
        # Reshape to (B*N_c, 3, H, W)
        img = img.view(B * N_c, C, H, W)
        feat = self.backbone(img) # (B*N_c, 2048, H/32, W/32)
        feat = self.neck(feat) # (B*N_c, embed_dim, H/32, W/32)
        
        # Reshape back to (B, N_c, embed_dim, H/32, W/32)
        feat = feat.view(B, N_c, -1, feat.shape[-2], feat.shape[-1])
        
        # 2. Initialize BEV queries
        queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1) # (B, N_q, embed_dim)
        
        # 3. Apply BEVFormer layers
        # In a real training, calibs should be passed correctly
        # For now, we assume calibs is a list of projection matrices [P_front, P_rear]
        for layer in self.layers:
            queries = layer(queries, feat, calibs)
            
        # 4. Reshape BEV features to 2D grid
        bev_feat = queries.transpose(1, 2).view(B, -1, Config.bev_h, Config.bev_w)
        
        # 5. Predict heatmaps
        hm, off = self.head(bev_feat)
        
        return hm, off
