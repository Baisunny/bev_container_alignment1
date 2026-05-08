import torch
import numpy as np
import cv2
from configs.config import Config

class PostProcessor:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def decode_heatmap(self, heatmap, offset=None):
        """
        heatmap: (B, C, H, W)
        offset: (B, C*2, H, W)
        """
        B, C, H, W = heatmap.shape
        
        # 1. Local maximum filter (3x3)
        h_max = torch.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
        keep = (h_max == heatmap).float()
        heatmap = heatmap * keep
        
        # 2. Get top points for each class
        # For our task, we only need 1 point per class (8 classes total)
        # Reshape to (B, C, H*W)
        scores, indices = torch.max(heatmap.view(B, C, -1), dim=2) # (B, C)
        
        v = (indices // W).float() # (B, C)
        u = (indices % W).float() # (B, C)
        
        # 3. Add offsets for sub-pixel accuracy
        if offset is not None:
            # offset: (B, C*2, H, W) -> [u_off, v_off] for each class
            # B, C*2, H, W
            # For class c: offset[:, 2*c], offset[:, 2*c+1]
            for c in range(C):
                u_off = offset[:, 2*c].view(B, -1).gather(1, indices[:, c:c+1]).squeeze(1)
                v_off = offset[:, 2*c+1].view(B, -1).gather(1, indices[:, c:c+1]).squeeze(1)
                u[:, c] += u_off
                v[:, c] += v_off
        
        # 4. Map to physical coordinates (mm)
        # u, v: (B, C) - coordinates in BEV grid
        x_mm = u * Config.bev_resolution + Config.x_range[0]
        y_mm = v * Config.bev_resolution + Config.y_range[0]
        
        # Filter by threshold
        valid_mask = scores > self.threshold # (B, C)
        
        # Calculate centroids for spreader and container
        # Spreader: points 0-3, Container: points 4-7
        spreader_x = (x_mm[:, 0:4] * scores[:, 0:4] * valid_mask[:, 0:4]).sum(1) / (scores[:, 0:4] * valid_mask[:, 0:4]).sum(1).clamp(min=1e-6)
        spreader_y = (y_mm[:, 0:4] * scores[:, 0:4] * valid_mask[:, 0:4]).sum(1) / (scores[:, 0:4] * valid_mask[:, 0:4]).sum(1).clamp(min=1e-6)
        
        container_x = (x_mm[:, 4:8] * scores[:, 4:8] * valid_mask[:, 4:8]).sum(1) / (scores[:, 4:8] * valid_mask[:, 4:8]).sum(1).clamp(min=1e-6)
        container_y = (y_mm[:, 4:8] * scores[:, 4:8] * valid_mask[:, 4:8]).sum(1) / (scores[:, 4:8] * valid_mask[:, 4:8]).sum(1).clamp(min=1e-6)
        
        # Horizontal offsets
        delta_x = spreader_x - container_x
        delta_y = spreader_y - container_y
        
        return {
            'points_mm': torch.stack([x_mm, y_mm], dim=-1), # (B, C, 2)
            'scores': scores,
            'spreader_centroid': torch.stack([spreader_x, spreader_y], dim=-1),
            'container_centroid': torch.stack([container_x, container_y], dim=-1),
            'delta_x': delta_x,
            'delta_y': delta_y
        }

class KalmanFilter:
    """
    Simple Kalman Filter for smoothing offsets.
    """
    def __init__(self, q=1e-2, r=1e-1):
        self.q = q # process noise
        self.r = r # measurement noise
        self.x_hat = None # estimated state
        self.p = 1.0 # estimation error covariance

    def update(self, measurement):
        if self.x_hat is None:
            self.x_hat = measurement
            return self.x_hat
        
        # Prediction
        p_pred = self.p + self.q
        
        # Update
        k = p_pred / (p_pred + self.r)
        self.x_hat = self.x_hat + k * (measurement - self.x_hat)
        self.p = (1 - k) * p_pred
        
        return self.x_hat
