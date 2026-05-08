import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import Config

class SpatialCrossAttention(nn.Module):
    """
    Project BEV queries to image features and update them.
    Simplified version using point sampling.
    """
    def __init__(self, embed_dim=256, num_cams=2, num_points_in_pillar=4):
        super(SpatialCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_cams = num_cams
        self.num_points = num_points_in_pillar
        
        # Projections for sampling
        self.sampling_offsets = nn.Linear(embed_dim, num_cams * num_points_in_pillar * 2)
        self.attention_weights = nn.Linear(embed_dim, num_cams * num_points_in_pillar)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, queries, img_feats, calibs):
        """
        queries: (B, H_bev*W_bev, embed_dim)
        img_feats: (B, num_cams, C, H_img, W_img)
        calibs: dict with 'front' and 'rear' calibration
        """
        B, N_q, D = queries.shape
        num_cams = self.num_cams
        num_points = self.num_points
        
        # 1. Generate 3D reference points for each BEV query
        # For simplicity, we assume queries are on a grid.
        # We sample Z at different heights.
        h_bev, w_bev = Config.bev_h, Config.bev_w
        x_range, y_range = Config.x_range, Config.y_range
        
        # Create grid (x, y)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(y_range[0], y_range[1], h_bev, device=queries.device),
            torch.linspace(x_range[0], x_range[1], w_bev, device=queries.device),
            indexing='ij'
        )
        points_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) # (N_q, 2)
        
        # Expand to 3D with different Z (pillars)
        # Assuming Z ranges from some value (e.g., -100 to 100 mm)
        z_coords = torch.linspace(-100, 100, num_points, device=queries.device)
        points_3d = []
        for z in z_coords:
            p3d = torch.cat([points_xy, torch.full((N_q, 1), z, device=queries.device)], dim=-1)
            points_3d.append(p3d)
        points_3d = torch.stack(points_3d, dim=1) # (N_q, num_points, 3)
        points_3d = points_3d.unsqueeze(0).expand(B, -1, -1, -1) # (B, N_q, num_points, 3)
        
        # 2. Project 3D points to images
        # We need the projection matrices from calibs
        # P_front, P_rear: (B, 3, 4)
        # For now, assume calibs are provided for each batch
        
        sampled_feats = []
        for cam_idx in range(num_cams):
            # Get P for this camera
            # (In a real scenario, this would be from the batch's calib)
            # Placeholder for projection matrix
            P = calibs[cam_idx] # (B, 3, 4)
            
            # Project points: p_img = P * [X, Y, Z, 1]^T
            p3d_hom = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1) # (B, N_q, num_points, 4)
            p_img_hom = torch.matmul(p3d_hom, P.transpose(-1, -2)) # (B, N_q, num_points, 3)
            
            # Normalize by depth (Z_cam)
            u = p_img_hom[..., 0] / (p_img_hom[..., 2] + 1e-6)
            v = p_img_hom[..., 1] / (p_img_hom[..., 2] + 1e-6)
            
            # Normalize to [-1, 1] for grid_sample
            u_norm = (u / Config.img_w) * 2 - 1
            v_norm = (v / Config.img_h) * 2 - 1
            
            grid = torch.stack([u_norm, v_norm], dim=-1) # (B, N_q, num_points, 2)
            
            # Sample from image features
            # img_feats: (B, num_cams, C, H_f, W_f)
            feat = img_feats[:, cam_idx] # (B, C, H_f, W_f)
            # Reshape grid for grid_sample: (B, H_grid, W_grid, 2)
            # Here H_grid = N_q, W_grid = num_points
            s_feat = F.grid_sample(feat, grid, align_corners=False) # (B, C, N_q, num_points)
            sampled_feats.append(s_feat)
            
        # 3. Aggregate features
        sampled_feats = torch.stack(sampled_feats, dim=1) # (B, num_cams, C, N_q, num_points)
        
        # Average over cams and points in pillar
        # In BEVFormer, this is usually weighted by attention
        out = sampled_feats.mean(dim=(1, 4)) # (B, C, N_q)
        out = out.transpose(1, 2) # (B, N_q, C)
        
        # Final projection
        queries = queries + self.output_proj(out)
        
        return queries

class BEVFormerLayer(nn.Module):
    def __init__(self, embed_dim=256):
        super(BEVFormerLayer, self).__init__()
        self.spatial_cross_attn = SpatialCrossAttention(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, queries, img_feats, calibs):
        # 1. Spatial Cross Attention
        queries = self.spatial_cross_attn(queries, img_feats, calibs)
        queries = self.norm1(queries)
        
        # 2. Self Attention (Temporal/Spatial)
        attn_out, _ = self.self_attn(queries, queries, queries)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # 3. FFN
        queries = queries + self.ffn(queries)
        queries = self.norm3(queries)
        
        return queries
