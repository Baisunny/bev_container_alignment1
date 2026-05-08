import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import Config

class BEVFormerEncoder(nn.Module):
    def __init__(self, embed_dim=256, bev_h=None, bev_w=None, num_layers=3, num_cams=2, num_heads=8, in_channels=(512,1024,2048)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_cams = num_cams
        self.num_heads = num_heads
        h = bev_h if bev_h is not None else Config.bev_h
        w = bev_w if bev_w is not None else Config.bev_w
        self.bev_h = h
        self.bev_w = w
        self.bev_queries = nn.Parameter(torch.randn(h * w, embed_dim))
        c8, c16, c32 = in_channels
        self.proj8 = nn.Conv2d(c8, embed_dim, 1)
        self.proj16 = nn.Conv2d(c16, embed_dim, 1)
        self.proj32 = nn.Conv2d(c32, embed_dim, 1)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.self_attn = nn.ModuleList([nn.MultiheadAttention(embed_dim, num_heads=self.num_heads, batch_first=True) for _ in range(num_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.ReLU(), nn.Linear(embed_dim*2, embed_dim)) for _ in range(num_layers)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def _bev_grid_world(self, device):
        ys = torch.linspace(Config.y_range[0], Config.y_range[1], self.bev_h, device=device)
        xs = torch.linspace(Config.x_range[0], Config.x_range[1], self.bev_w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)
        return pts

    def _project(self, P, pts_w, out_hw, stride):
        B = P.shape[0]
        Hs, Ws = out_hw
        pts = pts_w.view(1, self.bev_h*self.bev_w, 3).repeat(B, 1, 1)
        ones = torch.ones(B, pts.shape[1], 1, device=pts.device)
        pts_h = torch.cat([pts, ones], dim=-1)
        p = torch.matmul(pts_h, P.transpose(-1, -2))
        u = p[..., 0] / (p[..., 2] + 1e-6)
        v = p[..., 1] / (p[..., 2] + 1e-6)
        u = u.view(B, self.bev_h, self.bev_w)
        v = v.view(B, self.bev_h, self.bev_w)
        u = (u / (Config.img_w / stride)) * 2 - 1
        v = (v / (Config.img_h / stride)) * 2 - 1
        grid = torch.stack([u, v], dim=-1)
        return grid

    def _spatial_cross_attention(self, feats, calibs, flip=None):
        B = feats["stride8"].shape[0]
        pts_w = self._bev_grid_world(feats["stride8"].device)
        pts_w = pts_w
        grids8 = []
        grids16 = []
        grids32 = []
        for cam in range(self.num_cams):
            P = calibs[:, cam]
            g8 = self._project(P, pts_w, feats["stride8"].shape[-2:], 8)
            g16 = self._project(P, pts_w, feats["stride16"].shape[-2:], 16)
            g32 = self._project(P, pts_w, feats["stride32"].shape[-2:], 32)
            if flip is not None:
                f = flip.view(B, 1, 1).float()
                g8[..., 0] = g8[..., 0] * (1 - 2 * f)
                g16[..., 0] = g16[..., 0] * (1 - 2 * f)
                g32[..., 0] = g32[..., 0] * (1 - 2 * f)
            grids8.append(g8)
            grids16.append(g16)
            grids32.append(g32)
        grids8 = torch.stack(grids8, dim=1)
        grids16 = torch.stack(grids16, dim=1)
        grids32 = torch.stack(grids32, dim=1)
        f8 = []
        f16 = []
        f32 = []
        for cam in range(self.num_cams):
            f8.append(F.grid_sample(feats["stride8"], grids8[:, cam], align_corners=False))
            f16.append(F.grid_sample(feats["stride16"], grids16[:, cam], align_corners=False))
            f32.append(F.grid_sample(feats["stride32"], grids32[:, cam], align_corners=False))
        f8 = torch.stack(f8, dim=1).mean(dim=1)
        f16 = torch.stack(f16, dim=1).mean(dim=1)
        f32 = torch.stack(f32, dim=1).mean(dim=1)
        e8 = self.proj8(f8)
        e16 = self.proj16(f16)
        e32 = self.proj32(f32)
        e = (e8 + e16 + e32)
        e = e.view(B, self.embed_dim, self.bev_h*self.bev_w).transpose(1, 2)
        return e

    def forward(self, feats_per_cam, calibs, flip=None):
        B = feats_per_cam["stride8"].shape[0]
        queries = self.bev_queries.unsqueeze(0).expand(B, -1, -1)
        sampled = self._spatial_cross_attention(feats_per_cam, calibs, flip=flip)
        queries = self.norm_q(queries + sampled)
        for i in range(self.num_layers):
            a, _ = self.self_attn[i](queries, queries, queries)
            queries = self.ln1[i](queries + a)
            f = self.ffn[i](queries)
            queries = self.ln2[i](queries + f)
        bev_feat = queries.transpose(1, 2).view(B, self.embed_dim, self.bev_h, self.bev_w)
        return bev_feat
