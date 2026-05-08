import os
import yaml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from utils.data_loader import ContainerDataset
from models.backbone import ResNetBackbone
from models.bevformer import BEVFormerEncoder
from models.heads import HeatmapHead
from utils.postprocess import subpixel_peak, bev_to_phys, compute_bias

def build_calib_tensor(batch_calib, device):
    Ps = []
    for c in batch_calib:
        Kf = c['front']['K'].to(device); Rf = c['front']['R'].to(device); Tf = c['front']['T'].to(device)
        Kr = c['rear']['K'].to(device); Rr = c['rear']['R'].to(device); Tr = c['rear']['T'].to(device)
        Pf = torch.matmul(Kf, torch.cat([Rf, Tf], dim=1))
        Pr = torch.matmul(Kr, torch.cat([Rr, Tr], dim=1))
        Ps.append(torch.stack([Pf, Pr], dim=0))
    return torch.stack(Ps, dim=0)

def latest_checkpoint(path='checkpoints'):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        return None
    files.sort()
    return os.path.join(path, files[-1])

def project_world_to_img(P, xy):
    pts = np.hstack([xy, np.zeros((xy.shape[0], 1)), np.ones((xy.shape[0], 1))]).astype(np.float32)
    p = (P @ pts.T).T
    u = p[:, 0] / (p[:, 2] + 1e-6)
    v = p[:, 1] / (p[:, 2] + 1e-6)
    return np.stack([u, v], axis=1)

def main():
    with open('configs/default.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ContainerDataset('processed', cfg)
    val_ratio = 0.2
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    _, val_set = random_split(dataset, [n_train, n_val])
    dl_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    backbone = ResNetBackbone(pretrained=False).to(device)
    encoder = BEVFormerEncoder(embed_dim=256, bev_h=cfg['model']['bev_h'], bev_w=cfg['model']['bev_w'], num_layers=3, num_cams=2).to(device)
    head = HeatmapHead(in_channels=256).to(device)
    ckpt_path = latest_checkpoint()
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        encoder.load_state_dict(ckpt['encoder'])
        head.load_state_dict(ckpt['head'])
    backbone.eval(); encoder.eval(); head.eval()
    x_min, x_max = cfg['bev']['x_range'][0], cfg['bev']['x_range'][1]
    y_min, y_max = cfg['bev']['y_range'][0], cfg['bev']['y_range'][1]
    res = cfg['bev']['resolution']
    dx_series = []; dy_series = []
    os.makedirs('outputs/visuals', exist_ok=True)
    idx = 0
    for batch in dl_val:
        front = batch['front_img'][0].numpy().transpose(1, 2, 0)
        rear = batch['rear_img'][0].numpy().transpose(1, 2, 0)
        front_b = batch['front_img'].to(device)
        rear_b = batch['rear_img'].to(device)
        calib = batch['calib']
        calib_list = [{
            'front': {k: (v[0] if isinstance(v, torch.Tensor) else torch.from_numpy(v)).float() for k, v in calib['front'].items()},
            'rear': {k: (v[0] if isinstance(v, torch.Tensor) else torch.from_numpy(v)).float() for k, v in calib['rear'].items()}
        }]
        P = build_calib_tensor(calib_list, device)
        with torch.no_grad():
            f_feats = backbone(front_b)
            r_feats = backbone(rear_b)
            feats = {
                'stride8': (f_feats['stride8'] + r_feats['stride8']) / 2.0,
                'stride16': (f_feats['stride16'] + r_feats['stride16']) / 2.0,
                'stride32': (f_feats['stride32'] + r_feats['stride32']) / 2.0
            }
            bev = encoder(feats, P)
            pred_c, pred_s = head(bev)
        pred = torch.cat([pred_c, pred_s], dim=1)[0].cpu().numpy()
        pts_bev = []
        for k in range(pred.shape[0]):
            u, v = subpixel_peak(pred[k])
            pts_bev.append([u, v])
        pts_bev = np.array(pts_bev, dtype=np.float32)
        cont = pts_bev[:4]; spr = pts_bev[4:]
        cont_phys = np.array([bev_to_phys(u, v, x_min, y_min, res) for u, v in cont], dtype=np.float32)
        spr_phys = np.array([bev_to_phys(u, v, x_min, y_min, res) for u, v in spr], dtype=np.float32)
        Pf = P[0, 0].cpu().numpy(); Pr = P[0, 1].cpu().numpy()
        cont_f = project_world_to_img(Pf, cont_phys); spr_f = project_world_to_img(Pf, spr_phys)
        cont_r = project_world_to_img(Pr, cont_phys); spr_r = project_world_to_img(Pr, spr_phys)
        img_f = (front * 255).astype(np.uint8).copy()
        img_r = (rear * 255).astype(np.uint8).copy()
        for p in cont_f: cv2.circle(img_f, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        for p in spr_f: cv2.circle(img_f, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        for p in cont_r: cv2.circle(img_r, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        for p in spr_r: cv2.circle(img_r, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_f); axes[0].axis('off'); axes[0].set_title('Front')
        axes[1].imshow(img_r); axes[1].axis('off'); axes[1].set_title('Rear')
        plt.tight_layout()
        plt.savefig(f'outputs/visuals/sample_{idx:04d}_images.png')
        plt.close(fig)
        cx = np.mean(cont_phys, axis=0); sx = np.mean(spr_phys, axis=0)
        dx, dy = compute_bias(cx, sx)
        dx_series.append(dx); dy_series.append(dy)
        idx += 1
    t = np.arange(len(dx_series))
    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax2[0].plot(t, dx_series, label='ΔX'); ax2[0].legend(); ax2[0].grid(True)
    ax2[1].plot(t, dy_series, label='ΔY'); ax2[1].legend(); ax2[1].grid(True)
    ax2[1].set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig('outputs/visuals/bias_curves.png')
    plt.close(fig2)

if __name__ == '__main__':
    main()
