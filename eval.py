import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from utils.data_loader import ContainerDataset
from models.backbone import ResNetBackbone
from models.bevformer import BEVFormerEncoder
from models.heads import HeatmapHead
from utils.postprocess import subpixel_peak, confidence_weighted_centroid, bev_to_phys, compute_bias

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
    bev_h = cfg['model']['bev_h']; bev_w = cfg['model']['bev_w']
    x_min, x_max = cfg['bev']['x_range'][0], cfg['bev']['x_range'][1]
    y_min, y_max = cfg['bev']['y_range'][0], cfg['bev']['y_range'][1]
    res = cfg['bev']['resolution']
    rmse_sum = 0.0; rmse_count = 0
    dx_abs_sum = 0.0; dy_abs_sum = 0.0; bias_count = 0
    for batch in dl_val:
        front = batch['front_img'].to(device)
        rear = batch['rear_img'].to(device)
        target = batch['bev_heatmap'][0].cpu().numpy()
        calib = batch['calib']
        calib_list = [{
            'front': {k: (v[0] if isinstance(v, torch.Tensor) else torch.from_numpy(v)).float() for k, v in calib['front'].items()},
            'rear': {k: (v[0] if isinstance(v, torch.Tensor) else torch.from_numpy(v)).float() for k, v in calib['rear'].items()}
        }]
        P = build_calib_tensor(calib_list, device)
        with torch.no_grad():
            f_feats = backbone(front)
            r_feats = backbone(rear)
            feats = {
                'stride8': (f_feats['stride8'] + r_feats['stride8']) / 2.0,
                'stride16': (f_feats['stride16'] + r_feats['stride16']) / 2.0,
                'stride32': (f_feats['stride32'] + r_feats['stride32']) / 2.0
            }
            bev = encoder(feats, P)
            pred_c, pred_s = head(bev)
        pred = torch.cat([pred_c, pred_s], dim=1)[0].cpu().numpy()
        pred_pts = []
        gt_pts = []
        conf_pred = []
        conf_gt = []
        for k in range(pred.shape[0]):
            u_pred, v_pred = subpixel_peak(pred[k])
            u_gt, v_gt = subpixel_peak(target[k])
            pred_pts.append([u_pred, v_pred])
            gt_pts.append([u_gt, v_gt])
            conf_pred.append(float(np.max(pred[k])))
            conf_gt.append(float(np.max(target[k])))
        pred_pts = np.array(pred_pts, dtype=np.float32)
        gt_pts = np.array(gt_pts, dtype=np.float32)
        rmse = np.sqrt(np.mean((pred_pts - gt_pts) ** 2))
        rmse_sum += rmse; rmse_count += 1
        cont_pred = pred_pts[:4]; spr_pred = pred_pts[4:]
        cont_gt = gt_pts[:4]; spr_gt = gt_pts[4:]
        cx_pred, cy_pred = confidence_weighted_centroid(cont_pred, conf_pred[:4])
        sx_pred, sy_pred = confidence_weighted_centroid(spr_pred, conf_pred[4:])
        cx_gt, cy_gt = confidence_weighted_centroid(cont_gt, conf_gt[:4])
        sx_gt, sy_gt = confidence_weighted_centroid(spr_gt, conf_gt[4:])
        xcp, ycp = bev_to_phys(cx_pred, cy_pred, x_min, y_min, res)
        xsp, ysp = bev_to_phys(sx_pred, sy_pred, x_min, y_min, res)
        xcg, ycg = bev_to_phys(cx_gt, cy_gt, x_min, y_min, res)
        xsg, ysg = bev_to_phys(sx_gt, sy_gt, x_min, y_min, res)
        dx_pred, dy_pred = compute_bias((xcp, ycp), (xsp, ysp))
        dx_gt, dy_gt = compute_bias((xcg, ycg), (xsg, ysg))
        dx_abs_sum += abs(dx_pred - dx_gt)
        dy_abs_sum += abs(dy_pred - dy_gt)
        bias_count += 1
    metrics = {
        'rmse_pixels': rmse_sum / max(1, rmse_count),
        'dx_mae_mm': dx_abs_sum / max(1, bias_count),
        'dy_mae_mm': dy_abs_sum / max(1, bias_count),
        'samples': bias_count
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/eval_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
