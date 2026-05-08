import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.data_loader import ContainerDataset
from models.backbone import ResNetBackbone
from models.bevformer import BEVFormerEncoder
from models.heads import HeatmapHead

def build_calib_tensor(batch_calib):
    Ps = []
    for c in batch_calib:
        Kf = c['front']['K']; Rf = c['front']['R']; Tf = c['front']['T']
        Kr = c['rear']['K']; Rr = c['rear']['R']; Tr = c['rear']['T']
        Pf = torch.matmul(Kf, torch.cat([Rf, Tf], dim=1))
        Pr = torch.matmul(Kr, torch.cat([Rr, Tr], dim=1))
        Ps.append(torch.stack([Pf, Pr], dim=0))
    return torch.stack(Ps, dim=0)

def centroid_from_heatmaps(hm, x_range, y_range, resolution):
    C, H, W = hm.shape
    y = torch.arange(H, device=hm.device).view(1, H, 1).float()
    x = torch.arange(W, device=hm.device).view(1, 1, W).float()
    xs = []; ys = []
    for i in range(C):
        h = hm[i]
        s = h.sum()
        if s.item() <= 0:
            xs.append(torch.tensor(W / 2.0, device=hm.device))
            ys.append(torch.tensor(H / 2.0, device=hm.device))
        else:
            xw = (h * x).sum() / s
            yw = (h * y).sum() / s
            xs.append(xw)
            ys.append(yw)
    x_px = torch.stack(xs).mean()
    y_px = torch.stack(ys).mean()
    x_mm = x_range[0] + x_px * resolution
    y_mm = y_range[0] + y_px * resolution
    return x_mm.item(), y_mm.item()

def centroid_pixel(hm):
    C, H, W = hm.shape
    y = torch.arange(H, device=hm.device).view(1, H, 1).float()
    x = torch.arange(W, device=hm.device).view(1, 1, W).float()
    xs = []; ys = []
    for i in range(C):
        h = hm[i]
        s = h.sum()
        if s.item() <= 0:
            xs.append(torch.tensor(W / 2.0, device=hm.device))
            ys.append(torch.tensor(H / 2.0, device=hm.device))
        else:
            xw = (h * x).sum() / s
            yw = (h * y).sum() / s
            xs.append(xw)
            ys.append(yw)
    x_px = torch.stack(xs).mean()
    y_px = torch.stack(ys).mean()
    return x_px.item(), y_px.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--print_samples', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = cfg.get('data_root', 'data/processed')
    test_dir = os.path.join(data_root, args.split)
    use_custom = not os.path.isdir(test_dir)
    if use_custom:
        # Respect split filtering in config even when split subdirectory is absent.
        ds = ContainerDataset(args.split, cfg, augment=False)
        if len(ds) == 0:
            ds = ContainerDataset('val', cfg, augment=False)
        if len(ds) == 0:
            ds = ContainerDataset('train', cfg, augment=False)
        samples = ds.samples[-args.max_samples:] if len(ds.samples) > 0 else []
        ds.samples = samples
    else:
        ds = ContainerDataset(args.split, cfg, augment=False)
    os.makedirs('test_results', exist_ok=True)
    sample_names = [os.path.basename(p.rstrip('/')) for p in getattr(ds, 'samples', [])]
    with open('test_results/test_samples.txt', 'w') as f:
        for n in sample_names:
            f.write(str(n) + '\n')
    if args.print_samples:
        print('test_samples:', sample_names)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    backbone = ResNetBackbone(
        name=cfg['model']['backbone'],
        pretrained=False,
        freeze=False,
        pretrained_path=cfg['model'].get('pretrained_path')
    ).to(device)
    in_ch = (backbone.out_channels['stride8'], backbone.out_channels['stride16'], backbone.out_channels['stride32'])
    encoder = BEVFormerEncoder(
        embed_dim=cfg['model']['embed_dim'],
        bev_h=cfg['bev']['bev_h'],
        bev_w=cfg['bev']['bev_w'],
        num_layers=cfg['model']['num_layers'],
        num_cams=2,
        num_heads=cfg['model']['num_heads'],
        in_channels=in_ch
    ).to(device)
    head = HeatmapHead(in_channels=cfg['model']['embed_dim']).to(device)

    state = torch.load(args.checkpoint, map_location='cpu')
    backbone.load_state_dict(state['backbone'])
    encoder.load_state_dict(state['encoder'])
    head.load_state_dict(state['head'])
    backbone.eval(); encoder.eval(); head.eval()

    x_range = cfg['bev']['x_range']
    y_range = cfg['bev']['y_range']
    resolution = float(cfg['bev']['resolution'])
    bev_h = int(cfg['bev']['bev_h'])
    bev_w = int(cfg['bev']['bev_w'])
    norm_factor = float(cfg['bev'].get('norm_factor', 1.0))

    dx_list = []
    dy_list = []
    dist_list = []

    with torch.no_grad():
        ex_saved = False
        for i, batch in enumerate(dl):
            sample_name = None
            if hasattr(ds, 'samples') and i < len(ds.samples):
                sample_name = os.path.basename(ds.samples[i].rstrip('/'))
            front = batch['front_img'].to(device)
            rear = batch['rear_img'].to(device)
            calib = batch['calib']
            flip = batch.get('flip', torch.zeros(front.shape[0], dtype=torch.int64)).to(device)
            calib_list = []
            for j in range(front.shape[0]):
                calib_list.append({
                    'front': {k: v[j].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['front'].items()},
                    'rear': {k: v[j].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['rear'].items()}
                })
            P = build_calib_tensor(calib_list).to(device)
            f_feats = backbone(front)
            r_feats = backbone(rear)
            feats = {
                'stride8': (f_feats['stride8'] + r_feats['stride8']) / 2.0,
                'stride16': (f_feats['stride16'] + r_feats['stride16']) / 2.0,
                'stride32': (f_feats['stride32'] + r_feats['stride32']) / 2.0
            }
            bev = encoder(feats, P, flip=flip)
            pred_c, pred_s = head(bev)
            cx_mm, cy_mm = centroid_from_heatmaps(pred_c[0], x_range, y_range, resolution)
            sx_mm, sy_mm = centroid_from_heatmaps(pred_s[0], x_range, y_range, resolution)
            cx_px, cy_px = centroid_pixel(pred_c[0])
            sx_px, sy_px = centroid_pixel(pred_s[0])
            if 'bev_heatmap' in batch:
                gt = batch['bev_heatmap'][0].to(device)
                gt_c = gt[:4]
                gt_s = gt[4:]
                gcx_px, gcy_px = centroid_pixel(gt_c)
                gsx_px, gsy_px = centroid_pixel(gt_s)
                pred_pixel = (sx_px, sy_px)
                gt_pixel = (gsx_px, gsy_px)
            else:
                pred_pixel = (sx_px, sy_px)
                gt_pixel = (float('nan'), float('nan'))
            # 反归一化到真实像素坐标（若存在归一化系数）
            real_pred_x = (pred_pixel[0] / norm_factor * (bev_w - 1)) if norm_factor != 1.0 else pred_pixel[0]
            real_pred_y = (pred_pixel[1] / norm_factor * (bev_h - 1)) if norm_factor != 1.0 else pred_pixel[1]
            if not np.isnan(gt_pixel[0]):
                real_gt_x = (gt_pixel[0] / norm_factor * (bev_w - 1)) if norm_factor != 1.0 else gt_pixel[0]
                real_gt_y = (gt_pixel[1] / norm_factor * (bev_h - 1)) if norm_factor != 1.0 else gt_pixel[1]
            else:
                real_gt_x = float('nan'); real_gt_y = float('nan')
            dx_pixel = real_pred_x - (real_gt_x if not np.isnan(real_gt_x) else cx_px)
            dy_pixel = real_pred_y - (real_gt_y if not np.isnan(real_gt_y) else cy_px)
            dx = dx_pixel * resolution
            dy = dy_pixel * resolution
            dist = float(np.sqrt(dx * dx + dy * dy))
            dx_list.append(dx)
            dy_list.append(dy)
            dist_list.append(dist)
            if i < 5:
                prefix = f'[{i}]'
                if sample_name is not None:
                    prefix = f'[{i} {sample_name}]'
                print(prefix, f'dx_mm={dx:.3f}', f'dy_mm={dy:.3f}', f'dist_mm={dist:.3f}')
                print(f"Sample {sample_name}:")
                print(f"  pred_center_pixel: {pred_pixel}")
                print(f"  gt_center_pixel: {gt_pixel}")
                print(f"  pixel_dx: {dx_pixel:.4f}, pixel_dy: {dy_pixel:.4f}")
                print(f"  BEV_resolution_mm_per_pixel: {resolution}")
                print(f"  physical_dx_mm: {dx:.4f}, physical_dy_mm: {dy:.4f}, dist_mm: {dist:.4f}")

            if not ex_saved:
                hc = pred_c[0].sum(dim=0).detach().cpu().numpy()
                hs = pred_s[0].sum(dim=0).detach().cpu().numpy()
                plt.figure(figsize=(10,4))
                plt.subplot(1,2,1); plt.imshow(hc, cmap='hot'); plt.title('container')
                plt.subplot(1,2,2); plt.imshow(hs, cmap='hot'); plt.title('spreader')
                plt.tight_layout()
                plt.savefig('test_results/bev_heatmap.png', dpi=200)
                plt.close()
                ex_saved = True

    dx_arr = np.array(dx_list)
    dy_arr = np.array(dy_list)
    dist_arr = np.array(dist_list)
    np.save('test_results/bias.npy', {'dx_mm': dx_arr, 'dy_mm': dy_arr, 'dist_mm': dist_arr})

    plt.figure(figsize=(8,4))
    plt.plot(np.arange(len(dx_arr)), dx_arr / 10.0, label='ΔX (cm)')
    plt.plot(np.arange(len(dy_arr)), dy_arr / 10.0, label='ΔY (cm)')
    plt.xlabel('frame'); plt.ylabel('bias (cm)'); plt.legend(); plt.tight_layout()
    plt.savefig('test_results/bias_curve.png', dpi=200)
    plt.close()

    mean_err = float(dist_arr.mean()) if len(dist_arr) > 0 else 0.0
    std_err = float(dist_arr.std()) if len(dist_arr) > 0 else 0.0
    # Multi-threshold success rates (stricter than legacy 10mm metric)
    threshold_candidates = [1.0, 2.0, 5.0, 10.0]
    success_by_threshold = {}
    if len(dist_arr) > 0:
        for t in threshold_candidates:
            success_by_threshold[f"success_rate_{int(t) if t.is_integer() else t}mm"] = float((dist_arr < t).mean())
    else:
        for t in threshold_candidates:
            success_by_threshold[f"success_rate_{int(t) if t.is_integer() else t}mm"] = 0.0

    # Choose the strictest threshold that still provides robust pass rate.
    # Preference order: smallest threshold with >=95% pass rate; fallback to max pass-rate threshold.
    recommended_threshold = None
    for t in [1.0, 2.0, 5.0]:
        key = f"success_rate_{int(t)}mm"
        if success_by_threshold.get(key, 0.0) >= 0.95:
            recommended_threshold = t
            break
    if recommended_threshold is None:
        ranked = sorted(
            [(t, success_by_threshold[f"success_rate_{int(t) if t.is_integer() else t}mm"]) for t in threshold_candidates],
            key=lambda x: (x[1], -x[0]),
            reverse=True
        )
        recommended_threshold = ranked[0][0]

    metrics = {
        'samples': int(len(dist_arr)),
        'mean_mm': mean_err,
        'std_mm': std_err,
        'recommended_threshold_mm': float(recommended_threshold),
    }
    metrics.update(success_by_threshold)

    with open('test_results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('mean_mm:', mean_err)
    print('std_mm:', std_err)
    print('success_rate_1mm:', success_by_threshold['success_rate_1mm'])
    print('success_rate_2mm:', success_by_threshold['success_rate_2mm'])
    print('success_rate_5mm:', success_by_threshold['success_rate_5mm'])
    print('success_rate_10mm:', success_by_threshold['success_rate_10mm'])
    print('recommended_threshold_mm:', recommended_threshold)
    print('results_dir:', os.path.abspath('test_results'))

if __name__ == '__main__':
    main()
