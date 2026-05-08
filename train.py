import os
import argparse
import tempfile
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_loader import ContainerDataset
from models.backbone import ResNetBackbone
from models.bevformer import BEVFormerEncoder
from models.heads import HeatmapHead
from models.losses import GaussianFocalLoss

def build_calib_tensor(batch_calib):
    Ps = []
    for c in batch_calib:
        Kf = c['front']['K']; Rf = c['front']['R']; Tf = c['front']['T']
        Kr = c['rear']['K']; Rr = c['rear']['R']; Tr = c['rear']['T']
        Pf = torch.matmul(Kf, torch.cat([Rf, Tf], dim=1))
        Pr = torch.matmul(Kr, torch.cat([Rr, Tr], dim=1))
        Ps.append(torch.stack([Pf, Pr], dim=0))
    return torch.stack(Ps, dim=0)

def _safe_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path))
    os.close(tmp_fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--work_dir', type=str, default='work_dirs/exp1')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = ContainerDataset('train', cfg)
    val_set = ContainerDataset('val', cfg)

    num_workers = int(cfg['train'].get('num_workers', 0))
    dl_train = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(val_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=num_workers)

    backbone = ResNetBackbone(
        name=cfg['model']['backbone'],
        pretrained=bool(cfg['model'].get('pretrained', True)),
        freeze=bool(cfg['model'].get('freeze_backbone', True)),
        pretrained_path=cfg['model'].get('pretrained_path')
    ).to(device)
    in_ch = (backbone.out_channels['stride8'], backbone.out_channels['stride16'], backbone.out_channels['stride32'])
    encoder = BEVFormerEncoder(embed_dim=cfg['model']['embed_dim'], bev_h=cfg['bev']['bev_h'], bev_w=cfg['bev']['bev_w'], num_layers=cfg['model']['num_layers'], num_cams=2, num_heads=cfg['model']['num_heads'], in_channels=in_ch).to(device)
    head = HeatmapHead(in_channels=cfg['model']['embed_dim']).to(device)
    criterion = GaussianFocalLoss(args.config, gamma=2.0)

    params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=cfg['train']['lr'], weight_decay=cfg['train'].get('weight_decay', 0.05))
    if cfg['train'].get('lr_scheduler', 'cosine') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    else:
        scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    epochs = cfg['train']['epochs']
    save_every = cfg['train'].get('save_every', 5)
    patience = int(cfg['train'].get('patience', 20))
    best_val = float('inf')
    patience_left = patience

    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        if isinstance(state, dict):
            if 'backbone' in state:
                backbone.load_state_dict(state['backbone'])
            if 'encoder' in state:
                encoder.load_state_dict(state['encoder'])
            if 'head' in state:
                head.load_state_dict(state['head'])
            if 'optimizer' in state:
                optimizer.load_state_dict(state['optimizer'])
            if 'scheduler' in state:
                scheduler.load_state_dict(state['scheduler'])
            start_epoch = int(state.get('epoch', 0)) + 1
        backbone.to(device); encoder.to(device); head.to(device)

    os.makedirs(args.work_dir, exist_ok=True)
    batch_log_path = os.path.join(args.work_dir, 'batch_losses.csv')
    need_header = not os.path.exists(batch_log_path)
    batch_log_f = open(batch_log_path, 'a', newline='')
    batch_log_writer = csv.writer(batch_log_f)
    if need_header:
        batch_log_writer.writerow(['epoch', 'batch_idx', 'batch_loss'])

    for epoch in range(start_epoch, epochs + 1):
        backbone.train(); encoder.train(); head.train()
        tot_loss = 0.0; n_batches = 0
        batch_losses = []
        for batch_idx, batch in enumerate(dl_train):
            front = batch['front_img'].to(device)
            rear = batch['rear_img'].to(device)
            target = batch['bev_heatmap'].to(device)
            calib = batch['calib']
            flip = batch.get('flip', torch.zeros(front.shape[0], dtype=torch.int64)).to(device)
            calib_list = []
            for i in range(len(front)):
                calib_list.append({
                    'front': {k: v[i].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['front'].items()},
                    'rear': {k: v[i].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['rear'].items()}
                })
            P = build_calib_tensor(calib_list).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                f_feats = backbone(front)
                r_feats = backbone(rear)
                feats = {
                    'stride8': (f_feats['stride8'] + r_feats['stride8']) / 2.0,
                    'stride16': (f_feats['stride16'] + r_feats['stride16']) / 2.0,
                    'stride32': (f_feats['stride32'] + r_feats['stride32']) / 2.0
                }
                bev = encoder(feats, P, flip=flip)
                pred_c, pred_s = head(bev)
                gt_c = target[:, :4]
                gt_s = target[:, 4:]
                loss = criterion(pred_c, gt_c, pred_s, gt_s)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss = float(loss.item())
            batch_losses.append(batch_loss)
            batch_log_writer.writerow([epoch, batch_idx, f'{batch_loss:.8f}'])
            tot_loss += batch_loss; n_batches += 1
        batch_log_f.flush()
        scheduler.step()
        avg_loss = tot_loss / max(1, n_batches)
        if len(batch_losses) > 0:
            bt = torch.tensor(batch_losses, dtype=torch.float32)
            std_loss = float(bt.std(unbiased=False).item())
            min_loss = float(bt.min().item())
            max_loss = float(bt.max().item())
        else:
            std_loss = 0.0
            min_loss = 0.0
            max_loss = 0.0
        print(
            f'Epoch {epoch}/{epochs} Train Loss: {avg_loss:.6f} '
            f'(std={std_loss:.6f}, min={min_loss:.6f}, max={max_loss:.6f}, batches={n_batches})'
        )

        if epoch % save_every == 0:
            backbone.eval(); encoder.eval(); head.eval()
            eval_set = ContainerDataset('val', cfg, augment=False)
            dl_eval = DataLoader(eval_set, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=num_workers)
            eval_loss = 0.0; e_batches = 0
            with torch.no_grad():
                for batch in dl_eval:
                    front = batch['front_img'].to(device)
                    rear = batch['rear_img'].to(device)
                    target = batch['bev_heatmap'].to(device)
                    calib = batch['calib']
                    flip = batch.get('flip', torch.zeros(front.shape[0], dtype=torch.int64)).to(device)
                    calib_list = []
                    for i in range(len(front)):
                        calib_list.append({
                            'front': {k: v[i].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['front'].items()},
                            'rear': {k: v[i].to(device) if isinstance(v, torch.Tensor) else torch.from_numpy(v).to(device).float() for k, v in calib['rear'].items()}
                        })
                    P = build_calib_tensor(calib_list).to(device)
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        f_feats = backbone(front)
                        r_feats = backbone(rear)
                        feats = {
                            'stride8': (f_feats['stride8'] + r_feats['stride8']) / 2.0,
                            'stride16': (f_feats['stride16'] + r_feats['stride16']) / 2.0,
                            'stride32': (f_feats['stride32'] + r_feats['stride32']) / 2.0
                        }
                        bev = encoder(feats, P, flip=flip)
                        pred_c, pred_s = head(bev)
                        gt_c = target[:, :4]
                        gt_s = target[:, 4:]
                        l = criterion(pred_c, gt_c, pred_s, gt_s)
                    eval_loss += l.item(); e_batches += 1
            avg_eval = eval_loss / max(1, e_batches)
            print(f'Epoch {epoch}/{epochs} Train-Eval Loss: {avg_eval:.4f}')
            if avg_eval < best_val:
                best_val = avg_eval
                ckpt_dir = os.path.join(args.work_dir, 'checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                ok = _safe_torch_save({
                    'epoch': epoch,
                    'backbone': backbone.state_dict(),
                    'encoder': encoder.state_dict(),
                    'head': head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(ckpt_dir, 'best.pth'))
                if not ok:
                    print('Warning: failed to save best checkpoint.')
                patience_left = patience
            else:
                patience_left -= save_every
                if patience_left <= 0:
                    print('Early stopping.')
                    break

        if epoch % save_every == 0:
            ckpt_dir = os.path.join(args.work_dir, 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ok = _safe_torch_save({
                'epoch': epoch,
                'backbone': backbone.state_dict(),
                'encoder': encoder.state_dict(),
                'head': head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(ckpt_dir, f'bevformer_{epoch}.pth'))
            if not ok:
                print('Warning: failed to save periodic checkpoint.')

    batch_log_f.close()

if __name__ == '__main__':
    main()
