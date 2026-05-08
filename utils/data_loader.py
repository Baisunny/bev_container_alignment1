import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations import ReplayCompose
import torch.nn.functional as F

class ContainerDataset(Dataset):
    def __init__(self, split, config, augment=True):
        """
        Args:
            split: 'train' or 'val'
            config: dict containing 'camera_params_path', 'bev', etc.
        """
        self.split = split
        self.config = config
        root = config.get('data_root', 'data/processed')
        split_dir = os.path.join(root, split)
        self.data_dir = split_dir if os.path.isdir(split_dir) else root
        self.augment = augment
        
        # 假设数据组织为 {data_root}/{split}/sample_xxx/front.jpg, rear.jpg, calib.json, label.npy
        self.samples = []
        if os.path.exists(self.data_dir):
            for sample_name in sorted(os.listdir(self.data_dir)):
                sample_path = os.path.join(self.data_dir, sample_name)
                if os.path.isdir(sample_path):
                    self.samples.append(sample_path)
        
        # 根据配置中的样本列表进行过滤（可选）
        splits_cfg = config.get('splits', {})
        wanted = None
        if split == 'train':
            wanted = splits_cfg.get('train_samples')
        elif split == 'val':
            wanted = splits_cfg.get('val_samples')
        elif split == 'test':
            wanted = splits_cfg.get('test_samples')
        if isinstance(wanted, list) and len(wanted) > 0:
            name_set = set(wanted)
            filtered = []
            for p in self.samples:
                nm = os.path.basename(p.rstrip('/'))
                if nm in name_set:
                    filtered.append(p)
            self.samples = filtered
        
        self.img_size = (640, 480) # W, H
        self.photometric_aug = ReplayCompose([
            A.RandomBrightnessContrast(p=0.8),
            A.HueSaturationValue(p=0.8),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.5),
            A.HorizontalFlip(p=0.5)
        ])

    def __len__(self):
        return len(self.samples)

    def _preprocess_img(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            # 返回全零图作为 fallback
            return torch.zeros(3, self.img_size[1], self.img_size[0])
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        if self.augment:
            aug1 = self.photometric_aug(image=img)
            img = aug1["image"]
        # 归一化并转为 tensor (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        
        # 1. 加载图像
        front_img_path = os.path.join(sample_path, 'front.jpg')
        rear_img_path = os.path.join(sample_path, 'rear.jpg')
        
        img_f = cv2.imread(front_img_path)
        img_r = cv2.imread(rear_img_path)

        if img_f is not None:
            img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            img_f = cv2.resize(img_f, self.img_size)
        if img_r is not None:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            img_r = cv2.resize(img_r, self.img_size)

        flip_flag = False
        if self.augment and img_f is not None and img_r is not None:
            replay = self.photometric_aug(image=img_f)
            img_f = replay["image"]
            img_r = self.photometric_aug.replay(replay["replay"], image=img_r)["image"]
            for t in replay["replay"].get("transforms", []):
                class_name = t.get("__class_fullname__", "") or t.get("transform", "")
                if "HorizontalFlip" in class_name and t.get("applied", False):
                    flip_flag = True

        if img_f is None:
            front_img = torch.zeros(3, self.img_size[1], self.img_size[0])
        else:
            front_img = torch.from_numpy(img_f.astype(np.float32) / 255.0).permute(2, 0, 1)

        if img_r is None:
            rear_img = torch.zeros(3, self.img_size[1], self.img_size[0])
        else:
            rear_img = torch.from_numpy(img_r.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # 2. 加载 BEV 热图标签
        heatmap_path = os.path.join(sample_path, 'label.npy')
        if not os.path.exists(heatmap_path):
            heatmap_path = os.path.join(sample_path, 'heatmap.npy')
        strict_label = bool(self.config.get('data', {}).get('strict_label_loading', True))
        if os.path.exists(heatmap_path):
            try:
                bev_heatmap = np.load(heatmap_path).astype(np.float32)
                # 若发生水平翻转，沿宽度方向镜像每个通道
                if flip_flag:
                    bev_heatmap = bev_heatmap[:, :, ::-1].copy()
                bev_heatmap = torch.from_numpy(bev_heatmap)
                bev_h = self.config['bev']['bev_h']
                bev_w = self.config['bev']['bev_w']
                if bev_heatmap.shape[-2] != bev_h or bev_heatmap.shape[-1] != bev_w:
                    bev_heatmap = F.interpolate(bev_heatmap.unsqueeze(0), size=(bev_h, bev_w), mode='nearest').squeeze(0)
            except Exception as e:
                if strict_label:
                    raise RuntimeError(f"Failed to load label file: {heatmap_path}. Original error: {e}")
                bev_h = self.config['bev']['bev_h']
                bev_w = self.config['bev']['bev_w']
                bev_heatmap = torch.zeros(8, bev_h, bev_w)
        else:
            bev_h = self.config['bev']['bev_h']
            bev_w = self.config['bev']['bev_w']
            bev_heatmap = torch.zeros(8, bev_h, bev_w)
            
        # 3. 相机参数 calib.json
        calib_json_path = os.path.join(sample_path, 'calib.json')
        if os.path.exists(calib_json_path):
            with open(calib_json_path, 'r') as f:
                c = json.load(f)
            calib = {
                'front': {
                    'K': torch.tensor(c['front']['K'], dtype=torch.float32),
                    'R': torch.tensor(c['front']['R'], dtype=torch.float32),
                    'T': torch.tensor(c['front']['T'], dtype=torch.float32)
                },
                'rear': {
                    'K': torch.tensor(c['rear']['K'], dtype=torch.float32),
                    'R': torch.tensor(c['rear']['R'], dtype=torch.float32),
                    'T': torch.tensor(c['rear']['T'], dtype=torch.float32)
                }
            }
        else:
            calib = {
                'front': {
                    'K': torch.eye(3, dtype=torch.float32),
                    'R': torch.eye(3, dtype=torch.float32),
                    'T': torch.zeros(3, 1, dtype=torch.float32)
                },
                'rear': {
                    'K': torch.eye(3, dtype=torch.float32),
                    'R': torch.eye(3, dtype=torch.float32),
                    'T': torch.zeros(3, 1, dtype=torch.float32)
                }
            }
        
        return {
            'front_img': front_img,
            'rear_img': rear_img,
            'bev_heatmap': bev_heatmap,
            'calib': calib,
            'flip': torch.tensor(1 if flip_flag else 0, dtype=torch.int64)
        }
