import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from configs.config import Config
from utils.calib_utils import Calibration

class BEVDataset(Dataset):
    def __init__(self, coco_file, front_dir, rear_dir, calib_file, transform=None):
        self.front_dir = front_dir
        self.rear_dir = rear_dir
        self.transform = transform
        self.calib = Calibration(calib_file)
        
        # Load COCO
        with open(coco_file, 'r') as f:
            self.coco = json.load(f)
        
        # Build mapping from original name to annotations
        self.name_to_anno = {}
        for img_info in self.coco['images']:
            orig_name = img_info['extra']['name']
            img_id = img_info['id']
            # Find annotations for this image
            annos = [a for a in self.coco['annotations'] if a['image_id'] == img_id]
            self.name_to_anno[orig_name] = annos

        # Pair images (Assume they are sorted and matched 1:1)
        front_files = sorted([f for f in os.listdir(front_dir) if f.endswith('.bmp')])
        rear_files = sorted([f for f in os.listdir(rear_dir) if f.endswith('.bmp')])
        
        self.pairs = []
        for f, r in zip(front_files, rear_files):
            self.pairs.append((f, r))

    def __len__(self):
        return len(self.pairs)

    def _generate_heatmap(self, keypoints_bev):
        """
        keypoints_bev: (8, 2) - [u, v] coordinates in BEV grid
        Returns: (8, H, W) heatmap
        """
        h, w = Config.bev_h, Config.bev_w
        heatmap = np.zeros((8, h, w), dtype=np.float32)
        sigma = 2.0
        
        for i, (u, v) in enumerate(keypoints_bev):
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            
            # Draw Gaussian
            grid_y, grid_x = np.mgrid[0:h, 0:w]
            dist = (grid_x - u)**2 + (grid_y - v)**2
            heatmap[i] = np.exp(-dist / (2 * sigma**2))
            
        return heatmap

    def __getitem__(self, idx):
        front_name, rear_name = self.pairs[idx]
        
        # Load images
        img_f = cv2.imread(os.path.join(self.front_dir, front_name))
        img_r = cv2.imread(os.path.join(self.rear_dir, rear_name))
        
        img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        
        # Resize to input shape
        img_f = cv2.resize(img_f, (Config.img_w, Config.img_h))
        img_r = cv2.resize(img_r, (Config.img_w, Config.img_h))
        
        # Convert to tensor
        img_f = torch.from_numpy(img_f).permute(2, 0, 1).float() / 255.0
        img_r = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
        
        # Extract 2D keypoints from front and rear
        annos_f = self.name_to_anno.get(front_name, [])
        annos_r = self.name_to_anno.get(rear_name, [])
        
        # Combine points to get 3D world points
        # For simplicity, use front camera back-projection if available
        # or use both if possible (triangulation)
        all_points_3d = np.zeros((8, 3)) # 8 points
        
        if annos_f:
            kp_2d_f = np.array(annos_f[0]['keypoints']).reshape(-1, 3)[:, :2]
            points_3d_f = self.calib.back_project_2d_to_3d(kp_2d_f, 'front', z_world=0)
            all_points_3d = points_3d_f
        elif annos_r:
            kp_2d_r = np.array(annos_r[0]['keypoints']).reshape(-1, 3)[:, :2]
            points_3d_r = self.calib.back_project_2d_to_3d(kp_2d_r, 'rear', z_world=0)
            all_points_3d = points_3d_r
            
        # Project 3D points to BEV grid
        keypoints_bev = self.calib.project_3d_to_bev(all_points_3d, Config.x_range, Config.y_range, Config.bev_resolution)
        
        # Generate heatmaps
        target_heatmap = self._generate_heatmap(keypoints_bev)
        target_heatmap = torch.from_numpy(target_heatmap).float()
        
        # Prepare projection matrices
        P_f = torch.from_numpy(self.calib.front['P']).float()
        P_r = torch.from_numpy(self.calib.rear['P']).float()
        
        return {
            'img': torch.stack([img_f, img_r], dim=0), # (2, 3, H, W)
            'heatmap': target_heatmap,
            'calibs': torch.stack([P_f, P_r], dim=0) # (2, 3, 4)
        }
