import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import BEVDataset
from models.detector import BEVFormerDetector
from utils.losses import GaussianFocalLoss, OffsetLoss
from configs.config import Config
import os

def train():
    # 1. Dataset & DataLoader
    dataset = BEVDataset(
        coco_file='data/annotations/_annotations.coco.json',
        front_dir='data/raw/images/front',
        rear_dir='data/raw/images/rear',
        calib_file='data/raw/camera_calib.yaml.txt'
    )
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    
    # 2. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BEVFormerDetector(embed_dim=Config.bev_dim, num_layers=Config.num_bev_layers).to(device)
    
    # 3. Loss & Optimizer
    criterion_hm = GaussianFocalLoss()
    criterion_off = OffsetLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    
    # 4. Training loop
    model.train()
    for epoch in range(Config.epochs):
        epoch_loss = 0
        for batch in dataloader:
            img = batch['img'].to(device) # (B, 2, 3, H, W)
            target_hm = batch['heatmap'].to(device) # (B, 8, H_bev, W_bev)
            calibs = batch['calibs'].to(device) # (B, 2, 3, 4)
            
            optimizer.zero_grad()
            
            # Forward
            pred_hm, pred_off = model(img, calibs)
            
            # Loss
            loss_hm = criterion_hm(pred_hm, target_hm)
            # loss_off = criterion_off(pred_off, target_off, target_hm) # Simplified
            
            loss = loss_hm
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch [{epoch+1}/{Config.epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/bevformer_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    train()
