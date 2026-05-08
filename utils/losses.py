import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Modified Focal Loss for heatmap prediction.
        pred, target: (B, C, H, W)
        """
        pos_mask = (target == 1).float()
        neg_mask = (target < 1).float()
        
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-6) * pos_mask
        neg_loss = -torch.pow(1 - target, self.beta) * torch.pow(pred, self.alpha) * torch.log(1 - pred + 1e-6) * neg_mask
        
        num_pos = pos_mask.sum()
        if num_pos == 0:
            return neg_loss.mean()
        
        return (pos_loss.sum() + neg_loss.sum()) / num_pos

class OffsetLoss(nn.Module):
    def __init__(self):
        super(OffsetLoss, self).__init__()

    def forward(self, pred, target, mask):
        """
        L1 loss for offset prediction.
        pred: (B, C*2, H, W)
        target: (B, C*2, H, W)
        mask: (B, C, H, W)
        """
        # mask should be 1 at the locations of ground truth keypoints
        # Repeat mask for x and y
        B, C, H, W = mask.shape
        mask = mask.unsqueeze(2).expand(B, C, 2, H, W).reshape(B, C*2, H, W)
        
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        num_pos = mask.sum()
        if num_pos == 0:
            return 0
        
        return loss / num_pos
