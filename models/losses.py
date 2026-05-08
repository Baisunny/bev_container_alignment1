import torch
import torch.nn as nn
import yaml

class GaussianFocalLoss(nn.Module):
    def __init__(self, config_path='configs/default.yaml', gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.lambda_container = 1.0
        self.lambda_spreader = 1.0
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                loss_cfg = cfg.get('loss', {})
                self.lambda_container = float(loss_cfg.get('lambda_container', 1.0))
                self.lambda_spreader = float(loss_cfg.get('lambda_spreader', 1.0))
        except Exception:
            pass

    def _loss_branch(self, pred, target):
        eps = 1e-6
        pos = -torch.pow(1 - pred, self.gamma) * target * torch.log(pred.clamp(min=eps))
        neg = -torch.pow(pred, self.gamma) * (1 - target) * torch.log((1 - pred).clamp(min=eps))
        return (pos + neg).mean()

    def forward(self, pred_container, target_container, pred_spreader, target_spreader):
        lc = self._loss_branch(pred_container, target_container)
        ls = self._loss_branch(pred_spreader, target_spreader)
        return self.lambda_container * lc + self.lambda_spreader * ls
