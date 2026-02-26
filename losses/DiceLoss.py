# author: Ibrahim 
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, preds, targets):
        """
        preds: (N, H, W) or (N, ) probabilities in [0,1]
        targets: same shape, binary {0,1}
        """
        preds = preds.view(preds.size(0), -1).float()
        targets = targets.view(targets.size(0), -1).float()
        intersection = (preds * targets).sum(1)
        denom = preds.sum(1) + targets.sum(1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean()

class MultiClassDiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, include_background=True, smooth=1.0):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self.smooth = float(smooth)

    def forward(self, logits, target):
        """
        logits: (N, C, H, W) raw logits
        target: (N, H, W) int64 with class indices
        """
        n, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)  # (N,C,H,W)
        target_onehot = F.one_hot(target.long(), num_classes=c).permute(0,3,1,2).float()  # (N,C,H,W)

        if not self.include_background:
            start = 1
        else:
            start = 0

        losses = []
        for cls in range(start, c):
            p = probs[:, cls]
            t = target_onehot[:, cls]
            dice = (2.0 * (p * t).sum(dim=(1,2)) + self.smooth) / (p.sum(dim=(1,2)) + t.sum(dim=(1,2)) + self.smooth)
            losses.append(1.0 - dice)  # per-sample
        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        loss = torch.stack(losses, dim=0).mean()  # mean over classes then samples
        return loss