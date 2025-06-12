import torch

def chamfer_loss(T_pred, T_gt):
    """
    Symmetric Chamfer distance between two point sets.
    T_pred: [K,3], T_gt: [M,3]
    """
    d2 = torch.cdist(T_pred, T_gt, p=2)  
    return d2.min(dim=1)[0].mean() + d2.min(dim=0)[0].mean()