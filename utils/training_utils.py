import torch
import os
from datetime import datetime

def chamfer_loss(T_pred, T_gt):
    """
    Symmetric Chamfer distance between two point sets.
    T_pred: [K,3], T_gt: [M,3]
    """
    d2 = torch.cdist(T_pred, T_gt, p=2)  
    return d2.min(dim=1)[0].mean() + d2.min(dim=0)[0].mean()

def save_model(state_dict: dict, file_path: str, timestamp: bool = False) -> str:
    """
    Save a PyTorch model state_dict to disk, optionally appending a timestamp.

    Args:
        state_dict (dict): The model.state_dict() to save.
        file_path (str): The target path, e.g. "checkpoints/jointnet.pt".
        timestamp (bool): If True, append "_YYYYMMDD-HHMMSS" before the file extension.

    Returns:
        str: The actual path the model was saved to.
    """
    # Split base and extension
    base, ext = os.path.splitext(file_path)
    if timestamp:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f"{base}_{now}{ext}"
    else:
        save_path = file_path

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the state dict
    torch.save(state_dict, save_path)
    return save_path

def dict_to_device(d: dict, device: torch.device) -> dict:
    """
    Move all tensors (or lists of tensors) in d to the specified device.

    Args:
        d: A dict whose values may be torch.Tensor or list[torch.Tensor]
        device: e.g. "cpu" or "cuda"

    Returns:
        The same dict with all tensors relocated in-place.
    """
    for k, v in d.items():
        # single tensor
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
        # list of tensors
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, torch.Tensor):
                    new_list.append(item.to(device))
                else:
                    new_list.append(item)
            d[k] = new_list
    return d
