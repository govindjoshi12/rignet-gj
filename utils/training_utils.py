import torch
import os
from datetime import datetime
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, StepLR
import random
import yaml

from torch.serialization import safe_globals
from numpy._core.multiarray import _reconstruct
from numpy import dtype, ndarray
from numpy.dtypes import UInt32DType

def chamfer_loss(T_pred, T_gt):
    """
    Symmetric Chamfer distance between two point sets.
    T_pred: [K,3], T_gt: [M,3]
    """
    d2 = torch.cdist(T_pred, T_gt, p=2)  
    return d2.min(dim=1)[0].mean() + d2.min(dim=0)[0].mean()


def batch_multi_chamfer_loss(
    preds_list_of_lists: list[list[torch.Tensor]],
    gts_list:          list[torch.Tensor],
    device:            str = "cpu"
) -> list[torch.Tensor]:
    """
    Compute per-head average Chamfer loss over a batch of graphs.
    Useful for computing chamfer loss of displaced vertices and predicted joints
    simaltaneuosly or separately.

    Args:
        preds_list_of_lists: list of H heads, each a list of length M of [Ni, 3] Tensors
        gts_list:            list of M ground-truth [Ji, 3] Tensors
        device:              "cpu" or "cuda"

    Returns:
        List of H scalar losses, one per head.
    """
    H = len(preds_list_of_lists)
    M = len(gts_list)
    # Initialize per-head accumulators
    head_losses = [ [] for _ in range(H) ]

    for head_idx, preds_list in enumerate(preds_list_of_lists):
        for pred, gt in zip(preds_list, gts_list):
            p = pred.to(device)
            t = gt.to(device)
            head_losses[head_idx].append(chamfer_loss(p, t))

    # average each head’s list of losses
    return [ torch.stack(lst).mean() for lst in head_losses ]


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
    print("Saved to", save_path)

    return save_path


def save_checkpoint(path, model, optimizer, scheduler, epoch, args):
    """
    Save a full training checkpoint.

    This writes out everything needed to resume training exactly:
      - current epoch number
      - model weights
      - optimizer state (including moments)
      - scheduler state (wherever it is in its cycle)
      - all command line args / hyperparameters
      - RNG states for CPU, CUDA, NumPy, and Python `random`

    Args:
        path (str):
            Filesystem path to write the checkpoint (e.g. "latest.pt").
        model (torch.nn.Module):
            Your model instance whose `.state_dict()` you want to save.
        optimizer (torch.optim.Optimizer):
            The optimizer, to capture its internal state.
        scheduler (torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau):
            The LR scheduler, so you can keep stepping it after resume.
        epoch (int):
            The last completed epoch (so you'll resume at epoch+1).
        args (argparse.Namespace):
            Parsed command-line arguments; this will be saved via `vars(args)`.

    The checkpoint dict layout:
        {
            "epoch":   int,
            "model":   state_dict(),
            "optim":   optimizer.state_dict(),
            "sched":   scheduler.state_dict(),
            "args":    dict of all CLI args,
            "rng": {
                "cpu":   torch.get_rng_state(),
                "cuda":  torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "py":    random.getstate()
            }
        }
    """
    ckpt = {
        "epoch":    epoch,
        "model":    model.state_dict(),
        "optim":    optimizer.state_dict(),
        "sched":    scheduler.state_dict(),
        "args":     vars(args),
        "rng": {
            "cpu":   torch.get_rng_state(),
            "cuda":  torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "py":    random.getstate(),
        }
    }
    torch.save(ckpt, path)

def load_checkpoint(
    model,
    optimizer,
    scheduler ,
    checkpoint_path,
):
    """
    Reloads model, optimizer, scheduler, and determine start_epoch.

    Returns:
        model, optimizer, scheduler, start_epoch
    """

    start_epoch = 1
    with safe_globals([_reconstruct, ndarray, dtype, UInt32DType]):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        torch.set_rng_state(   ckpt["rng"]["cpu"])
        torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        np.random.set_state(   ckpt["rng"]["numpy"])
        random.setstate(       ckpt["rng"]["py"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Loaded '{checkpoint_path}', starting/resuming at epoch {start_epoch}")

    return model, optimizer, scheduler, start_epoch


def load_yaml_config(config_path, args):
    """
    Read a YAML file and override matching attributes on `args`.

    - `config_path` (str): path to a YAML file.
    - `args` (argparse.Namespace): your parsed CLI args.

    For each key/value in the YAML:
      • If `args` already has an attribute of that name, we cast the new value
        to the type of the existing default (unless the default was None, in
        which case we assign it directly).
      • If `args` does *not* have that attribute, we still set it on `args`.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    for k, v in cfg.items():
        if hasattr(args, k):
            default = getattr(args, k)
            # if default is None, just assign
            if default is None:
                setattr(args, k, v)
            else:
                try:
                    setattr(args, k, type(default)(v))
                except Exception:
                    # fallback if casting fails
                    setattr(args, k, v)
        else:
            # new key — just attach it
            setattr(args, k, v)

    return args


def make_scheduler(optimizer, kind="onecycle", kwargs=None):
    """
    Factory for learning rate schedulers.

    Given an optimizer and a string key, returns an initialized scheduler
    instance. The caller can then simply call `scheduler.step()` or
    `scheduler.step(metric)` for Plateau.

    Args:
        optimizer (torch.optim.Optimizer):
            The optimizer whose learning rate we want to schedule.
        kind (str):
            One of:
              - "onecycle" : torch.optim.lr_scheduler.OneCycleLR
              - "plateau"  : torch.optim.lr_scheduler.ReduceLROnPlateau
              - "step"     : torch.optim.lr_scheduler.StepLR
        kwargs (dict or None):
            Keyword arguments to pass into the scheduler’s constructor 
            (e.g. `{"max_lr":1e-3, "total_steps":1000}` for onecycle).

    Returns:
        An instance of the requested scheduler, already bound to `optimizer`.

    Raises:
        ValueError: if `kind` is not one of the recognized scheduler names.
    """
    if kwargs is None:
        kwargs = {}

    if kind == "onecycle":
        # OneCycleLR(max_lr=..., total_steps=..., pct_start=..., etc.)
        return OneCycleLR(optimizer, **kwargs)
    elif kind == "plateau":
        # ReduceLROnPlateau(mode='min' or 'max', factor=..., patience=..., etc.)
        return ReduceLROnPlateau(optimizer, **kwargs)
    elif kind == "step":
        # StepLR(step_size=..., gamma=..., etc.)
        return StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler kind '{kind}'; "
                         f"must be one of ['onecycle', 'plateau', 'step'].")


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


# Single Threshold PCK
def compute_pck(joints_pred_list, joints_gt_list, threshold: float) -> float:
    """
    Compute PCK@threshold across a dataset.

    Args:
        joints_pred_list: list of [Ki,3] Tensors (predicted joint sets)
        joints_gt_list:   list of [Ji,3] Tensors (ground-truth joints)
        threshold:        float distance threshold

    Returns:
        pck: fraction of GT joints with at least one pred within threshold
    """
    correct = 0
    total_gt = 0
    for jp, jg in zip(joints_pred_list, joints_gt_list):
        # pairwise distances [Ki, Ji]
        d2 = torch.cdist(jp, jg, p=2)
        min_dists, _ = d2.min(dim=0)   # for each GT joint
        correct   += (min_dists <= threshold).sum().item()
        total_gt  += jg.size(0)
    return correct / total_gt if total_gt > 0 else 0.0


# Multi‐threshold PCK curve
def compute_pck_curve(joints_pred_list, joints_gt_list, thresholds: np.ndarray) -> np.ndarray:
    """
    Compute PCK for each threshold in `thresholds`.

    Args:
        joints_pred_list: list of [Ki,3] Tensors
        joints_gt_list:   list of [Ji,3] Tensors
        thresholds:       1D numpy array of T thresholds

    Returns:
        pck_vals: length-T numpy array of PCK@each threshold
    """
    return np.array([
        compute_pck(joints_pred_list, joints_gt_list, th)
        for th in thresholds
    ])