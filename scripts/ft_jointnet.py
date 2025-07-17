import sys
sys.path.insert(1, '/Users/govind/Documents/cs/tamu/csce685-rignet/utils')

import argparse
import pickle
import time
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from training_utils import chamfer_loss
from dataset import RigNetDataset
from models import JointNet

def disp_head_and_overall_losses(
    model: JointNet,
    verts,
    one_ring,
    geodesic,
    joints_gt
):
    # pre-cluster loss
    q_pred = verts + model.disp_head(verts, one_ring, geodesic)
    L_disp  = chamfer_loss(q_pred, joints_gt)

    # clustered joints
    joints_pred = model(verts, one_ring, geodesic)
    L_joint = chamfer_loss(joints_pred, joints_gt)

    return L_disp, L_joint

# Validation
def validate(model: JointNet, dataset, device="cpu"):
    """
    Compute avg total loss on dataset without backprop.
    """
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Validating"):
            G = dataset[i]
            verts = G['vertices'].to(device)
            one_ring = G['one_ring'].to(device)
            geodesic = G['geodesic'].to(device)
            joints_gt = G['joints'].to(device)

            L_disp, L_joint = disp_head_and_overall_losses(model, verts, one_ring, geodesic, joints_gt)

            total += (L_disp + L_joint).item()
    return total / len(dataset)

# Training
def train(model: JointNet, 
          optimizer, 
          train_ds, 
          val_ds=None,
          epochs=10, 
          device="cpu", 
          logdir="runs/jointnet"):
    
    model.to(device)
    writer = SummaryWriter(logdir)

    for epoch in range(1, epochs+1):
        model.train()
        running_total = 0.0
        running_disp  = 0.0
        running_joint = 0.0

        for i in tqdm(range(len(train_ds)), desc=f"Epoch {epoch}/{epochs}"):
            G = train_ds[i]
            verts = G['vertices'].to(device)
            one_ring = G['one_ring'].to(device)
            geodesic = G['geodesic'].to(device)
            joints_gt = G['joints'].to(device)

            L_disp, L_joint = disp_head_and_overall_losses(model, verts, one_ring, geodesic, joints_gt)
            loss = L_disp + L_joint

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_total += loss.item()
            running_disp  += L_disp.item()
            running_joint += L_joint.item()

            step = (epoch-1) * len(train_ds) + i

            # Write current averages
            writer.add_scalar("train/total_loss", loss.item(), step)
            writer.add_scalar("train/disp_loss",  L_disp.item(), step)
            writer.add_scalar("train/joint_loss", L_joint.item(), step)

        avg_total = running_total / len(train_ds)
        avg_disp  = running_disp  / len(train_ds)
        avg_joint = running_joint / len(train_ds)

        writer.add_scalar("train/epoch_total", avg_total, epoch)
        writer.add_scalar("train/epoch_disp",  avg_disp,  epoch)
        writer.add_scalar("train/epoch_joint", avg_joint, epoch)

        if val_ds is not None:
            avg_val = validate(model, val_ds, device)
            writer.add_scalar("val/epoch_total", avg_val, epoch)
            print(f"[Epoch {epoch}] train={avg_total:.4e} (disp {avg_disp:.4e}, joint {avg_joint:.4e}), val={avg_val:.4e}")
        else:
            print(f"[Epoch {epoch}] train={avg_total:.4e} (disp {avg_disp:.4e}, joint {avg_joint:.4e})")

    writer.close()

# Argument Parsing and Main
def parse_args():
    p = argparse.ArgumentParser("RigNet Joint-Prediction Training")
    p.add_argument("--train-path", type=str, required=False, help="training .pkl")
    p.add_argument("--val-path", type=str, help="validation .pkl")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logdir", type=str, default="runs/jointnet")
    p.add_argument("--config", type=str, help="optional YAML config")
    p.add_argument("--load-state", type=str, help="path to pretrained JointNet state_dict")

    return p.parse_args()

def main():
    args = parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            # cast numeric fields back to the right type
            if k in ("lr", "wd"):
                v = float(v)
            elif k in ("epochs",):
                v = int(v)
            setattr(args, k, v)

    # Data
    train_ds = RigNetDataset(args.train_path)
    val_ds   = RigNetDataset(args.val_path) if args.val_path else None

    # Model
    jointnet = JointNet(is_training=True)
    if args.load_state:
        sd = torch.load(args.load_state, map_location="cpu")
        jointnet.load_state_dict(sd, strict=False)

    # Optimizer (all params)
    optimizer = torch.optim.AdamW(
        jointnet.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    # Logdir
    ts = time.strftime("%Y%m%d-%H%M%S")
    logdir = f"{args.logdir}_lr{args.lr}_wd{args.wd}_{ts}"

    # Train
    train(jointnet, 
          optimizer,
          train_ds, 
          val_ds,
          epochs=args.epochs,
          device=args.device,
          logdir=logdir)

    # Save final checkpoint
    torch.save(jointnet.state_dict(), f"{logdir}/jointnet_final.pt")

if __name__ == "__main__":
    main()
