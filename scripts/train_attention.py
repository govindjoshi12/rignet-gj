import sys
sys.path.insert(1, '/Users/govind/Documents/cs/tamu/csce685-rignet/utils')

import open3d as o3d
import numpy as np

import random
import pickle
from tqdm import tqdm
import argparse
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from dataset import RigNetDataset
from models import JointNet

# Validate
def validate(model, dataset, loss_fn, device="cpu"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc=f"Validating"):
            G = dataset[i]
            verts = G['vertices'].to(device)
            one_ring = G['one_ring'].to(device)
            geodesic = G['geodesic'].to(device)
            attn_mask = G['attn_mask'].to(device)
            logits = model(verts, one_ring, geodesic).view(-1)
            total_loss += loss_fn(logits, attn_mask).item()
    return total_loss / len(dataset)

# Train with Tensorboard Logging
def train(model: JointNet, optimizer, 
          train_ds, val_ds=None,
          epochs=10, loss_fn = nn.BCEWithLogitsLoss(),
          device="cpu", logdir="runs/attention"):
    
    model.to(device)
    writer = SummaryWriter(logdir)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for i in tqdm(range(len(train_ds)), desc=f"Epoch {epoch}/{epochs}"):
            G = train_ds[i]
            verts = G['vertices'].to(device)
            one_ring = G['one_ring'].to(device)
            geodesic = G['geodesic'].to(device)
            attn_mask = G['attn_mask'].to(device)

            logits = model(verts, one_ring, geodesic).view(-1)
            loss = loss_fn(logits, attn_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step = (epoch-1) * len(train_ds) + i
            writer.add_scalar("train/loss", loss.item(), step)

        avg_train = running_loss / len(train_ds)
        writer.add_scalar("train/epoch_loss", avg_train, epoch)

        if val_ds is not None:
            avg_val = validate(model, val_ds, loss_fn, device)
            writer.add_scalar("val/epoch_loss", avg_val, epoch)
            print(f"Epoch {epoch}: train loss = {avg_train:.4e}, val loss = {avg_val:.4e}")
        else:
            print(f"Epoch {epoch}: train loss = {avg_train:.4e}")

    writer.close()


def parse_args():
    p = argparse.ArgumentParser(description="Pretrain GMEdgeNet Attention Head")
    p.add_argument("--train-path", type=str, required=False,
                   help="Path to train .pkl file")
    p.add_argument("--val-path", type=str, required=False,
                   help="Path to validation .pkl file")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--logdir", type=str, default="runs/attention_pretrain")
    p.add_argument("--config", type=str,
                   help="Optional path to YAML config file")
    p.add_argument("--load-state", type=str, help="path to pretrained JointNet state_dict")
    
    return p.parse_args()

def main():
    args = parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
            # print(cfg)
        # override only fields present in cfg
        for k, v in cfg.items():
            # cast numeric fields back to the right type
            if k in ("lr", "wd"):
                v = float(v)
            elif k in ("epochs",):
                v = int(v)
            setattr(args, k, v)
                
    train_ds = RigNetDataset(args.train_path)
    val_ds   = RigNetDataset(args.val_path) if args.val_path else None

    # Model
    jointnet = JointNet(is_training=True)
    if args.load_state:
        sd = torch.load(args.load_state, map_location="cpu")
        jointnet.load_state_dict(sd, strict=False)
    attn_module = jointnet.attn_head

    optimizer = torch.optim.AdamW(
        params=attn_module.parameters(),
        lr=args.lr, 
        weight_decay=args.wd
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logdir = f"{args.logdir}_lr{args.lr}_wd{args.wd}_{timestamp}"
    print(logdir)

    train(
        attn_module,
        optimizer,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        device=args.device,
        logdir=logdir
    )

    # TODO: Save Model

    # Hyperparam search
    # lr=1e-4 is too high. Loss is continuously oscillating
    # lr=1e-5 is stable for longer but also begins to oscillate
    # lr=1e-6 more stable but also the decrease is much slower

    torch.save(jointnet.state_dict(), f"{logdir}/jointnet_final.pt")

if __name__ == "__main__":
    main()
