import sys
sys.path.insert(1, '../utils')

import os
import time
import argparse
import yaml
import random
import traceback

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from dataset import RigNetDataset, collate_fn, FILE_PATHS
from models import JointDisplacementModule
from training_utils import (
    dict_to_device,
    batch_multi_chamfer_loss,
    compute_pck_curve,
    save_checkpoint,
    load_checkpoint,
    load_yaml_config
)

def validate_disp_head(model, dl, device, thresholds, writer=None, epoch=None):
    """
    Run validation for the displacement head.

    - Computes average Chamfer loss over all batches.
    - Computes batch-wise average PCK at each threshold.
    - Computes AUC under the PCK curve.
    - Logs scalars to TensorBoard if writer+epoch provided.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    thresholds = np.array(thresholds)
    pck_sums   = np.zeros_like(thresholds, dtype=float)

    with torch.no_grad():
        for batch in tqdm(dl, desc="Validating disp head"):
            # Move all tensors in the batch to the target device
            batch = dict_to_device(batch, device)

            # Forward pass: predict displacement deltas
            disp   = model(
                batch['vertices'],
                batch['one_ring'],
                batch['geodesic']
            )
            q_pred = batch['vertices'] + disp  # displaced vertices

            # Split the concatenated q_pred back into per-graph lists
            verts_per_graph = batch['vertices_per_graph'].tolist()
            pred_lists = torch.split(q_pred, verts_per_graph, dim=0)
            gt_lists   = batch['joints_list']

            # Compute Chamfer loss via the multi-head utility (only head 0)
            loss_disp = batch_multi_chamfer_loss(
                [pred_lists],  # list of H=1 lists
                gt_lists,
                device=device
            )[0]
            total_loss += loss_disp.item()

            # Compute PCK curve for this batch
            batch_pcks = compute_pck_curve(pred_lists, gt_lists, thresholds)
            pck_sums  += batch_pcks

            n_batches += 1

            del batch, disp, q_pred, verts_per_graph, pred_lists, gt_lists, loss_disp, batch_pcks
            torch.cuda.empty_cache()

    # Average metrics over all batches
    avg_loss    = total_loss / n_batches
    avg_pcks    = pck_sums / n_batches
    auc         = np.trapezoid(avg_pcks, thresholds)

    # Print summary to console
    print(f"[Validate] chamfer_loss={avg_loss:.4e}")
    for th, pck in zip(thresholds, avg_pcks):
        print(f"  PCK@{th:.3f} = {pck*100:.1f}%")
    print(f"  AUC(0-{thresholds[-1]:.3f}) = {auc:.4f}")

    # Log to TensorBoard if requested
    if writer is not None and epoch is not None:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        for i, th in enumerate(thresholds):
            writer.add_scalar(f"val/PCK@{th:.3f}", avg_pcks[i], epoch)
        writer.add_scalar("val/AUC", auc, epoch)

    return avg_loss, avg_pcks, auc


def train_disp_head(
    model,
    optimizer,
    scheduler,
    train_dl,
    val_dl,
    device,
    writer,
    start_epochs,
    max_epochs,
    thresholds,
    args,
    save_every,
    best_loss = float('inf'),
    best_auc = float('-inf')
):
    """
    Training loop for the displacement head, saving full checkpoints:

    - Every `save_every` epochs: overwrite rolling checkpoint via save_checkpoint.
    - Whenever val loss improves: save_checkpoint("best_loss.pt").
    - Whenever val AUC improves: save_checkpoint("best_auc.pt").
    """

    scaler = GradScaler()

    for epoch in range(start_epochs, max_epochs+1):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{max_epochs}")):
            batch = dict_to_device(batch, device)

            optimizer.zero_grad()
            # ------ AMP Mixed Precision -------
            with autocast(device_type=str(device)):
                disp   = model(
                    batch['vertices'],
                    batch['one_ring'],
                    batch['geodesic']
                )
                q_pred = batch['vertices'] + disp

                verts_per_graph = batch['vertices_per_graph'].tolist()
                pred_lists = torch.split(q_pred, verts_per_graph, dim=0)

                loss_disp = batch_multi_chamfer_loss(
                    [pred_lists],
                    batch['joints_list'],
                    device=device
                )[0]

            scaler.scale(loss_disp).backward()
            scaler.step(optimizer)
            scaler.update()
            # ---------------------------------

            # Step the LR scheduler
            scheduler.step()
            torch.cuda.empty_cache()

            running_loss += loss_disp.item()
            step = (epoch-1) * len(train_dl) + i
            writer.add_scalar("train/batch_loss", loss_disp.item(), step)

        avg_train = running_loss / len(train_dl)
        writer.add_scalar("train/epoch_loss", avg_train, epoch)
        print(f"[Epoch {epoch}] train_loss={avg_train:.4e}")

        # Validation
        if val_dl is not None:
            val_loss, val_pcks, val_auc = validate_disp_head(
                model, val_dl, device, thresholds, writer, epoch
            )

            # Best‐loss checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(writer.log_dir, "best_loss.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch, args
                )
                print(f"New best val_loss={best_loss:.4e}, checkpointed")

            # Best‐AUC checkpoint
            if val_auc > best_auc:
                best_auc = val_auc
                ckpt_path = os.path.join(writer.log_dir, "best_auc.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch, args
                )
                print(f"New best AUC={best_auc:.4f}, checkpointed")

        # Rolling checkpoint every save_every epochs
        if epoch % save_every == 0:
            ckpt_path = os.path.join(writer.log_dir, f"latest.pt")
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch, args
            )
            print(f"Saved rolling checkpoint at epoch {epoch}")

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser("Train RigNet Displacement Head")
    p.add_argument("--train-path",   type=str, default=FILE_PATHS['train'])
    p.add_argument("--val-path",     type=str, default=FILE_PATHS['val'])
    p.add_argument("--epochs",       type=int, default=1000)
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--wd",           type=float, default=1e-6)
    p.add_argument("--edge-dropout", type=int, default=15)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--logdir",       type=str, default="runs/disp_pretrain")
    p.add_argument("--checkpoint",   type=str, default=None, help="Path to full training checkpoint")
    p.add_argument("--save-every",   type=str, help="Save checkpoint every N epochs")
    p.add_argument("--thresholds",   nargs="+", type=float,
                   default=np.arange(0.0, 0.21, 0.0025))
    p.add_argument("--config",       type=str)
    return p.parse_args()


def main():
    args = parse_args()

    # override via YAML if provided
    if args.config:
        load_yaml_config(args.config, args)

    # determine device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # data loaders
    train_ds = RigNetDataset(args.train_path)
    train_dl = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=args.num_workers)
    val_dl = None
    if args.val_path:
        val_ds = RigNetDataset(args.val_path)
        val_dl = DataLoader(val_ds,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=collate_fn)

    # TensorBoard setup
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(args.logdir,
                          f"lr{args.lr}_wd{args.wd}_{timestamp}")
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    print("Logging to", logdir)
    
    # Initial setup
    steps_per_epoch = len(train_dl)
    print("edge dropout: %d" % args.edge_dropout)
    model = JointDisplacementModule(edge_dropout=args.edge_dropout).to(device)
    
    # TODO: supply other optimizers and LR Schedulers
    # Even if these get initialized with default values (not specified by user)
    # load_checkpoint will override them if checkpoint is provided
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch
    )
    current_epoch = 1

    if args.checkpoint:
        model, optimizer, scheduler, current_epoch = load_checkpoint(
            model,
            optimizer,
            scheduler,
            args.checkpoint
        )
        
    best_loss = float('inf')
    best_auc  = -float('inf')

    print(current_epoch)

    # Retry loop in case of crash
    while current_epoch <= args.epochs:
        try:
            train_disp_head(
                model, optimizer, scheduler,
                train_dl, val_dl,
                device, writer,
                current_epoch, args.epochs,
                args.thresholds,
                args,
                save_every=args.save_every,
                best_loss=best_loss, best_auc=best_auc
            )
            break   # finished successfully

        except Exception:
            print(f"\nCrash at epoch {current_epoch}, reloading checkpoint…")
            traceback.print_exc()

            # point CLI arg to the latest rolling checkpoint
            latest_ckpt = os.path.join(writer.log_dir, "latest.pt")
            if not os.path.exists(latest_ckpt):
                raise RuntimeError(f"Missing rolling checkpoint: {latest_ckpt}")
            args.checkpoint = latest_ckpt

            # re-call load_checkpoint to reload everything and get new start_epoch
            model, optimizer, scheduler, current_epoch = load_checkpoint(
                model,
                optimizer,
                scheduler,
                args.checkpoint
            )

    # once done
    print("Training complete.")
    if val_dl:
        validate_disp_head(model, val_dl, device, args.thresholds)
    save_checkpoint(
        os.path.join(writer.log_dir, "final.pt"),
        model, optimizer, scheduler, args.epochs, args
    )


if __name__ == "__main__":
    main()