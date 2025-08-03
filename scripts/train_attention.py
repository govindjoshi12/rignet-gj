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
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from dataset import RigNetDataset, collate_fn, FILE_PATHS, POS_ATTN_AVG
from models import JointNet, VertexAttentionModule

from training_utils import (
    dict_to_device,
    save_checkpoint,
    load_checkpoint,
    load_yaml_config
)

from confusion_utils import confusion_counts, accuracy_precision_recall_f1_support_from_counts


def validate_attn_head(model, dl, criterion, device, writer=None, epoch=None):
    """
    Run validation for the attention head.

    - Computes average BCE loss over all batches.
    - Computes overall accuracy, precision, recall, and F1 for the positive class.
    - Logs scalars to TensorBoard if writer+epoch provided.

    Returns:
        avg_loss (float),
        accuracy (float),
        precision (array[2]),
        recall (array[2]),
        f1_score (array[2]),
        support (array[2])
    """
    model.eval()
    total_loss = 0.0
    total_TP = total_FP = total_TN = total_FN = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Validating attn head"):
            batch = dict_to_device(batch, device)

            # Forward pass
            logits = model(
                batch['vertices'],
                batch['one_ring'],
                batch['geodesic']
            ).squeeze()
            loss = criterion(logits, batch['attn_mask'])
            total_loss += loss.item()

            # Get numpy preds + gts
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            gts   = batch['attn_mask'].cpu().numpy().astype(int)

            # Predictions and ground truth
            TP, FP, TN, FN = confusion_counts(preds, gts)
            total_TP += TP
            total_FP += FP
            total_TN += TN
            total_FN += FN

            # clear memory
            del logits, loss, probs, preds, gts, batch
            torch.cuda.empty_cache()

    # Compute average loss
    avg_loss = total_loss / len(dl)

    # Compute metrics
    acc, precision, recall, f1, support = accuracy_precision_recall_f1_support_from_counts(
        total_TP, total_FP, total_TN, total_FN
    )

    # Positive class metrics (index=1)
    pos_prec = precision[1]
    pos_rec  = recall[1]

    print(f"[Validate] loss={avg_loss:.4e}, acc={acc:.4f}, "
          f"prec@1={pos_prec:.4f}, rec@1={pos_rec:.4f}")

    # TensorBoard logging
    if writer is not None and epoch is not None:
        writer.add_scalar("val/epoch_loss", avg_loss, epoch)
        writer.add_scalar("val/accuracy",    acc,      epoch)
        writer.add_scalar("val/precision",   pos_prec, epoch)
        writer.add_scalar("val/recall",      pos_rec,  epoch)

    return avg_loss, acc, precision, recall, f1, support


def train_attn_head(
    model,
    optimizer,
    scheduler,
    criterion,
    train_dl,
    val_dl,
    device,
    writer,
    start_epoch,
    max_epochs,
    args,
    save_every=100,
    best_loss=float('inf'),
    best_f1=0.0
):
    """
    Training loop for the attention head, with checkpointing:

    - Every `save_every` epochs: overwrite rolling checkpoint via save_checkpoint.
    - Whenever validation loss improves: save_checkpoint("best_loss.pt").
    - Whenever validation accuracy improves: save_checkpoint("best_acc.pt").
    - Steps the scheduler once per mini-batch.
    """

    scaler = GradScaler(device=device)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}/{max_epochs}")):
            batch = dict_to_device(batch, device)

            # Forward + loss

            # ------- Mixed Precision -------
            optimizer.zero_grad()
            with autocast(device_type=str(device)):
                logits = model(
                    batch['vertices'],
                    batch['one_ring'],
                    batch['geodesic']
                ).squeeze()
                loss = criterion(logits, batch['attn_mask'])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # -------------------------------

            scheduler.step()       # one-cycle LR step per batch
            
            running_loss += loss.item()
            step = (epoch - 1) * len(train_dl) + i
            writer.add_scalar("train/batch_loss", loss.item(), step)

            # clear memory
            del logits, loss
            del batch
            torch.cuda.empty_cache()

        avg_train_loss = running_loss / len(train_dl)
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch)
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4e}")

        # Validation pass
        if val_dl is not None:
            val_loss, _, _, _, val_f1, _ = validate_attn_head(
                model, val_dl, criterion, device, writer, epoch
            )

            del _

            # Best‐loss checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                ckpt_path = os.path.join(writer.log_dir, "best_loss.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch, args
                )
                print(f"-> New best val_loss={best_loss:.4e}, saved best_loss.pt")

            # Best‐f1 (of positive mask class) checkpoint
            if val_f1[1] > best_f1:
                best_f1 = val_f1[1]
                ckpt_path = os.path.join(writer.log_dir, "best_f1.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch, args
                )
                print(f"-> New best val_f1={best_f1:.4f}, saved best_f1.pt")

        # Rolling checkpoint every `save_every` epochs
        if epoch % save_every == 0:
            ckpt_path = os.path.join(writer.log_dir, "latest.pt")
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch, args
            )
            print(f"-> Saved rolling checkpoint at epoch {epoch}")


def parse_args():
    p = argparse.ArgumentParser("Pretrain RigNet Attention Head")
    p.add_argument("--train-path",   type=str, default=FILE_PATHS['train'])
    p.add_argument("--val-path",     type=str, default=FILE_PATHS['val'])
    p.add_argument("--epochs",       type=int, default=5000)
    p.add_argument("--batch-size",   type=int, default=2)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--wd",           type=float, default=1e-6)
    p.add_argument("--edge-dropout", type=int, default=15)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--logdir",       type=str, default="runs/attention_pretrain")
    p.add_argument("--checkpoint",   type=str, default=None,
                   help="Path to full training checkpoint")
    p.add_argument("--save-every",   type=int, default=100,
                   help="Save rolling checkpoint every N epochs")
    p.add_argument("--pos-attn-avg", type=float, default=POS_ATTN_AVG)
    p.add_argument("--config",       type=str,
                   help="Optional YAML config file")
    return p.parse_args()


def main():
    args = parse_args()
    # 1) override via YAML
    if args.config:
        load_yaml_config(args.config, args)

    # 2) device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 3) data loaders
    train_ds = RigNetDataset(args.train_path)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_dl = None
    if args.val_path:
        val_ds = RigNetDataset(args.val_path)
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

    # 4) TensorBoard writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        args.logdir,
        f"lr{args.lr}_wd{args.wd}_{timestamp}"
    )
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    print("Logging to", logdir)

    # 5) build model + optimizer + scheduler
    steps_per_epoch = len(train_dl)
    print("edge dropout: %d" % args.edge_dropout)
    model = VertexAttentionModule(edge_dropout=args.edge_dropout).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch
    )

    # default start epoch
    current_epoch = 1

    # 6) resume from checkpoint if provided
    if args.checkpoint:
        model, optimizer, scheduler, current_epoch = load_checkpoint(
            model,
            optimizer,
            scheduler,
            args.checkpoint
        )

    # 7) build loss
    pos_weight = torch.tensor(
        (1 - args.pos_attn_avg) / args.pos_attn_avg
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 8) retry loop
    best_loss = float('inf')
    best_f1  = 0.0

    while current_epoch <= args.epochs:
        try:
            train_attn_head(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                train_dl=train_dl,
                val_dl=val_dl,
                device=device,
                writer=writer,
                start_epoch=current_epoch,
                max_epochs=args.epochs,
                args=args,
                save_every=args.save_every,
                best_loss=best_loss,
                best_f1=best_f1
            )
            break
        except Exception:
            print(f"\nCrash at epoch {current_epoch}, reloading latest.pt …")
            traceback.print_exc()
            latest_ckpt = os.path.join(writer.log_dir, "latest.pt")
            if not os.path.exists(latest_ckpt):
                raise RuntimeError(f"Rolling checkpoint not found: {latest_ckpt}")
            args.checkpoint = latest_ckpt
            model, optimizer, scheduler, current_epoch = load_checkpoint(
                model,
                optimizer,
                scheduler,
                args.checkpoint
            )

    # 9) final evaluation & save
    if val_dl:
        validate_attn_head(model, val_dl, criterion, device, writer, epoch=None)

    final_path = os.path.join(writer.log_dir, "final.pt")
    save_checkpoint(final_path, model, optimizer, scheduler, args.epochs, args)
    print("Saved final checkpoint to", final_path)


if __name__ == "__main__":
    main()
