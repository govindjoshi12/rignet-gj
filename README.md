# RigNet Joint Prediction Reimplementation (WIP)

This repository contains my ongoing work to re-implement the **Joint Prediction** stage of **RigNet: Neural Rigging for Articulated Characters** (Xu _et al._, 2020). RigNet is an end‑to‑end method that predicts skeletons and skinning weights directly from 3D meshes; here I focus on reproducing its first module—joint prediction via GMEdgeNet and mean‑shift clustering. The goal is to provide a hands-on introduction to:

- Graph neural networks on 3D meshes
- Attention-based masking for joint localization
- Training large neural models (multi-GPU, mixed precision, checkpointing)

---

## 📦 Data

The original dataset is provided by the RigNet authors. To obtain it:

- Email the corresponding author of the paper, or
- If you are at Texas A&M, contact `govind@tamu.edu` directly for a copy.

Once you have **ModelResource\_RigNetv1\_preprocessed.zip** in your working directory:

```bash
unzip ModelResource_RigNetv1_preprocessed.zip
```

This will create a folder structure with:

- `obj/` — raw `.obj` meshes
- `rig_info/` — joint and bone definitions
- `*_final.txt` — train/val/test splits

**Next steps**:

1. Generate per-vertex attention masks:
   ```bash
   python create_attn_masks.py --data-root ./ModelResource_RigNetv1_preprocessed
   ```
2. Precompute mesh graphs (topological + geodesic + joints + masks):
   ```bash
   python preprocess.py --data-root ./ModelResource_RigNetv1_preprocessed
   ```

---

## 📝 Notebooks

We include several Jupyter notebooks documenting both algorithm development and exploratory analysis:

- **Early Prototypes**: initial experiments with toy models and imports; these may be outdated and will eventually be removed.
- **Mean‑Shift Clustering Implementation**: development and testing of the Epanechnikov‑kernel mean‑shift clustering head (`mean_shift_clustering.ipynb`).
- **Attention Mask Construction**: derives the per-joint raycasting and filtering pipeline to build binary masks (`create_attn_masks.ipynb`).
- **Batching**: Derives graph input-batching and output-unbatching
- **Batched Overfit**: demonstrates overfitting on a small batch of meshes, hyperparameter tuning, and training curve inspection (`batched_overfit.ipynb`).
- **Visualize Model Outputs**: load a trained checkpoint and render predicted joints and attention heatmaps locally using Open3D (`visualize_model_outputs.ipynb`).

> Visualizations in these notebooks use **Trimesh** for mesh handling and **Open3D** for interactive rendering.

---

## 🛠️ Utilities

This folder contains generic modules and helpers used throughout the project:

- **mean\_shift\_clustering.py**: Epanechnikov-kernel mean-shift clustering functions (`mean_shift_update_step`, `mode_extraction`, etc.).
- **models.py**: implementations of `GMEdgeConv`, `GMEdgeNet`, and high-level wrappers:
  - `VertexAttentionModule`
  - `JointDisplacementModule`
  - `MeanShiftClusterer`
  - `JointNet` (batched joint-prediction)
- **visualization\_utils.py**: helper functions to visualize meshes, attention heatmaps, and predicted joints via Open3D.
- **training\_utils.py**: training helpers, loss functions, checkpointing utilities, and PCK computations.
- **cls\_validation\_utils.py**: standalone confusion‐matrix and precision/recall/F1/support helpers for binary classification.

---

## 📑 Reports

The `reports/` folder contains design notes, experiment logs, and PDF write‑ups summarizing key steps and findings during the implementation of this project.

1. [0‑proposal.pdf](reports/0-proposal.pdf)  
   Project proposal and high‑level plan.  
2. [1‑data.pdf](reports/1-data.pdf)  
   Data loading, preprocessing, and visualization.  
3. [2‑mean‑shift‑clustering.pdf](reports/2-mean-shift-clustering.pdf)  
   Derivation and implementation of the mean‑shift clustering module.
4. [4‑simple-baseline.pdf](reports/2-mean-shift-clustering.pdf)  
   Results from training a simple baseline model.
5. [5‑attn-mask.pdf](reports/2-mean-shift-clustering.pdf)  
   Deriving attention masks for pretraining the attention module. 

---

## 🚆 Model Training

We provide two standalone training scripts for the two core modules:

1. **Attention Head** (`train_attention.py`): pre-trains only the `VertexAttentionModule`.
2. **Displacement Head** (`train_displacement.py`): trains only the `JointDisplacementModule`.

> The original paper describes joint joint training (pretrain attention, then fine‑tune everything with combined Chamfer losses), including the bandwidth parameter in the mean-shift head. Our current implementation of clustering-head training is a work in progress, so for now we train attention and displacement separately.

Both scripts can be invoked via command line or by passing a YAML config (stored under `configs/`). Each script supports:

- `--config path/to/config.yaml` to load hyperparameters
- `--checkpoint path/to/latest.pt` to resume from a saved checkpoint (model+optimizer+scheduler+amp scale)
- Automatically retry the training loop on random device/assertion errors, resuming from the latest rolling checkpoint.

**Examples**:

```bash
python train_attention.py --config ../configs/attention.yaml

python train_displacement.py \
  --config ../configs/displacement.yaml \
  --checkpoint ../runs/disp_pretrain/lr5e-05_wd1e-06_20250717-073438/latest.pt
```

### 📊 TensorBoard

All training scripts log metrics (losses, PCK/AUC, precision/recall) to TensorBoard under the given `--logdir`. To launch:

```bash
# Forward port 6006 on remote to your local 6006
ssh -L 6006:localhost:6006 user@remote.host.edu
# On the remote server shell:
cd path/to/project
tensorboard --logdir runs/ --bind_all --port 6006
# Then open http://localhost:6006 in your browser
```

> If using VS Code Remote, ports are forwarded automatically.

---

## 🔮 Next Steps

1. **Scheduler Script**: build a wrapper that launches multiple `train_attention.py` and `train_displacement.py` jobs over different hyperparameters, then selects the best model based on validation metrics and evaluates it on the test set.

2. **Clustering-Head Training**: implement training of the mean‑shift bandwidth parameter by:
   - Freezing attention & displacement heads, then updating only `h`, or
   - Following the paper’s schedule: pretrain attention, then jointly fine‑tune attention, displacement, and bandwidth with combined Chamfer losses.

3. **Symmetry Post‑processing**: many meshes are symmetric; after joint prediction, mirror half the joints across the symmetry axis (paper’s approach) to improve consistency.

4. **Downstream Bone‑Prediction Stage**: the paper feeds *ground‑truth* joints into the bone-prediction network rather than predicted joints due to varying joint counts. Investigate how to adapt bone‑prediction to accept variable‑length joint lists or integrate joint‑count normalization.

> With most GMEdgeConv/GMEdgeNet building blocks in place, the next challenge is fully reproducing the joint prediction workflow end‑to‑end and then moving on to bone prediction as in RigNet.

## References

- Z. Xu, Y. Zhou, E. Kalogerakis, C. Landreth, K. Singh, “RigNet: Neural Rigging for Articulated Characters,” _ACM Transactions on Graphics_, vol. 39, no. 4, 2020.  
  https://doi.org/10.1145/3386569.3392379