# RigNet Joint Prediction Reimplementation

This repository contains my ongoing work to re-implement the **Joint Prediction** stage of **RigNet: Neural Rigging for Articulated Characters** (Xu _et al._, 2020). RigNet is an end‑to‑end method that predicts skeletons and skinning weights directly from 3D meshes; here I focus on reproducing its first module—joint prediction via GMEdgeNet and mean‑shift clustering.

## Reports

Write‑ups of each phase are available as PDF reports in the `reports/` folder:

1. [0‑proposal.pdf](reports/0-proposal.pdf)  
   Project proposal and high‑level plan.  
2. [1‑data.pdf](reports/1-data.pdf)  
   Data loading, preprocessing, and visualization.  
3. [2‑mean‑shift‑clustering.pdf](reports/2-mean-shift-clustering.pdf)  
   Derivation and implementation of the mean‑shift clustering module.

## References

- Z. Xu, Y. Zhou, E. Kalogerakis, C. Landreth, K. Singh, “RigNet: Neural Rigging for Articulated Characters,” _ACM Transactions on Graphics_, vol. 39, no. 4, 2020.  
  https://doi.org/10.1145/3386569.3392379