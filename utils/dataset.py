import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

# Root Paths

FILE_PATHS = {
    "train": "../data/ModelResource_RigNetv1_preproccessed/mesh_graphs/train.pkl",
    "val": "../data/ModelResource_RigNetv1_preproccessed/mesh_graphs/val.pkl",
    "test": "../data/ModelResource_RigNetv1_preproccessed/mesh_graphs/test.pkl",
}

# Calculated and printed in `create_attn_mask.py`
POS_ATTN_AVG = 0.2964

# Dataset
class RigNetDataset(Dataset):
    def __init__(self, file_path, num_samples=None, seed=42):
        with open(file_path, 'rb') as f:
            self.examples = pickle.load(f)
        
        if num_samples:
            rng = np.random.default_rng(seed=seed)
            self.examples = rng.choice(self.examples, size=num_samples, replace=False)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        # Pull out the raw dict (with NumPy arrays)
        raw = self.examples[index]

        # Build a new map of torch tensors, all on CPU
        G = {
            'vertices': torch.from_numpy(raw['vertices']).float(),      # [V,3]
            'one_ring': torch.from_numpy(raw['one_ring']).long(),       # [2, E_topo]
            'geodesic': torch.from_numpy(raw['geodesic']).long(),       # [2, E_geo]
            'attn_mask': torch.from_numpy(raw['attn_mask']).float(),    # [V]
            'joints': torch.from_numpy(raw['joints']).float(),          # [J,3]
        }
        return G


# collate_fn for batching
def collate_fn(batch: list[dict]):

    verts_list = [b['vertices'] for b in batch]
    topo_list = [b['one_ring'] for b in batch]
    geodesic_list = [b['geodesic'] for b in batch]
    attn_mask_list = [b['attn_mask'] for b in batch]

    # No need to concatenate joints
    # they are only used after the batched graph is processed and unbatched
    joints_list = [b['joints'] for b in batch]

    verts_per_graph = torch.tensor([verts.size(0) for verts in verts_list]).long()
    
    # Tensor of all vertices in batch
    V = torch.concat(verts_list)

    # Vertex-Graph Mapping (maps each vertex to its graph)
    graph_idxs = torch.arange(len(verts_per_graph))
    vertex_graph_indices = graph_idxs.repeat_interleave(verts_per_graph).long()

    # Edge Index Offsets
    # Each edge is represented by a pair of vertex indices
    # Edge indices of each graph must be offset by the number of vertices that came before
    # The offset of the first graph is zero,
    # second graph is len(G1_vertices), third is len(G1_verts) + len(G2_verts), and so on...
    offsets = torch.cat([
        torch.tensor([0]), 
        torch.cumsum(verts_per_graph, dim=0)[:-1]
    ]).long()

    topo_offset_list = []
    geo_offset_list = []
    for topo_b, geo_b, offset in zip(topo_list, geodesic_list, offsets):
        topo_offset_list.append(topo_b + offset)
        geo_offset_list.append(geo_b + offset)

    E_topo = torch.concat(topo_offset_list, dim=1)
    E_geo = torch.concat(geo_offset_list, dim=1)

    # Attention masks don't need to unbatched during inference
    attn_mask = torch.cat(attn_mask_list)

    return {
        "vertices": V,
        "one_ring": E_topo,
        "geodesic": E_geo,
        "attn_mask": attn_mask,
        "graph_idxs": vertex_graph_indices,
        "vertices_per_graph": verts_per_graph,
        "joints_list": joints_list 
    }
