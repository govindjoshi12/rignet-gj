import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_max

from mean_shift_clustering import mean_shift_clustering, mode_extraction

class GMEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        in_channels:  F_in, dimension of input vertex features
        out_channels: F_out, dimension of output vertex features
        """
        super().__init__()
        hidden = out_channels 
        # Edge MLPs for topo and geo neighborhoods
        self.mlp_topo = nn.Sequential(
            nn.Linear(2*in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mlp_geo = nn.Sequential(
            nn.Linear(2*in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # Fuse MLP to combine topo+geo pooled features
        self.mlp_fuse = nn.Sequential(
            nn.Linear(2*hidden, out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index_topo, edge_index_geo):

        N, F_in = x.shape

        # --- one-ring ---

        # [E_topo] (num one-ring/topological edges)
        i_t, j_t = edge_index_topo  

        # Index all xi-xj vertex pairs
        xi = x[i_t] # [E_topo, F_in]
        xj = x[j_t] # [E_topo, F_in]
        edge_feat_t = torch.cat([xi, xj - xi], dim=1) # [E_topo, 2*F_in]
        # Apply topological convolutional mlp
        edge_feat_t = self.mlp_topo(edge_feat_t) # [E_topo, hidden]

        # Pool max over neighbors per i_t
        # During mesh graph creation, it was guaranteed that each vertex has at least one neighbor (itself)
        topo_pooled, _ = scatter_max(edge_feat_t, i_t, dim=0, dim_size=N)

        # --- geodesic edges ---
        i_g, j_g = edge_index_geo
        xi_g = x[i_g]
        xj_g = x[j_g]
        edge_feat_g = torch.cat([xi_g, xj_g - xi_g], dim=1)  # [E_geo, 2*F_in]
        edge_feat_g = self.mlp_geo(edge_feat_g) # [E_geo, hidden]

        geo_pooled, _ = scatter_max(edge_feat_g, i_g, dim=0, dim_size=N)
        # geo_pooled: [N, hidden]

        # --- fuse ---
        combined = torch.cat([topo_pooled, geo_pooled], dim=1)  # [N, 2*hidden]
        x_out = self.mlp_fuse(combined) # [N, out_channels]

        return x_out


class GMEdgeNet(nn.Module):
    """A stack of edge convolutions with a prediction head of variable output size.
    
    The convolution layers are hardcoded for now for the purposes of the joint net. 
    """

    def __init__(self, out=3):

        super().__init__()

        self.conv1 = GMEdgeConv(3, 64)
        self.conv2 = GMEdgeConv(64, 256)
        self.conv3 = GMEdgeConv(256, 512)

        self.global_mlp = nn.Sequential(
            nn.Linear(832, 1024),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(1859, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, out), 
            # No final activation. Outputs should be unconstrained
        )

    def forward(self, verts, one_ring, geodesic):

        N, D = verts.shape

        out64 = self.conv1(verts, one_ring, geodesic)
        out256 = self.conv2(out64, one_ring, geodesic)
        out512 = self.conv3(out256, one_ring, geodesic)

        x832 = torch.cat([out64, out256, out512], dim=1)

        # Channelwise max: [832]
        x832_max, x832_max_idxs = torch.max(x832, dim=0)

        x835 = torch.cat([verts, x832], dim=1)
        glob1024 = self.global_mlp(x832_max)

        x1859 = torch.cat([x835, glob1024.expand(N, 1024)], dim=1)
        out = self.head(x1859)
        
        return out

# Simple Wrappers

class JointDisplacementModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = GMEdgeNet(out=3)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class VertexAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GMEdgeNet(out=1)
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# Utility method
def _drop_edges(edges, drop_amt):
    # edges: [2, E]
    # 0 <= drop_amt < E

    i, j = edges # both [E_geo]
    E = i.size(0)

    # pick the E_geo - edge_dropout indices *to keep*
    # Randperm gets a random permutation of elements from 0 to E
    # Then we only keep the first E - drop_amt elements
    keep_idx = torch.randperm(E, device=i.device)[: E - drop_amt]
    i = i[keep_idx]
    j = j[keep_idx]
    # reassemble
    edges = torch.stack([i, j], dim=0)
    return edges


class JointFeatureNet(nn.Module):
    def __init__(self, edge_dropout=15):
        """
        JointFeatureNet extracts rich joint-candidate features from a mesh by predicting per-vertex
        displacements and attention scores over both topological and geodesic neighborhoods.

        Args:
            edge_dropout (int): number of geodesic edges to randomly drop (only during training),
                regularizing the network by simulating variations in mesh tessellation.

        Forward inputs:
            x (Tensor[N, 3]): vertex positions.
            edges_topo (LongTensor[2, E_topo]): one-ring edge index pairs.
            edges_geo (LongTensor[2, E_geo]): geodesic edge index pairs.

        Returns:
            q (Tensor[N, 3]): displaced vertex positions (toward joint centers).
            attn (Tensor[N]): per-vertex attention scores in [0,1].

        Edge dropout:
            During training, if `0 < edge_dropout*2 < E_geo`, `_drop_edges` is called to randomly
            remove `edge_dropout` entries from the geodesic edge list before feature extraction.
            This acts as a form of mesh-edge dropout, making the model more robust to varying
            tessellation and preventing over-reliance on any fixed subset of geodesic connections.
        
        This docstring was generated by ChatGPT o4-mini.
        """

        super().__init__()
        self.disp_head = JointDisplacementModule()
        self.attn_head = VertexAttentionModule()
        
        self.edge_dropout = edge_dropout

    def forward(self, x, edges_topo, edges_geo):
        # x: [N,3]
        N, D = x.shape

        # geodesic edge dropout
        num_geo_edges = edges_geo[0].size(0)
        # Only drop is edge_dropout is positive
        # and less than half of edges would be dropped
        if self.training \
            and self.edge_dropout > 0 \
            and self.edge_dropout * 2 < num_geo_edges:
            edges_geo = _drop_edges(edges_geo, self.edge_dropout)

        disp = self.disp_head(x, edges_topo, edges_geo) # [N,3]
        q = x + disp # displaced points [N,3]

        # per-vertex attention [0, 1] 
        attn = F.sigmoid(self.attn_head(x, edges_topo, edges_geo))

        return q, attn

# ---- Mean Shift Clustering Head ----

class MeanShiftClusterer(nn.Module):

    def __init__(self, 
                 initial_h=0.05, 
                 train_iters=10,
                 infer_iters=50):
        
        super().__init__()
        self.h = initial_h

        self.train_iters = train_iters
        self.infer_iters = infer_iters

    def forward(self, q: torch.Tensor, attn: torch.Tensor = None):
        # q: [N, 3], attn: [N, 1]

        if attn is None:
            N, _ = q.size(dim=0)
            attn = torch.ones(size=(N, 1))

        max_iters = self.train_iters if self.training else self.infer_iters

        T_pred = mean_shift_clustering(q, attn, self.h, max_iters=max_iters)
        T_pred = mode_extraction(T_pred, attn, self.h)
        return T_pred

# --- Overall Joint Prediction Network ---

class JointNet(nn.Module):
    def __init__(self, 
                 edge_dropout=15,
                 initial_h=0.05,
                 train_iters=10,
                 infer_iters=50):
        super().__init__()
        self.feature_extractor = JointFeatureNet(edge_dropout=edge_dropout)
        self.attn_head = self.feature_extractor.attn_head
        self.disp_head = self.feature_extractor.disp_head
        self.clustering_head = MeanShiftClusterer(
            initial_h=initial_h,
            train_iters=train_iters,
            infer_iters=infer_iters
        )
    
    def _extract_graph(self, G: dict):

        verts = G["vertices"]
        E_topo = G["one_ring"]
        E_geo = G["geodesic"]

        return verts, E_topo, E_geo
    
    def attn_head_parameters(self):
        """Return parameters of the attention head."""

        return self.attn_head.parameters()

    def attn_head_forward(self, G: dict):
        """
        Run attn_head forward pass using G as input.

        Returns:
            [N, 1] tensor of attention *logits* for each vertex. 
        """

        return self.attn_head(*self._extract_graph(G))

    def disp_head_forward(self, G: dict):
        """
        Run disp_head forward pass using G as input.

        Returns:
            [N, 3] tensor of displacements for each vertex. 
        """

        return self.disp_head(*self._extract_graph(G))

    def forward(self, G: dict):
        """
        Run joint prediction on either a single graph or a batch of graphs.

        Args:
            G (dict): Graph data, must contain:
                - "vertices": Tensor of shape [N_total, 3]
                - "one_ring": LongTensor of shape [2, E_topo_total]
                - "geodesic": LongTensor of shape [2, E_geo_total]
            Optionally:
                - "vertices_per_graph": 1D Tensor of ints of length N_total
                  If present, enables batched mode; otherwise, treats input as a single graph.

        Returns:
            List[Tensor]: A list of length M (number of graphs).
                Each entry is a Tensor of shape [K_i, 3], the predicted joint locations
                for graph i. In unbatched mode, returns a single-element list.
        """
        verts, E_topo, E_geo = self._extract_graph(G)

        q, attn = self.feature_extractor(verts, E_topo, E_geo)

        # Determine batching: if vertices_per_graph is provided, split accordingly.
        vpg = G.get("vertices_per_graph", None)
        if vpg is None:
            # Unbatched: single graph
            return [self.clustering_head(q, attn)]

        # Batched: split both q and attn by the given vertex counts
        # Splitting and indexing on a contigutous range is faster than using boolean masks
        splits = torch.split(torch.arange(verts.size(0)), vpg.tolist())
        outs = []
        for idxs in splits:
            q_b = q[idxs]
            attn_b = attn[idxs]
            outs.append(self.clustering_head(q_b, attn_b))
        return outs
