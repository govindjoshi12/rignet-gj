import sys
sys.path.insert(1, '../utils')

import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import heapq
import open3d as o3d
import argparse
from mesh_utils import load_and_preprocess_mesh

# --- MESH GRAPH (TOPOLOGICAL NEIGHBORHOODS) ---

def convert_mesh_to_graph(obj_path: str):
    """
    Reads an obj file and converts it to:
      - vertices: list of (x,y,z) coords
      - adjacency: dict mapping vertex index to list of (neighbor_index, distance)
    
    We first collect all 'v ' lines into `vertices`, and all 'f ' lines into `faces` as
    lists of integer vertex indices. Then we build the adjacency.

    return dictionary G = {
        "obj_path": obj_path,
        "vertices": vertices,
        "one_ring_distances": adjaceny dict of tuples (node, distance),
        "one_ring": edge list
    }
    """

    mesh, centroid = load_and_preprocess_mesh(obj_path,
                                              min_verts=1000,
                                              min_tris=1000,
                                              max_tris=8000)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    adjacency = defaultdict(list)
    
    # build adjacency from faces
    total_edge_lengths = 0.0
    num_edges = 0
    for idxs in faces:
        # assume triangular faces
        n = len(idxs)
        for j in range(n):
            u = idxs[j]
            v = idxs[(j + 1) % n]
            dist = np.linalg.norm(np.subtract(vertices[u], vertices[v]))
            total_edge_lengths += dist
            num_edges += 1
            adjacency[u].append((v, float(dist)))
            adjacency[v].append((u, float(dist)))
    
    # compute mean edge length
    mean_edge_length = (total_edge_lengths / num_edges) if num_edges else 0.0
    
    # deduplicate each adjacency list
    # Edge List (actually used by the network)

    edge_list = []
    for node, nbrs in adjacency.items():
        unique_nbrs = list(set(nbrs))
        adjacency[node] = unique_nbrs
        edge_list += [[node, nbr[0]] for nbr in unique_nbrs]

    # Add a self‑edge [i, i] for every vertex
    # to deal with max pooling edge cases
    for i in range(len(vertices)):
        edge_list.append([i, i])

    # G = (V, E)
    G = {
        "vertices": vertices,
        "num_faces": len(faces),
        "one_ring_distances": dict(adjacency),
        "one_ring": edge_list,
        "centroid": centroid
    }

    return G

# --- GEODESIC NEIGHBORHOODS ---

def get_geodesic_adjacency_graph_from_mesh_graph(
    adjacency_distances: dict[int, list[tuple[int, float]]],
    geodesic_distance: float = 0.06
):
    """
    Given:
      adjacency: dict[node] -> list of (neighbor, edge_length)
      geodesic_distance: maximum path-length to consider
    
    Returns:
      edge_list
    """
    geodesic_edge_dict: dict[int, list[int]] = {}

    # For each start node, run a Dijkstra‐like expansion until distance > threshold
    for start in adjacency_distances:
        # min‐heap of (cum_distance, node)
        heap: list[tuple[float, int]] = [(0.0, start)]
        visited_dist: dict[int, float] = {start: 0.0}
        
        while heap:
            cum_dist, node = heapq.heappop(heap)
            # if this popped entry is stale (we found a shorter path before), skip
            if cum_dist > visited_dist[node]:
                continue
            # explore neighbors
            for nbr, edge_len in adjacency_distances[node]:
                new_dist = cum_dist + edge_len
                # if within threshold and either unvisited or found shorter path
                if new_dist <= geodesic_distance and (nbr not in visited_dist or new_dist < visited_dist[nbr]):
                    visited_dist[nbr] = new_dist
                    heapq.heappush(heap, (new_dist, nbr))
        
        # All visited_dist.keys() (excluding the start itself, if desired) are within the radius
        # We’ll include the start too, since geodesic radius 0 includes itself
        geodesic_edge_dict[start] = list(visited_dist.keys())

    edge_list = []
    for node, nbrs in geodesic_edge_dict.items():
        edge_list += [[node, nbr] for nbr in nbrs]

    return edge_list

# --- JOINTS ---

def get_joint_locations(rig_path, centroid):
    # extraxt join locations
    # disregard bone info. Only want joint locations
    joints = []
    with open(rig_path, "r") as f:
        tokens = f.readline().split()
        while(tokens[0] == "joints"):
            joints.append(list(map(float, tokens[2:])))
            tokens = f.readline().split()

    joints = np.array(joints) - centroid
    
    return joints

# --- ATTENTION MASKS

def get_attn_mask(attn_mask_path):
    with open(attn_mask_path, "r") as f:
        mask = list(map(int, f.read().splitlines()))
    return np.array(mask)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(
        description="Precompute mesh-graph pickles (topological + geodesic + joints + attn_mask)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/ModelResource_RigNetv1_preproccessed",
        help="Root folder containing obj/, rig_info/, attn_masks/, and *_final.txt splits."
    )
    args = parser.parse_args()

    data_root = args.data_root
    obj_folder = os.path.join(data_root, "obj")
    rig_folder = os.path.join(data_root, "rig_info")
    attn_mask_folder = os.path.join(data_root, "attn_masks")
    mesh_graphs_folder = os.path.join(data_root, "mesh_graphs")

    os.makedirs(mesh_graphs_folder, exist_ok=True)

    MAX_GEODESIC_DISTANCE = 0.06
    splits = ["train", "test", "val"]

    for split in splits:
        split_file = os.path.join(data_root, f"{split}_final.txt")
        with open(split_file, "r") as f:
            mesh_indices = list(map(int, f.read().splitlines()))

        graph_list = []
        for mesh_idx in tqdm(mesh_indices, 
                             desc=f"Building graphs for {split}"):
            
            obj_path = os.path.join(obj_folder, f"{mesh_idx}.obj")
            rig_path = os.path.join(rig_folder, f"{mesh_idx}.txt")
            attn_path = os.path.join(attn_mask_folder, f"{mesh_idx}.txt")

            # topological graph
            G = convert_mesh_to_graph(obj_path)

            # geodesic neighborhoods
            G['geodesic'] = get_geodesic_adjacency_graph_from_mesh_graph(
                G['one_ring_distances'],
                MAX_GEODESIC_DISTANCE
            )
            del G['one_ring_distances']

            # joints and attn mask
            G['joints'] = get_joint_locations(rig_path, G['centroid'])
            G['attn_mask'] = get_attn_mask(attn_path)

            # reshape edge‐lists to 2 x E arrays
            G['one_ring']  = np.asarray(G['one_ring']).T
            G['geodesic']  = np.asarray(G['geodesic']).T

            G['mesh_index'] = mesh_idx
            graph_list.append(G)

        out_path = os.path.join(mesh_graphs_folder, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(graph_list, f)
        print(f"Saved {len(graph_list)} graphs to {out_path}")


if __name__ == "__main__":
    main()
