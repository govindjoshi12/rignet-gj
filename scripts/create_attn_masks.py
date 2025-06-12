"""
Constructing Per-Pixel Binary Attention Masks from Joint and Bone Information

The RigNet paper finds that pre-training the attention modules weights 
with a cross-entropy loss function with a per-vertex attention mask can 
improve performance. In this notebook, we construct these masks.

- Cast K rays from the joint in the direction of the plane orthogonal 
to the bone centered at the joint (K=14)
- Perform triangle-ray intersection and gather vertices closest to 
intersection points.
    - If <6 vertices found, just to kNN centered at the joint with k=6
    - If triangle-ray intersection fails, do a "nearby triangle" search 
        - We need this because there ARE rare instances where we rays won't 
        intersect with triangles. This ensures that there at least some training 
        signal retained for each joint
- Find the 20th percentile distance of vertices. Multiply this distance by 2. This distance is threshold for retaining points.

Notes: 
- RigNet decimates meshes to 3k verts before doing this. We don't do this.
"""

import sys
sys.path.insert(1, '../utils')

import os
import glob
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm
import argparse
from mesh_utils import load_and_preprocess_mesh

def parse_rig(rig_path: str, centroid, mesh_idx):
    # extraxt join locations
    # disregard bone info. Only want joint locations

    # file structure:
    # joint name x y z
    # root name
    # skin name1 weight1 name2 weight2 ...
    # hier name1 name2

    # Will assume this rigid order

    jointname2idx = {}
    joints = []
    bones = []
    root_idx = ""
    with open(rig_path, "r") as f:
        tokens = f.readline().split()
        while(tokens[0] == "joints"):
            name = tokens[1]
            loc = list(map(float, tokens[2:]))

            # name to index mapping
            next_idx = len(joints)
            jointname2idx[name] = next_idx
            joints.append(loc)
            tokens = f.readline().split()
        
        # root
        root_idx = jointname2idx[tokens[1]]

        # Skip skin info
        while tokens[0] != "hier":
            tokens = f.readline().split()
        
        # Hier info
        while tokens:
            try:
                b1 = jointname2idx[tokens[1]]
                b2 = jointname2idx[tokens[2]]
                bones.append([b1, b2])
            except:
                print(f"Warning: bone ({b1}->{b2}) invalid in mesh {mesh_idx}, skipping bone.")

            tokens = f.readline().split()

    joints = np.array(joints) - centroid
    
    return joints, bones, root_idx

# --- Ray-Casting --- 

# Method for finding orthonormal plane to v
def pick_arbitrary(v):
    # pick axis least aligned with v
    abs_v = np.abs(v)

    # If the x-component is the smallest, pick the x-axis
    if abs_v[0] <= abs_v[1] and abs_v[0] <= abs_v[2]:
        return np.array([1.0, 0.0, 0.0])
    # x-component is not the smallest. Check if y is smaller than z. 
    elif abs_v[1] <= abs_v[2]:
        return np.array([0.0, 1.0, 0.0])
    # z-component is the smallest.
    else:
        return np.array([0.0, 0.0, 1.0])


def get_orthonormal_plane(v, eps=1e-8):
    # assume v is already unit length
    a = pick_arbitrary(v)
    # project a on-to v, then remove component along v
    u0 = a - np.dot(a, v) * v
    # normalize
    u0 /= np.linalg.norm(u0) + eps
    # second orthonormal vector
    w = np.cross(v, u0)
    w /= np.linalg.norm(w) + eps
    return u0, w


def form_rays(joints, bones, K=14):
    """
    joints: list of (x,y,z) arrays, shape [J,3]
    bones: list of (parent_idx, child_idx) pairs
    K: number of rays per bone-end
    
    Returns:
      origins: np.ndarray [2*K*len(bones), 3]
      dirs: np.ndarray [2*K*len(bones), 3]
    """
    origins_list = []
    dirs_list = []
    joint_idx_list = []
    
    for p_idx, c_idx in bones:
        p_pos = np.array(joints[p_idx])
        c_pos = np.array(joints[c_idx])
        
        # bone direction
        v = c_pos - p_pos
        v = v / (np.linalg.norm(v) + 1e-10)
        
        # orthonormal plane basis
        u0, w = get_orthonormal_plane(v)
        
        # sample K angles around the circle
        thetas = np.linspace(0, 2*np.pi, K, endpoint=False)

        # [K, 3]
        dirs_k = np.stack([np.cos(t) * u0 + np.sin(t) * w for t in thetas], axis=0)
        dirs_k /= (np.linalg.norm(dirs_k, axis=1, keepdims=True) + 1e-10)
        
        # create 2 origins (parent & child), each repeated K times
        bone_origins = np.vstack([p_pos, c_pos]) # [2,3]
        origin_2K = np.repeat(bone_origins, K, axis=0) # [2*K,3]

        # joint indices 
        joint_idxs = np.vstack([p_idx, c_idx])
        joints_2K = np.repeat(joint_idxs, K, axis=0)
        
        # tile dirs twice to match origins
        dirs_2K = np.tile(dirs_k, (2, 1)) # [2*K,3]
        
        origins_list.append(origin_2K)
        dirs_list.append(dirs_2K)
        joint_idx_list.append(joints_2K)
    
    # concatenate all bones so we can shoot all rays from every joint together
    origins = np.concatenate(origins_list, axis=0)
    dirs = np.concatenate(dirs_list, axis=0)
    joint_idxs = np.concatenate(joint_idx_list, axis=0)
    
    return origins, dirs, joint_idxs


def shoot_rays(mesh, origins, dirs, mesh_idx):
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    
    try:
        locs, ray_ids, tri_ids = intersector.intersects_location(origins, dirs)
    except Exception as e:
        print(f"Mesh {mesh_idx} ray-intersect failed: {e}; skipping this mesh.")
        return None

    hits_by_ray = defaultdict(list)
    tris_by_ray = defaultdict(list)

    # iterate over every intersection
    for pt, r, t in zip(locs, ray_ids, tri_ids):
        hits_by_ray[r].append(pt)
        tris_by_ray[r].append(t)
    
    selected_hits = []
    # list of tuple[ray_idx, point, triangle, distance from origin]

    for r in range(origins.shape[0]):
        hits = hits_by_ray.get(r, [])
        tris = tris_by_ray.get(r, [])

        if hits:
            pts = np.stack(hits, axis=0) # [m, 3]
            dists = np.linalg.norm(pts - origins[r], axis=1) # origins[r] broadcasted
            k = np.argmin(dists) # get closest point
        
            selected_hits.append((r, pts[k], tris[k], dists[k]))
        else:

            # Fallback: get faces near the origin
            # Index origins with newaxis indexing 
            close_tris = trimesh.proximity.nearby_faces(mesh, origins[r][None, :])

            # get vertices from triangle indices
            vs = mesh.faces[close_tris].flatten()

            # Use np.asarray to index mesh.vertices with an array (vs)
            fallback_pts = np.asarray(mesh.vertices)[vs]
            
            # record all fallback_pts 
            for pt in fallback_pts:
                selected_hits.append((r, pt, None, None))
    
    return selected_hits


def collect_hits_by_joint(selected_hits, ray_joint_idxs):

    hits_by_joint = defaultdict(list)

    for r, pt, *_ in selected_hits:
        # which joint generated ray r
        j = ray_joint_idxs[r][0] 
        hits_by_joint[j].append(pt)

    # inspect how many hits each joint got
    # for j, pts in hits_by_joint.items():
    #     print(f"Joint {j} has {len(pts)} hit points")
    
    return hits_by_joint


def filter_hits(hits_by_joint, joints):
    filtered_pts   = []
    filtered_jidxs = []

    for j, pts in hits_by_joint.items():
        if len(pts) == 0:
            continue
        
        # Stack to (M_j, 3)
        pts_arr = np.stack(pts, axis=0)
        
        # Joint position:
        joint_pos = np.array(joints[j])[None, :] # shape (1,3)
        
        # distances of each hit pt to joint j
        dists = np.linalg.norm(pts_arr - joint_pos, axis=1)  # shape (M_j,)
        
        # 20th percentile
        p20 = np.percentile(dists, 20)
        
        # threshold = 2 * p20
        keep_mask = (dists < 2 * p20)
        
        # collect filtered points
        kept_pts = pts_arr[keep_mask]
        filtered_pts.append(kept_pts)
        
        # record joint index for each kept point
        filtered_jidxs.append(np.full(len(kept_pts), j, dtype=int))

    # Flatten lists into arrays
    if filtered_pts:
        hit_pts = np.concatenate(filtered_pts, axis=0) # (P,3)
        hit_joints = np.concatenate(filtered_jidxs, axis=0) # (P,)
    else:
        # In case filtered pts is empty
        hit_pts = np.zeros((0,3), dtype=float)
        hit_joints = np.zeros((0,), dtype=int)

    # print(f"After filtering: {hit_pts.shape[0]} total hit-points across {len(hits_by_joint)} joints")

    return hit_pts, hit_joints


def build_attention_mask(vtx_ori, hit_pts, radius=0.02):
    """
    vtx_ori : (V,3) array of original mesh vertices
    hit_pts : (P,3) array of filtered surface hits
    radius   : float radius threshold in mesh units
    
    Returns:
      attn_mask : (V,) boolean array
    """
    V = vtx_ori.shape[0]
    attn_mask = np.zeros(V, dtype=bool)
    
    # build KD-tree on vertices for faster indexing
    tree = cKDTree(vtx_ori)
    
    # for each hit point, find all vertices within radius
    # this returns a list of lists; we can flatten it
    neighbors = tree.query_ball_point(hit_pts, r=radius)
    neighbors = np.unique(np.concatenate(neighbors)).astype('int')
    attn_mask[neighbors] = True
    
    return attn_mask, vtx_ori[neighbors]


def create_attn_mask(
    mesh_idx: int,
    obj_path: str,
    rig_path: str,
    attn_path: str
):

    mesh, centroid = load_and_preprocess_mesh(obj_path,
                                              min_verts=1000,
                                              min_tris=1000,
                                              max_tris=8000)

    verts = np.asarray(mesh.vertices)
    
    joints, bones, root_idx = parse_rig(rig_path, centroid, mesh_idx)
    origins, dirs, ray_joint_idxs = form_rays(joints, bones, K=14)
    hits = shoot_rays(mesh, origins, dirs, mesh_idx)

    if hits is None:
        print(f"Skipping Mesh {mesh_idx}")
        return

    hits_by_joint = collect_hits_by_joint(hits, ray_joint_idxs)

    hit_pts, hit_joints = filter_hits(hits_by_joint, joints)
    attn_mask, marked_verts = build_attention_mask(verts, hit_pts, radius=0.02)
    # print(f"{attn_mask.sum()} / {len(attn_mask)} vertices marked as attention.")

    assert len(attn_mask) == len(verts)

    np.savetxt(attn_path, attn_mask.astype(int), fmt='%d')

    # Return mask to calculate positive class weight
    return attn_mask

# --- Main Method ---

def main():
    
    # parse command‐line args
    parser = argparse.ArgumentParser(
        description="Build per-vertex binary attention masks"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/ModelResource_RigNetv1_preproccessed",
        help="Path to the preprocessed rig/obj/attn_masks folders."
    )
    args = parser.parse_args()

    # recompute all folder paths from data_root
    data_root = args.data_root
    obj_folder = os.path.join(data_root, "obj")
    rig_folder = os.path.join(data_root, "rig_info")
    attn_folder = os.path.join(data_root, "attn_masks")

    # make sure output folder exists
    os.makedirs(attn_folder, exist_ok=True)

    # find all “*_final.txt” index files, read mesh idxs
    mesh_idx_files = glob.glob(os.path.join(data_root, "*_final.txt"))
    mesh_idxs = []
    for idx_file in mesh_idx_files:
        with open(idx_file, 'r') as f:
            # each line is an integer mesh index
            mesh_idxs.extend(map(int, f.read().splitlines()))

    print(f"Found {len(mesh_idxs)} meshes to process…")

    # loop and build
    pos_frac_total = 0
    total_meshes = 0
    for mesh_idx in tqdm(mesh_idxs):
        try:
            obj_path = os.path.join(obj_folder, f'{mesh_idx}.obj')
            rig_path = os.path.join(rig_folder, f'{mesh_idx}.txt')
            attn_path = os.path.join(attn_folder, f'{mesh_idx}.txt')

            mask = create_attn_mask(mesh_idx, obj_path, rig_path, attn_path)
            pos_frac_total += mask.sum() / len(mask)

            total_meshes += 1

        except Exception as e:
            # catch any unforeseen errors per-mesh so we don't lose the whole run
            print(f"[ERROR] mesh {mesh_idx}: {e}")
    
    pos_frac_avg = pos_frac_total / total_meshes
    print(f"Average Positive Class Percentage: {pos_frac_avg:.4f}")

if __name__ == "__main__":
    main()