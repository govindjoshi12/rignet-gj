import numpy as np
import open3d as o3d
import trimesh
from open3d.visualization import rendering

# Screenshot util for visualization on Remote Server 

def _offscreen_snapshot(geometries, width=800, height=600, 
                        fov=60.0, center=(0,0,0), eye=(1,1,1), up=(0,0,1)):
    """render geometries to an image and return as numpy array.
    
    center: what the camera is looking at,
    eye: position of the camera,
    up: +z
    """
    r = rendering.OffscreenRenderer(width, height)
    scene = r.scene
    mat = rendering.MaterialRecord()
    for geo in geometries:
        scene.add_geometry("", geo, mat)
    r.setup_camera(fov, np.array(center), np.array(eye), np.array(up))
    img = np.asarray(r.render_to_image())
    return img


# --- TRIMESH ---

def visualize_trimesh(mesh_tm: trimesh.Trimesh,
                      mesh_frame_color = [0.7, 0.7, 0.7],
                      joints: np.ndarray = None,
                      joint_color = [1.0, 0.0, 0.0],
                      snapshot=False,
                      fov=60.0,
                      center=(0, 0, 0),
                      eye=(1, 1, 1),
                      up=(0, 0, 1)):
    """
    Visualize a Trimesh in Open3D as a wireframe (LineSet), with optional joints.
    
    Args:
        mesh_tm: trimesh.Trimesh instance
        joints:  optional (J,3) numpy array of joint positions
    """
    # --- Convert to LineSet wireframe ---
    verts = np.asarray(mesh_tm.vertices)
    edges = mesh_tm.edges_unique  # (E,2) array of [u, v] index pairs

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(verts)
    lines.lines = o3d.utility.Vector2iVector(edges)
    
    # color each edge light gray
    colors = np.tile(mesh_frame_color, (len(edges), 1))
    lines.colors = o3d.utility.Vector3dVector(colors)

    # --- Prepare geometries for rendering ---
    geometries = [lines]

    if joints is not None and len(joints) > 0:
        # render each joint as a small red sphere
        for j in joints:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sph.translate(j)
            sph.paint_uniform_color(joint_color)
            sph.compute_vertex_normals()
            geometries.append(sph)

    # --- Draw all ---

    if snapshot:
        return _offscreen_snapshot(
            geometries=geometries,
            fov=fov,
            center=center,
            eye=eye,
            up=up
        )
    else:
        o3d.visualization.draw_geometries(geometries)


# --- MESH GRAPH ---

def _points_to_spheres(points, colors, radius=0.01):
    spheres = []
    for i, (x, y, z) in enumerate(np.asarray(points)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate((x, y, z))
        sphere.paint_uniform_color(colors[i])
        spheres.append(sphere)
    
    return spheres

def visualize_mesh_graph(vertices: np.ndarray,
                        edge_list: np.ndarray,
                        joints_gt: np.ndarray = None,
                        joints_pred: np.ndarray = None,
                        displaced_verts: np.ndarray = None,
                        snapshot=False,
                        fov=60.0,
                        center=(0, 0, 0),
                        eye=(1, 1, 1),
                        up=(0, 0, 1)):

    pts = vertices.astype(dtype=np.float64)
    lines = edge_list.astype(dtype=np.int32)
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines)
    )

    colors = [[0.6, 0.6, 0.6] for _ in lines]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = []
    geometries.append(line_set)

    if joints_gt is not None:
        # green
        geometries.extend(_points_to_spheres(
            joints_gt, 
            colors=np.tile([0, 1, 0], reps=(len(joints_gt), 1))
        ))
        
    if joints_pred is not None:
        # blue
        geometries.extend(_points_to_spheres(
            joints_pred, 
            colors=np.tile([0, 0, 1], reps=(len(joints_pred), 1))
        ))
    
    if displaced_verts is not None:
        # red
        geometries.extend(_points_to_spheres(
            displaced_verts, 
            colors=np.tile([1, 0, 0], reps=(len(displaced_verts), 1)),
            radius=0.003
        ))

    if snapshot:
        return _offscreen_snapshot(
            geometries=geometries,
            fov=fov,
            center=center,
            eye=eye,
            up=up
        )
    else:
        o3d.visualization.draw_geometries(
            geometries,
            mesh_show_back_face=True,
            window_name="Mesh Graph",
            width=800, height=600
        )
    
# ---- ATTENTION HEATMAP ----

def visualize_attention_heatmap(
    verts: np.ndarray,
    edges: np.ndarray,
    attn_pred: np.ndarray,
    attn_gt:   np.ndarray = None,
    joints_gt: np.ndarray = None,
    color_low: np.ndarray = np.array([0.0, 0.0, 1.0]),  # blue
    color_high:np.ndarray = np.array([1.0, 0.0, 0.0]),  # red
    snapshot=False,
    fov=60.0,
    center=(0, 0, 0),
    eye=(1, 1, 1),
    up=(0, 0, 1)
):
    # --- wireframe mesh ---
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts),
        lines= o3d.utility.Vector2iVector(edges),
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.7,0.7,0.7], (len(edges),1))
    )

    # --- interpolate between low and high colors ---
    # attn_pred is assumed in [0,1], shape (N,)
    attn = attn_pred.reshape(-1,1)  # (N,1)
    rgb = (1 - attn) * color_low[None,:] + attn * color_high[None,:]  # (N,3)

    # --- predicted attention as a point cloud ---
    pcd = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(verts)
    )
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    geometries = [ls, pcd]

    if joints_gt is not None:
        # red
        geometries.extend(_points_to_spheres(joints_gt, 
                                          colors=np.tile([1, 0, 0], reps=(len(joints_gt), 1))))

    # --- optionally overlay GT in pure green ---
    if attn_gt is not None:
        gt_idxs = np.nonzero(attn_gt.astype(bool))[0]
        if gt_idxs.size:
            gt_pcd = o3d.geometry.PointCloud(
                points=o3d.utility.Vector3dVector(verts[gt_idxs])
            )
            gt_pcd.colors = o3d.utility.Vector3dVector(
                np.tile([0.0,1.0,0.0], (len(gt_idxs),1))
            )
            geometries.append(gt_pcd)

    if snapshot:
        return _offscreen_snapshot(
            geometries=geometries,
            fov=fov,
            center=center,
            eye=eye,
            up=up
        )
    else:
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Attention Heatmap (pred+gt)"
        )
