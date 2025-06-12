import trimesh

def subdivide_to_min_verts(mesh: trimesh.Trimesh, min_verts: int = 1000) -> trimesh.Trimesh:
    """
    Repeatedly apply Loop subdivision until mesh has at least min_verts vertices.
    Stops early if subdivision no longer increases the vertex count.
    """
    current = len(mesh.vertices)
    # early exit
    if current >= min_verts:
        return mesh

    while current < min_verts:
        # apply one iteration of Loop subdivision
        mesh_sub = mesh.subdivide()  # same as mesh.subdivide_loop()
        new_count = len(mesh_sub.vertices)

        # if no growth, bail out
        if new_count <= current:
            break

        mesh = mesh_sub
        current = new_count

    return mesh

def decimate_to_range(mesh: trimesh.Trimesh, min_tris=1000, max_tris=8000, shrink_factor=0.8):
    """
    Quadric-decimates mesh so that its triangle count ends up in [min_tris, max_tris],
    or stops early if it can't get any smaller.
    """
    current = len(mesh.triangles)
    
    # If we’re already within the target band, do nothing
    if min_tris <= current <= max_tris:
        return mesh
    
    # Only decimate when above max_tris
    while current > max_tris:
        # pick a new target strictly between min_tris and current
        target = int(current * shrink_factor)
        # clamp to the lower bound so we don't go below min_tris
        target = max(target, min_tris)
        
        # if target is not strictly less, we can’t make progress
        if target >= current:
            break
        
        mesh_dec = mesh.simplify_quadric_decimation(face_count=target)
        new_count = len(mesh_dec.triangles)
        
        # if no triangles were lost, bail
        if new_count >= current:
            break
        
        mesh = mesh_dec
        current = new_count
    
    mesh.remove_unreferenced_vertices()
    return mesh

# --- LOAD AND PREPROCESS ---

def load_and_preprocess_mesh(obj_path,
                             min_verts=1000,
                             min_tris=1000,
                             max_tris=8000):
    """
    1) Load + repair raw mesh
    2) Center at origin
    3) Subdivide up to >= min_verts
    4) Decimate down into [min_tris, max_tris]
    5) Final clean + normals
    Returns (mesh, centroid), so you can apply the same centering to your rig/joints.
    """

    # --- 1) Load & initial repair ---
    mesh = trimesh.load_mesh(obj_path, process=False)

    # drop any zero‐area or duplicate bits
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes() # closes small cracks that might break subdivision
    
    # --- 2) Center at origin ---
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    
    # --- 3) Grow small meshes up to min_verts ---
    mesh = subdivide_to_min_verts(mesh, min_verts=min_verts)
    
    # --- 4) Shrink big meshes into [min_tris, max_tris] ---
    mesh = decimate_to_range(mesh,
                             min_tris=min_tris,
                             max_tris=max_tris)
    
    # --- 5) Final cleanup 
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()

    return mesh, centroid