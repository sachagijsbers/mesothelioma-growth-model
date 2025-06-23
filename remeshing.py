import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, Delaunay
from pymeshfix import MeshFix

def write_geo_from_points(
    combined_points, filename="mesh_with_bulges.geo",
    connection_radius=0.1, min_distance=1.0
):
    """
    Generate a GMSH .geo file from sparsified points and reduced lines.

    Args:
        combined_points (np.ndarray): Nx3 array of 3D points.
        filename (str): Path to output .geo file.
        connection_radius (float): Radius used for line connections.
        min_distance (float): Minimum allowed spacing between connected points.
    """
    print(f"[INFO] Using cKDTree with radius {connection_radius}")
    
    # Optional downsampling (remove very close points)
    tree = cKDTree(combined_points)
    kept = []
    visited = np.zeros(len(combined_points), dtype=bool)

    for i, pt in enumerate(combined_points):
        if visited[i]:
            continue
        neighbors = tree.query_ball_point(pt, r=min_distance)
        for j in neighbors:
            visited[j] = True
        kept.append(i)

    combined_points = combined_points[kept]
    print(f"[INFO] Downsampled to {len(combined_points)} points with spacing ≥ {min_distance}")

    # Connect points using radius
    tree = cKDTree(combined_points)
    lines_set = set()
    for i, pt in enumerate(combined_points):
        neighbors = tree.query_ball_point(pt, r=connection_radius)
        for j in neighbors:
            if i != j:
                a, b = sorted([i + 1, j + 1])  # 1-based
                lines_set.add((a, b))
    lines = sorted(lines_set)

    # 2D Delaunay
    tri = Delaunay(combined_points[:, :2])
    triangles = tri.simplices

    # Build line map
    line_map = {}
    line_id_counter = 1
    line_defs = []

    def add_line_to_map(p1, p2):
        nonlocal line_id_counter
        key = (min(p1, p2), max(p1, p2))
        if key not in line_map:
            line_map[key] = line_id_counter
            line_defs.append((line_id_counter, key[0], key[1]))
            line_id_counter += 1

    for a, b in lines:
        add_line_to_map(a, b)

    def get_existing_line(p1, p2):
        key = (min(p1, p2), max(p1, p2))
        direction = 1 if p1 < p2 else -1
        if key not in line_map:
            return None
        line_id = line_map[key]
        return line_id if direction == 1 else -line_id

    line_loops = []
    with open(filename, 'w') as f:
        f.write("// GMSH .geo file\n\n")

        # Points
        f.write("// Points\n")
        for i, pt in enumerate(combined_points):
            f.write(f"Point({i+1}) = {{{pt[0]}, {pt[1]}, {pt[2]}, 1.0}};\n")

        # Lines
        f.write("\n// Lines\n")
        for line_id, a, b in line_defs:
            f.write(f"Line({line_id}) = {{{a}, {b}}};\n")

        # Line Loops
        f.write("\n// Line Loops\n")
        loop_id_counter = 1
        for tri_pts in triangles:
            pts = combined_points[tri_pts]
            x1, y1 = pts[0][:2]
            x2, y2 = pts[1][:2]
            x3, y3 = pts[2][:2]
            orientation = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            if orientation < 0:
                tri_pts[[1, 2]] = tri_pts[[2, 1]]

            p1, p2, p3 = tri_pts + 1
            l1 = get_existing_line(p1, p2)
            l2 = get_existing_line(p2, p3)
            l3 = get_existing_line(p3, p1)

            if None in (l1, l2, l3):
                continue

            f.write(f"Line Loop({loop_id_counter}) = {{{l1}, {l2}, {l3}}};\n")
            line_loops.append(loop_id_counter)
            loop_id_counter += 1

        # Surfaces
        f.write("\n// Surfaces\n")
        for sid in line_loops:
            f.write(f"Surface({sid}) = {{{sid}}};\n")

        # Volume
        f.write("\n// Surface Loop\n")
        surface_ids_str = ", ".join(str(s) for s in line_loops)
        if surface_ids_str:
            f.write(f"Surface Loop(1) = {{{surface_ids_str}}};\n")
            f.write(f"Volume(1) = {{1}};\n")

    print(f"[✓] Wrote: {len(combined_points)} points, {line_id_counter - 1} lines, {len(line_loops)} surfaces to '{filename}'.")


def generate_alpha_mesh(
    combined_points,
    alpha=0.5,
    downsample_voxel_size=0.5,
    output_filename="tumor_alpha_mesh.stl",
    meshfix_repair=True
):
    """
    Generates a surface mesh from a point cloud using alpha shape reconstruction.
    
    This function takes a set of 3D points (e.g., tumor boundary points), applies voxel downsampling,
    and uses alpha shape reconstruction to generate a triangular surface mesh. It then applies 
    mesh cleaning and optional repair using MeshFix to ensure the resulting mesh is manifold 
    and watertight. The final mesh is saved as an STL file.

    Parameters:
    ----------
    combined_points : (N, 3) ndarray
        The input 3D point cloud representing the shape to mesh (e.g., tumor surface).
    alpha : float, default=0.5
        The alpha value for alpha shape reconstruction. Smaller values result in tighter fitting shapes.
    downsample_voxel_size : float, default=0.5
        Voxel size used for downsampling the point cloud before meshing.
    output_filename : str, default="tumor_alpha_mesh.stl"
        File path where the output STL mesh will be saved.
    meshfix_repair : bool, default=True
        If True, applies MeshFix repair if the alpha shape mesh is not watertight.

    Returns:
    -------
    mesh : open3d.geometry.TriangleMesh
        The generated and (optionally) repaired triangular surface mesh.
    """

    # Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)

    print(f"Downsampling point cloud to {downsample_voxel_size} voxel size...")
    pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    print(f"Downsampled to {np.asarray(pcd.points).shape[0]} points")

    pcd.estimate_normals()

    # Alpha shape reconstruction
    print(f"Running alpha shape reconstruction with alpha={alpha}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # Clean and simplify
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)
    mesh = mesh.simplify_vertex_clustering(voxel_size=0.3)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    mesh.compute_vertex_normals()
    
    # Check if watertight
    is_watertight = mesh.is_edge_manifold() and mesh.is_vertex_manifold() and mesh.is_watertight()
    print(f"Is the mesh watertight? {'Yes' if is_watertight else 'No'}")
    
    print("Alpha mesh manifold edge:", mesh.is_edge_manifold())
    print("Alpha mesh manifold vertex:", mesh.is_vertex_manifold())
    print("Alpha mesh watertight:", mesh.is_watertight())
    print("Mesh bounding box max:", np.max(np.asarray(mesh.vertices), axis=0))
    print("Before MeshFix:", mesh.get_max_bound() - mesh.get_min_bound())

    # MeshFix repair if still not watertight
    if not is_watertight and meshfix_repair:
        print("[INFO] Repairing mesh using MeshFix...")
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        mf = MeshFix(vertices, triangles)
        mf.repair(verbose=False, joincomp=False, remove_smallest_components=False)
        vertices = mf.v
        triangles = mf.f

        if len(triangles) > 0:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()
            mesh.remove_duplicated_triangles()
            mesh.remove_degenerate_triangles()
            mesh.remove_non_manifold_edges()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            is_watertight = mesh.is_edge_manifold() and mesh.is_vertex_manifold() and mesh.is_watertight()
            print(f"[INFO] MeshFix result watertight? {'Yes' if is_watertight else 'No'}")

    # Save
    o3d.io.write_triangle_mesh(output_filename, mesh)
    print(f"Saved STL mesh to '{output_filename}'")

    return mesh


def poisson_surface_reconstruction(voxelized_points: np.ndarray, output_path="mesh_with_bulges.stl", depth=8, voxel_size=None):
    """
    Performs Poisson surface reconstruction on a set of 3D points.

    Args:
        voxelized_points (np.ndarray): Nx3 array of 3D points.
        output_path (str): Path to save the resulting mesh.
        depth (int): Octree depth for the Poisson reconstruction. Higher values give more detail.
        voxel_size (float or None): Optional voxel downsampling before reconstruction.
    """
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(voxelized_points)

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[INFO] Downsampled to {len(pcd.points)} points.")

    # Estimate normals
    print("[INFO] Estimating normals...")
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 30 * avg_dist
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Poisson Surface Reconstruction
    print("[INFO] Performing Poisson reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    density_threshold = np.percentile(densities, 5)
    mesh.remove_vertices_by_mask(densities < density_threshold)

    # Clean up the mesh
    print("[INFO] Cleaning mesh...")
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_duplicated_vertices()
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    # Validation
    print(f"[INFO] Watertight: {mesh.is_watertight()}")
    print(f"[INFO] Edge manifold: {mesh.is_edge_manifold()}")
    print(f"[INFO] Vertex manifold: {mesh.is_vertex_manifold()}")
    print(f"[INFO] Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")

    # Save the mesh
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"[INFO] Saved mesh to {output_path}")

