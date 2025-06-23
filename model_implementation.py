import dolfinx
import numpy as np
import ufl
from petsc4py import PETSc
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import nibabel as nib
from dolfinx.fem import Function
import meshio
from dolfinx import fem
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from dolfinx.mesh import exterior_facet_indices, compute_incident_entities
import gmsh
from collections import deque
import trimesh
from dolfinx.io import XDMFFile
from mpi4py import MPI
    
def gmsh_new_mesh(input_file, output_file):
    """
    Generate a 3D tetrahedral mesh from an STL file using GMSH.
    
    Parameters:
    - input_file: path to input STL file (without extension or with .stl)
    - output_file: base path for output mesh files (no extension)
    """
    
    # Ensure input has .stl extension
    if not input_file.endswith(".stl"):
        input_file += ".stl"
        
    repaired_stl = output_file + "_repaired.stl"
    msh_file = output_file + ".msh"
    vtk_file = output_file + ".vtk"
    vis_file = output_file + "_visualization.png"
    
    # Step 1: Load and repair the STL file using trimesh
    mesh = trimesh.load_mesh(input_file)

    # Check if the mesh is watertight and repair it if necessary
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to repair...")
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

        # Remove duplicate faces and degenerate faces
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        print(f"Repaired mesh saved as {repaired_stl}.")
    else:
        print("Mesh is already watertight.")

    mesh.update_faces(mesh.unique_faces())

    mesh.export(repaired_stl)

    # Step 2: Initialize GMSH and import the repaired STL file
    gmsh.initialize()
    gmsh.model.add("mesh_model")

    # Import the repaired STL file into GMSH
    gmsh.merge(repaired_stl)

    # Synchronize the geometry
    gmsh.model.geo.synchronize()

    # Step 3: Identify the surfaces imported from the STL
    entities = gmsh.model.getEntities(dim=2)
    print(entities)

    if not entities:
        raise ValueError("No entities found")

    # surface_ids = [entity[1] for entity in entities]
    surface_ids = list(set(entity[1] for entity in entities))  # Remove duplicates

    print(surface_ids)

    # Step 4: Create a surface loop and volume in GMSH
    surface_loop = gmsh.model.geo.addSurfaceLoop(surface_ids)
    print(surface_loop)

    volume = gmsh.model.geo.addVolume([surface_loop])

    # Verify if the volume is created successfully
    if volume == 0:
        raise ValueError("Failed to create volume. Please check the surface loop.")

    # Define physical groups for volume and surfaces
    gmsh.model.addPhysicalGroup(3, [volume], tag=1)  # Physical group for the volume
    gmsh.model.addPhysicalGroup(2, surface_ids, tag=2)  # Physical group for the boundary surfaces

    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Set mesh size
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1.0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)  # Adapt to curvature
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Smooth transition from boundary to interior
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)  # Increase the mesh size to reduce the element count
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Use Netgen
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.4)  # Stop once mesh quality reaches 0.4

    # Generate the tetrahedral mesh
    gmsh.model.mesh.generate(3)

    gmsh.write(msh_file)
    print("Generated and saved 3D mesh.")

    # Export the mesh to VTK for adding density
    gmsh.write(vtk_file)

    # Finalize Gmsh
    gmsh.finalize()
    
    print(f"Saved mesh to {msh_file} and {vtk_file}")

def mesh_load(mesh_name):
    msh = meshio.read(mesh_name + ".msh")

    # Extract tetrahedra and associated cell data
    if "tetra" not in msh.cells_dict:
        raise ValueError("Mesh does not contain tetrahedra.")

    tetra_cells = msh.cells_dict["tetra"]

    # Get physical region markers for tetrahedra
    if "gmsh:physical" in msh.cell_data_dict and "tetra" in msh.cell_data_dict["gmsh:physical"]:
        tet_data = msh.cell_data_dict["gmsh:physical"]["tetra"]
        print(f"Found physical region IDs for tetrahedra: {np.unique(tet_data)}")
    else:
        # Default region ID 1 if missing
        tet_data = np.ones(len(tetra_cells), dtype=np.int32)
        print("No physical region IDs found for tetrahedra. Using dummy ID 1.")
    
    print("Mesh point shape:", msh.points.shape)
    print("Tetra shape:", msh.cells_dict["tetra"].shape)
    print("Tet region tag shape:", tet_data.shape)
    assert msh.cells_dict["tetra"].shape[0] == tet_data.shape[0]
    
    tets = msh.cells_dict["tetra"]
    pts = msh.points

    # Volume of each tet (scalar triple product)
    def tet_volume(a, b, c, d):
        return np.abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0

    volumes = np.array([tet_volume(pts[t[0]], pts[t[1]], pts[t[2]], pts[t[3]]) for t in tets])
    print("Min/Max tet volume:", volumes.min(), volumes.max())
    print("Tetrahedra with zero or negative volume:", np.sum(volumes < 1e-12))
    
    print("NaNs in points:", np.isnan(msh.points).any())
    print("Infs in points:", np.isinf(msh.points).any())

    # Write filtered mesh to XDMF (only tetrahedra + markers)
    meshio.write(
        mesh_name + ".xdmf",
        meshio.Mesh(
            points=msh.points,
            cells=[("tetra", tetra_cells)],
            cell_data={"dom_marker": [tet_data]},
        ),
        file_format="xdmf",
    )
    
    print("written")
    
    # Load mesh in DolfinX
    with XDMFFile(MPI.COMM_WORLD, mesh_name + ".xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        # Precompute all connectivities (safe)
        for d1 in range(mesh.topology.dim + 1):
            for d2 in range(mesh.topology.dim + 1):
                mesh.topology.create_connectivity(d1, d2)

    return mesh

def avg_density_in_tumor_at_vertices(ct_scan_path, seg_mask_path, mesh):
    ct_img = nib.load(ct_scan_path)
    ct_data = np.asanyarray(ct_img.dataobj)  # unscaled raw data
    hdr = ct_img.header
    hu = np.asanyarray(ct_img.dataobj).astype(np.float32)

    # Apply scaling
    slope, intercept = ct_img.header.get_slope_inter()
    if slope is None or np.isnan(slope): slope = 1.0
    if intercept is None or np.isnan(intercept): intercept = 0.0
    hu = hu * slope + intercept
    print("HU range:", np.min(hu), "to", np.max(hu))
    
    # Load CT scan and segmentation mask
    seg_mask = nib.load(seg_mask_path).get_fdata().astype(bool)  # Binary mask (1 = tumor, 0 = background)
    # ct_scan = nib.load(ct_scan_path).get_fdata()  # HU values (3D volume)

    # Convert HU to density (g/cm³)
    # density_map = 1.0 + (hu / 1000.0)
    density_map = np.clip(((hu + 1000) / 1000.0), a_min=0.0, a_max=None)

    # Extract the coordinates of the vertices from the mesh
    vertex_coordinates = mesh.geometry.x  # Shape: (n_vertices, 3)

    # Build a KD-tree for efficient spatial queries (find the nearest voxel to each vertex)
    grid_shape = ct_data.shape
    grid_coords = np.array(np.unravel_index(np.arange(np.prod(grid_shape)), grid_shape)).T  # (n_voxels, 3)
    kd_tree = cKDTree(grid_coords)

    # Initialize array to store the updated density for each vertex
    updated_vertex_density = np.zeros(len(vertex_coordinates))

    # Define the neighborhood size (kernel size for averaging)
    N = 2  # This defines the neighborhood size
    kernel_size = 2 * N + 1  # Example: 5x5x5 neighborhood for 3D

    # Loop through each vertex and compute the new density
    for i, vertex in enumerate(vertex_coordinates):
        # Find the nearest voxel in the CT scan grid using KD-tree
        _, idx_nearest = kd_tree.query(vertex)
        nearest_voxel = grid_coords[idx_nearest]

        # Get the neighborhood of this voxel (get nearby voxel indices)
        neighborhood_indices = kd_tree.query_ball_point(nearest_voxel, N)  # Get indices of the neighborhood

        # Flatten the neighborhood_indices to a 1D array
        neighborhood_indices = np.array(neighborhood_indices).flatten()

        # Get the corresponding 3D coordinates of the neighboring voxels
        neighborhood_voxels = grid_coords[neighborhood_indices]

        # Extract the corresponding densities from the density map
        neighborhood_densities = density_map[tuple(neighborhood_voxels.T)]  # Assuming 3D indexing

        # Compute the average density of the vertex and its neighborhood
        if np.mean(neighborhood_densities) < 0.0:
            updated_vertex_density[i] = 0.0
        else:
            updated_vertex_density[i] = np.mean(neighborhood_densities)  # Update the density of the vertex

    # Return the array of updated densities for each vertex
    return updated_vertex_density

def plot_density_on_vertices(vertex_coordinates, vertex_densities):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        vertex_coordinates[:, 0],
        vertex_coordinates[:, 1],
        vertex_coordinates[:, 2],
        c=vertex_densities,
        cmap='viridis'
    )
    fig.colorbar(scatter, ax=ax, label='Average Density (g/cm³)')
    ax.set_title('Average Tissue Density at Mesh Vertices')
    plt.savefig("aaa_density_at_vertices.png", dpi=300)
    
def setup_solver(a, L, bcs):
    # PETSc options to avoid preallocation issues
    opts = PETSc.Options()
    opts["mat_mumps_icntl_14"] = 200  # Increase memory allocation for LU decomposition
    opts["mat_mumps_icntl_7"] = 2     # Increase verbosity for debugging
    
    # Advanced solver setup for different configurations
    solver = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, 
        petsc_options={
            "ksp_type": "cg",  # Conjugate Gradient solver "cg",
            "pc_type": "hypre",  # Geometric Algebraic Multigrid preconditioner "gamg"
            "ksp_rtol": 1e-6,  # Relax residual tolerance
            "mg_levels_ksp_type": "chebyshev",  # Smoother on levels
            "mg_levels_pc_type": "jacobi",  # Preconditioner on levels
            "ksp_monitor": None,  # Optional: monitor convergence
        }
    )
    return solver

# Stress tensor (3D)
def sigma(u, mu, lam):
    epsilon = ufl.sym(ufl.grad(u))  # Strain tensor
    return lam * ufl.tr(epsilon) * ufl.Identity(len(u)) + 2 * mu * epsilon

def material_property(E, nu):
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lam

def apply_centremass_bc(mesh, densities, V, density_threshold=2.0):
    """
    Applies Dirichlet boundary conditions to vertices in the mesh based on density:
    - Vertices with density >= density_threshold will have zero displacement (fixed).
    - The center of mass will also have zero displacement.
    
    Parameters:
    - mesh: The finite element mesh
    - V: Vector function space for displacement
    - density_threshold: The threshold density for applying the boundary condition (default: 2.0)
    
    Returns:
    - List of DirichletBC objects fixing displacement for selected vertices
    """
    # Step 1: Get mesh vertex coordinates and corresponding densities
    coords = mesh.geometry.x
    
    # Step 2: Identify vertices with density >= threshold
    fixed_vertices_indices = np.where(densities >= density_threshold)[0]
    print(f"[INFO] Applying BC to {len(fixed_vertices_indices)} vertices with density >= {density_threshold} g/cm³")

    # Step 3: Calculate the center of mass of the mesh (average position of all vertices weighted by density)
    weighted_coords = coords * densities[:, np.newaxis]  # Weight coordinates by density
    center_of_mass = weighted_coords.sum(axis=0) / densities.sum()  # Weighted average position
    
    # Step 4: Find the index of the vertex closest to the center of mass
    distances = np.linalg.norm(coords - center_of_mass, axis=1)
    center_of_mass_index = np.argmin(distances)
    print(f"[INFO] Applying BC to center of mass at index {center_of_mass_index}, position {center_of_mass}")

    # Step 5: Combine the selected vertices and the center of mass vertex
    all_fixed_indices = np.concatenate((fixed_vertices_indices, [center_of_mass_index]))
    
    # Step 6: Apply Dirichlet BCs (u = 0) at the selected vertices for each component (X, Y, Z)
    bcs = []
    for i in range(3):  # X, Y, Z
        dofs = fem.locate_dofs_topological(V.sub(i), 0, all_fixed_indices)
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V.sub(i))
        bcs.append(bc)

    return bcs

def direction_calc(boundary_vertex_indices, vertex_coords, boundary_coords):
    # Build KDTree with boundary coords
    kdtree = KDTree(boundary_coords)

    # For each vertex, find nearest boundary point
    distances, closest_ids = kdtree.query(vertex_coords)
    closest_boundary_coords = boundary_coords[closest_ids]

    # Compute directions
    directions = closest_boundary_coords - vertex_coords
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit_directions = np.divide(directions, norms, out=np.zeros_like(directions), where=norms != 0)

    # Compute inward directions for boundary vertices only
    # Get interior (non-boundary) vertex indices
    all_vertex_indices = np.arange(len(vertex_coords))
    interior_vertex_mask = np.ones_like(all_vertex_indices, dtype=bool)
    interior_vertex_mask[boundary_vertex_indices] = False
    interior_vertex_indices = all_vertex_indices[interior_vertex_mask]

    # Get coordinates of interior vertices
    interior_coords = vertex_coords[interior_vertex_indices]

    # Build KDTree with interior coords
    interior_kdtree = KDTree(interior_coords)

    # Query for closest interior vertex for each boundary vertex
    distances_in, closest_interior_ids = interior_kdtree.query(boundary_coords)
    closest_interior_coords = interior_coords[closest_interior_ids]

    # Compute directions from boundary to interior
    inward_vectors = closest_interior_coords - boundary_coords
    inward_norms = np.linalg.norm(inward_vectors, axis=1, keepdims=True)
    inward_unit_vectors = np.divide(
        inward_vectors, inward_norms, out=np.zeros_like(inward_vectors), where=inward_norms != 0
    )

    # Assign these directions into unit_directions (only for boundary vertices)
    unit_directions[boundary_vertex_indices] = inward_unit_vectors
    
    return unit_directions

def calc_volume_per_vertex(segmentation, mesh):
    spacing = segmentation.header.get_zooms()  
    print("Voxel Spacing (x, y, z):", spacing)
    
    # Calculate voxel volume in mm³
    voxel_volume_mm3 = np.prod(spacing)  # in mm³
    print("Voxel Volume (mm³):", voxel_volume_mm3)
    # Convert to m³
    voxel_volume_m3 = voxel_volume_mm3 * 1e-9  # convert mm³ → m³
    print("Voxel Volume (m³):", voxel_volume_m3)
    # Get the number of voxels in the segmentation mask
    num_voxels = np.sum(segmentation.get_fdata().astype(bool))
    print("Number of voxels in segmentation mask:", num_voxels)
    # Calculate the total volume of the tumor in m³
    total_volume_m3 = num_voxels * voxel_volume_m3
    print("Total Volume (m³):", total_volume_m3)
    # Get the number of vertices in the mesh
    num_vertices = mesh.geometry.x.shape[0]
    print("Number of vertices:", num_vertices)
    # Calculate the volume per vertex
    volume_per_vertex = total_volume_m3 / num_vertices
    print("Volume per vertex (m³):", volume_per_vertex)
    return volume_per_vertex

def updated_density_with_border_sampling_and_interpolation(ct_scan_path, seg_mask_path, mesh):
    import nibabel as nib
    import numpy as np
    from scipy.spatial import cKDTree

    # --- Load CT and segmentation mask ---
    ct_img = nib.load(ct_scan_path)
    ct_data = np.asanyarray(ct_img.dataobj)
    seg_mask = nib.load(seg_mask_path).get_fdata().astype(bool)
    
    # Get voxel spacing
    spacing = ct_img.header.get_zooms()[:3]  # (z, y, x) voxel size
    affine = ct_img.affine

    # HU scaling
    slope = ct_img.header.get("scl_slope", 1.0)
    inter = ct_img.header.get("scl_inter", 0.0)
    if slope is None or np.isnan(slope): slope = 1.0
    if inter is None or np.isnan(inter): inter = 0.0
    hu = ct_data * slope + inter
    density_map = 1.0 + (hu / 1000.0)  # HU to density

    # --- Convert mesh coordinates to voxel space ---
    vertex_world = mesh.geometry.x  # shape (n_vertices, 3)
    vertex_voxel = np.linalg.inv(affine)[:3, :3] @ vertex_world.T + np.linalg.inv(affine)[:3, 3:4]
    vertex_voxel = vertex_voxel.T

    # Create KDTree of all voxel indices (to find neighbors)
    grid_shape = seg_mask.shape
    grid_coords = np.argwhere(np.ones_like(seg_mask))  # full grid coordinates
    grid_tree = cKDTree(grid_coords)

    # Get mask voxel indices for segmentation
    inside_voxels = np.argwhere(seg_mask)
    inside_tree = cKDTree(inside_voxels)

    # Build result array
    updated_vertex_density = np.zeros(len(vertex_voxel))
    inside_vertex_mask = np.zeros(len(vertex_voxel), dtype=bool)

    # Neighborhood kernel size
    N = 2  # for 5x5x5
    neighborhood_radius = N * max(spacing)

    # --- Loop over each vertex ---
    for i, v in enumerate(vertex_voxel):
        # Check if inside segmentation (by nearest voxel test)
        voxel_idx = np.round(v).astype(int)
        if np.all((voxel_idx >= 0) & (voxel_idx < seg_mask.shape)) and seg_mask[tuple(voxel_idx)]:
            inside_vertex_mask[i] = True

            # Extract local neighborhood within N
            neighborhood = []
            for dz in range(-N, N+1):
                for dy in range(-N, N+1):
                    for dx in range(-N, N+1):
                        ni = voxel_idx + np.array([dz, dy, dx])
                        if np.all((ni >= 0) & (ni < seg_mask.shape)):
                            if seg_mask[tuple(ni)]:
                                neighborhood.append(tuple(ni))

            if neighborhood:
                densities = [density_map[pt] for pt in neighborhood]
                updated_vertex_density[i] = np.mean(densities)
            else:
                updated_vertex_density[i] = 0.0  # fallback

        else:
            # Outside segmentation: interpolate from nearest inside point
            dist, idx = inside_tree.query(v, k=5)  # nearest 5 inside voxels
            weights = 1 / (dist + 1e-5)
            weights /= np.sum(weights)
            densities = density_map[tuple(inside_voxels[idx].T)]
            updated_vertex_density[i] = np.dot(weights, densities)

    # === Optionally extract border vertices (distance to seg mask edge) ===
    from scipy.ndimage import distance_transform_edt
    inverse_mask = ~seg_mask
    dist_map = distance_transform_edt(inverse_mask, sampling=spacing)
    surface_mask = (dist_map <= neighborhood_radius)

    surface_voxels = np.argwhere(surface_mask)
    surface_tree = cKDTree(surface_voxels)

    border_vertex_mask = np.array([
        surface_mask[tuple(np.round(v).astype(int))] if np.all((v >= 0) & (v < seg_mask.shape)) else False
        for v in vertex_voxel
    ])

    return updated_vertex_density

# --- Load tumor mesh ---
mesh = mesh_load("tumor_final")

# Compute average density in tumor at vertices
print("[INFO] Calculating average density in tumor at mesh vertices...")
density = avg_density_in_tumor_at_vertices_new(ct_scan_path, seg_mask_path, mesh)
# plot_density_on_vertices(mesh.geometry.x, density)

relaxation_steps = 5

# Step 1: Define material properties (Young's modulus, Poisson's ratio)
E = 1e5  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio
mu, lam = material_property(E, nu)  # Define the material properties (mu, lam)
k = 5000 # a scaling constant (e.g. 1000–5000 to get Pascals)
rho0 = 1.0 # Reference pressure (Pa)

for i in range(0, 3):
    if i > 0:
        mesh_new = mesh_load("bulged_mesh_new")
        new_density = avg_density_in_tumor_at_vertices(ct_scan_path, seg_mask_path, mesh_new)
        mesh = mesh_new
        density = new_density
        print(f"[INFO] Updated density on new mesh: {new_density[:5]}")
        plot_density_on_vertices(mesh.geometry.x, density)
    else:
        print(f"[INFO] Using original density: {density[:5]}")
    
    # --- Define function spaces and FEM setup ---
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,))) # vector space
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # scalar space

    # Define displacement trial/test
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_sol = Function(V)
    pressure_fem = Function(Q)

    # Get exterior facets and vertices
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    boundary_facets = exterior_facet_indices(mesh.topology)
    boundary_vertex_indices = compute_incident_entities(mesh.topology, boundary_facets, 2, 0)

    # Get coordinates
    vertex_coords = mesh.geometry.x
    boundary_coords = vertex_coords[boundary_vertex_indices]

    unit_directions = direction_calc(boundary_vertex_indices, vertex_coords, boundary_coords)

    # Count number of non-zero direction vectors
    non_zero_mask = ~np.all(unit_directions == 0, axis=1)
    num_non_zero = np.count_nonzero(non_zero_mask)
    print(f"[INFO] Found {num_non_zero} vertices with non-zero direction vectors.")

    # calculate pressure
    pressure = k * (density - rho0)  # Pressure in Pa
    pressure = np.clip(pressure, 0, None)
    pressure_fem.x.array[:] = pressure

    # Evaluate density at each vertex (interpolation, or direct .x.array if aligned)
    densities = pressure_fem.x.array  # shape (num_vertices,)
    force_vectors = unit_directions * densities[:, np.newaxis]  # shape (num_vertices, 3)

    print("Non-zero force vectors:", np.count_nonzero(np.linalg.norm(force_vectors, axis=1)))
    print("Force vector magnitudes:", np.linalg.norm(force_vectors, axis=1)[:10])
    
    np.save("pressure.npy", pressure)
    np.save("unit_directions.npy", unit_directions)

    # Create a function for force and assign the force vectors to it
    force_function = Function(V)  # V should be a vector function space, e.g., ("Lagrange", 3)
    force_function.x.array[::3] = force_vectors[:, 0]  # X-component of force
    force_function.x.array[1::3] = force_vectors[:, 1]  # Y-component of force
    force_function.x.array[2::3] = force_vectors[:, 2]  # Z-component of force

    # Print the force vectors to check
    print(f"[INFO] Force vectors:\n{force_vectors[:5]}")

    # --- Define weak form ---
    dx = ufl.Measure("dx", domain=mesh)
    # We can now apply the force at the vertices, modifying the weak form:
    a = ufl.inner(sigma(u, mu, lam), ufl.grad(v)) * dx
    L = ufl.dot(force_function, v) * dx  # Apply force vector to weak form

    bcs = apply_centremass_bc(mesh, densities, V, density_threshold=2.0)
    print(f"[INFO] Number of Dirichlet vertices at bottom: {len(bcs)}")

    for it in range(relaxation_steps):
        print(f"\t[INFO] Relaxation iteration {it+1}/{relaxation_steps}")
        problem = setup_solver(a, L, bcs)
        u_sol = problem.solve()
        print(u_sol.x.array[:5])

    print(f"[INFO] Displacement min/max: {u_sol.x.array.min()}, {u_sol.x.array.max()}")

    # Extract displacement vectors from solution
    displacement_vectors = u_sol.x.array.reshape(-1, 3)
    np.save("displacement_vectors.npy", displacement_vectors)
    
    magnitudes = np.linalg.norm(displacement_vectors, axis=1)
    np.save("magnitudes.npy", magnitudes)

    # Normalize magnitudes into per-vertex probabilities (max 1)
    probabilities = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))

    # Save per-vertex probability array
    np.save("probabilities.npy", probabilities)
    np.savetxt("vertex_displacement_probabilities.csv", probabilities, delimiter=",")
    print("[INFO] Saved vertex displacement probabilities.")

    def select_bulges(
        original_points, new_points, displacement_vectors,
        displacement_threshold=0.03, neighbor_radius=None, max_region_size=100
    ):
        """
        Select and grow bulge regions from displaced points, preserving original connectivity.

        Args:
            original_points (np.ndarray): Nx3 array of original point positions.
            new_points (np.ndarray): Nx3 array of displaced point positions.
            displacement_vectors (np.ndarray): Nx3 array of displacement vectors.
            displacement_threshold (float): Displacement threshold to start a bulge.
            neighbor_radius (float or None): Radius for region growing. Auto-estimated if None.
            max_region_size (int): Max number of points per bulge region.

        Returns:
            combined_points (np.ndarray): Original points + bulged points.
            connection_map (List[Tuple[int, int]]): Index pairs for original ↔ bulged connections.
        """
        displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)

        if neighbor_radius is None:
            tree = cKDTree(original_points)
            dists, _ = tree.query(original_points, k=2)
            neighbor_radius = np.mean(dists[:, 1]) * 1.5
            print(f"[INFO] Estimated neighbor radius: {neighbor_radius:.6f}")

        seed_indices = np.where(displacement_magnitudes > displacement_threshold)[0]
        print(f"[INFO] {len(seed_indices)} seed points selected (threshold = {displacement_threshold}).")

        if len(seed_indices) == 0:
            return original_points.copy(), []

        tree = cKDTree(new_points)
        visited = np.zeros(len(new_points), dtype=bool)
        extended_indices = set()

        # Region growing with limit
        for seed_idx in seed_indices:
            if visited[seed_idx]:
                continue
            region = set()
            queue = deque([seed_idx])
            while queue and len(region) < max_region_size:
                idx = queue.popleft()
                if visited[idx]:
                    continue
                visited[idx] = True
                region.add(idx)
                neighbors = tree.query_ball_point(new_points[idx], r=neighbor_radius)
                for nbr in neighbors:
                    if not visited[nbr]:
                        queue.append(nbr)
            extended_indices.update(region)

        # Build new points and connection map
        connection_map = []
        bulge_points = []
        index_mapping = {}

        for idx in sorted(extended_indices):
            new_pt = new_points[idx]
            bulge_points.append(new_pt)
            new_idx = len(original_points) + len(bulge_points) - 1
            connection_map.append((idx, new_idx))  # connection: original idx → bulge idx
            index_mapping[idx] = new_idx

        combined_points = np.vstack([original_points, bulge_points])
        print(f"[INFO] {len(bulge_points)} new bulge points added.")
        print(f"[INFO] {len(connection_map)} connections mapped from original to bulges.")

        return combined_points, bulge_points, connection_map

    def estimate_neighbor_radius(points):
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=2)  # k=1 is self
        neighbor_radius = 2.0 * np.mean(dists[:, 1])
        print(f"[INFO] Estimated neighbor radius: {neighbor_radius:.6f}")
    
        return neighbor_radius
    
    def check_and_redistribute_points(original_points, new_points, min_distance_factor=1.05, max_disp_scale=0.1):
        """
        Ensure that combined points (original + new) are spaced far enough from each other by redistributing the new points.

        Args:
            original_points (np.ndarray): Array of original points (shape: N, 3).
            new_points (np.ndarray): Array of new points (shape: M, 3).
            min_distance_factor (float): Factor to ensure minimum distance between points.
            max_disp_scale (float): Maximum displacement scale for moving points that are too close.

        Returns:
            np.ndarray: Adjusted points array (original + new) with sufficient spacing.
        """
        # Combine the original and new points
        combined_points = np.vstack([original_points, new_points])

        # Create a k-d tree for efficient nearest neighbor search
        tree = cKDTree(combined_points)

        # Calculate distances between all points (excluding themselves)
        distances, indices = tree.query(combined_points, k=2)  # k=2 includes the point itself

        # We are interested in the distances to the nearest neighbor (excluding the point itself)
        distances = distances[:, 1]

        # Calculate the average spacing between points in the combined set
        avg_distance = np.mean(distances)
        print(f"[INFO] Average distance between combined points: {avg_distance:.4f}")

        # Minimum allowed distance based on the average distance with a factor
        min_distance = min_distance_factor * avg_distance

        # Move points that are too close to each other
        adjusted_points = combined_points.copy()

        # Iterate over all point pairs and move points that are too close
        for i, dist in enumerate(distances):
            if dist < min_distance:
                # Get the index of the point that is too close
                idx_to_move = indices[i, 1]

                # Move the point slightly away from the conflicting point
                displacement_vector = np.random.normal(scale=max_disp_scale * avg_distance, size=3)
                adjusted_points[idx_to_move] += displacement_vector

        return adjusted_points
    
    original_points = mesh.geometry.x 

    displacement = u_sol.x.array.reshape(-1, 3)
    new_points = original_points + displacement
    new_points = np.unique(new_points, axis=0)

    multiplier = 5
    displacement_threshold = np.mean(np.linalg.norm(displacement, axis=1)) + multiplier * np.std(np.linalg.norm(displacement, axis=1))
    
    neighbor_radius = estimate_neighbor_radius(original_points)

    buldge_new_points, bulge_points, connection_map = select_bulges(original_points=mesh.geometry.x, new_points=new_points, displacement_vectors=displacement_vectors, displacement_threshold=displacement_threshold, neighbor_radius=neighbor_radius)

    np.save("bulge_points.npy", bulge_points)

    # Add remeshing step
    # Now make new mesh using GMSH
    # gmsh_new_mesh("mesh_with_connected_points", "bulged_mesh_new")
    # print(f"[INFO] GMSH mesh generation completed.")