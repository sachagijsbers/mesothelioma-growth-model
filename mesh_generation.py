import nibabel as nib
from skimage.measure import marching_cubes
import meshio
import gmsh

# Load segmentation and CT data
segmentation = nib.load("segmentation.nii.gz")

# Get geometry information
spacing = segmentation.header.get_zooms()  
print("Voxel Spacing (x, y, z):", spacing)
segmentation = segmentation.get_fdata()
verts, faces, _, _ = marching_cubes(segmentation, level=0.8, spacing=spacing)

# Save the surface as STL
meshio.write_points_cells(
    "tumor_surface.stl",
    verts,
    [("triangle", faces)]
)

# Load, simplify and repair the STL file using trimesh
mesh = trimesh.load_mesh("tumor_surface.stl")

# Simplify surface: reduce face count by 50%
mesh = mesh.simplify_quadric_decimation(0.5)

# Check if the mesh is watertight and repair it if necessary
if not mesh.is_watertight:
    print("Mesh is not watertight. Attempting to repair...")
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    # Remove duplicate faces and degenerate faces
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    print("Repaired mesh saved as 'repaired_tumor_surface_sec.stl'.")
else:
    print("Mesh is already watertight.")

mesh.update_faces(mesh.unique_faces())
mesh.export("repaired_tumor_surface.stl")

# Initialize GMSH and import the repaired STL file
gmsh.initialize()
gmsh.model.add("tumor_mesh")

# Import the repaired STL file into GMSH
gmsh.merge("repaired_tumor_surface.stl")

# Synchronize the geometry
gmsh.model.geo.synchronize()

# Identify the surfaces imported from the STL
entities = gmsh.model.getEntities(dim=2)
if not entities:
    raise ValueError("No entities found")
  
# Remove duplicates
surface_ids = list(set(entity[1] for entity in entities))  

# Create a surface loop and volume in GMSH
surface_loop = gmsh.model.geo.addSurfaceLoop(surface_ids)
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
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)  # Adapt to curvature
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Smooth transition from boundary to interior
gmsh.option.setNumber("Mesh.MeshSizeMax", 3.0)  # Increase the mesh size to reduce the element count
gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Use Netgen
gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.4)  # Stop once mesh quality reaches 0.4

# Generate the tetrahedral mesh
gmsh.model.mesh.generate(3)

gmsh.write("tumor.msh")
print("Generated and saved 3D mesh.")

# Export the mesh to VTK for adding density
gmsh.write("tumor_tetrahedral_mesh_sec.vtk")
