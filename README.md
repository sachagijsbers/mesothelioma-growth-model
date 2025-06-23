# mesothelioma-growth-model
CT-Based Biomechanical Tumor Growth Model for Mesothelioma


Tumor Mesh Generation Pipeline
mesh.py generates a high-quality tetrahedral volume mesh from a tumor segmentation (in NIfTI format), preparing it for biomechanical modeling and finite element simulation. It includes:

* Surface extraction from segmentation
* Mesh simplification and repair
* Volume meshing with GMSH
* Exporting to .msh and .vtk for simulation use


remeshing.py: provides tools to process 3D point clouds (e.g., from medical image segmentations or tumor surfaces) into high-quality surface or volumetric meshes suitable for simulation, visualization, or further processing. It includes methods for alpha shape and Poisson surface reconstruction, as well as .geo file export for use with GMSH.

* write_geo_from_points(...): Generates a 2D triangulated .geo file from sparse 3D points, using Delaunay triangulation and spatial connectivity. This is useful for generating surface loops and volume definitions in GMSH.

* generate_alpha_mesh(...): Constructs a smooth watertight triangle mesh from a point cloud using Open3D's alpha shape reconstruction. Includes optional voxel downsampling, smoothing, simplification, and manifold repair via MeshFix.

* poisson_surface_reconstruction(...): Uses Poisson surface reconstruction to create a watertight mesh from a point cloud with normals. Optionally downsamples points, cleans up the mesh, and exports as STL.


This project simulates the deformation of a tumor mesh using CT-derived density-based internal forces via linear elasticity in DolfinX. The main model loop can be found in fenicsx_model_implementation.py. It incorporates:

* Directionally resolved force vectors computed from CT-scan tissue density.
* Application of density-dependent pressure to the tumor boundary.
* Iterative solution of the elasticity problem using FEM.
* Detection and propagation of displacement-driven tumor bulges.
* Export of results including displacements, probabilities, and bulge regions.
* Optional GMSH remeshing of the displaced tumor geometry.

