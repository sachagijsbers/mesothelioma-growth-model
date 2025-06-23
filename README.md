# mesothelioma-growth-model
CT-Based Biomechanical Tumor Growth Model for Mesothelioma


remeshing.py: provides tools to process 3D point clouds (e.g., from medical image segmentations or tumor surfaces) into high-quality surface or volumetric meshes suitable for simulation, visualization, or further processing. It includes methods for alpha shape and Poisson surface reconstruction, as well as .geo file export for use with GMSH.

* write_geo_from_points(...): Generates a 2D triangulated .geo file from sparse 3D points, using Delaunay triangulation and spatial connectivity. This is useful for generating surface loops and volume definitions in GMSH.

* generate_alpha_mesh(...): Constructs a smooth watertight triangle mesh from a point cloud using Open3D's alpha shape reconstruction. Includes optional voxel downsampling, smoothing, simplification, and manifold repair via MeshFix.

* poisson_surface_reconstruction(...): Uses Poisson surface reconstruction to create a watertight mesh from a point cloud with normals. Optionally downsamples points, cleans up the mesh, and exports as STL.
