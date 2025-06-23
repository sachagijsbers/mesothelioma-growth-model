# Mesothelioma Growth Model
## CT-Based Biomechanical Tumor Growth Simulation Using Finite Element Modeling
This project provides a complete biomechanical modeling pipeline to simulate and analyze the growth of mesothelioma, a cancer typically forming in the pleura surrounding the lungs. The model uses CT imaging data and tumor segmentations to apply tissue-density-informed internal forces and simulate tumor deformation and growth patterns.

## Overview
The model simulates tumor deformation using linear elasticity in DolfinX, guided by tissue density derived from CT scans. It integrates image-based geometry extraction, mesh processing, and finite element modeling to understand how mechanical forces—such as pressure from surrounding tissues—affect the direction and probability of tumor expansion.

## What You Need
To run the full pipeline, you will need:
* A CT scan of the thorax (e.g., in NIfTI .nii.gz format)
* A tumor segmentation mask (same space as CT)

### Python Environment
You will need Python 3.10+ and the following packages:
numpy
scipy
matplotlib
pyvista
meshio
pandas
open3d
trimesh
imageio
nibabel
scikit-image
gmsh (CLI, for meshing)
dolfinx (via Docker or compiled locally for FEM)

## Project Structure
### Tumor Mesh Generation
mesh_generation.py: generates a high-quality tetrahedral volume mesh from a tumor segmentation (in NIfTI format), preparing it for biomechanical modeling and finite element simulation. It includes:

* Surface extraction from segmentation
* Mesh simplification and repair
* Volume meshing with GMSH
* Exporting to .msh and .vtk for simulation use

### Tumor Remeshing
remeshing.py: provides tools to process 3D point clouds (e.g., from medical image segmentations or tumor surfaces) into high-quality surface or volumetric meshes suitable for simulation, visualization, or further processing. It includes methods for alpha shape and Poisson surface reconstruction, as well as .geo file export for use with GMSH.

* write_geo_from_points(...): Generates a 2D triangulated .geo file from sparse 3D points, using Delaunay triangulation and spatial connectivity. This is useful for generating surface loops and volume definitions in GMSH.

* generate_alpha_mesh(...): Constructs a smooth watertight triangle mesh from a point cloud using Open3D's alpha shape reconstruction. Includes optional voxel downsampling, smoothing, simplification, and manifold repair via MeshFix.

* poisson_surface_reconstruction(...): Uses Poisson surface reconstruction to create a watertight mesh from a point cloud with normals. Optionally downsamples points, cleans up the mesh, and exports as STL.

### MAIN PIPELINE: Biomechanical Simulation
This project simulates the deformation of a tumor mesh using CT-derived density-based internal forces via linear elasticity in DolfinX. The main model loop can be found in fenicsx_model_implementation.py. It incorporates:

* Directionally resolved force vectors computed from CT-scan tissue density.
* Application of density-dependent pressure to the tumor boundary.
* Iterative solution of the elasticity problem using FEM.
* Detection and propagation of displacement-driven tumor bulges.
* Export of results including displacements, probabilities, and bulge regions.
* Optional GMSH remeshing of the displaced tumor geometry.


### Visualisations & Analysis

#### Tumor Density Visualization and Analysis from CT Scans
density_vis.py contains Python scripts to visualize tumor density overlays and histograms from CT scans and segmentation masks using NIfTI files.

* Overlay tumor density on CT axial slices
* Apply 5×5 neighborhood smoothing to enhance visualization
* Zoom into regions of interest
* Generate annotated density histograms for whole scan and tumor region
* Save outputs as high-quality PNG images and export HU data

#### Pressure Distributions Analysis
pressure_vis.py provides tools to analyze and visualize pressure data derived from biomechanical modeling or CT-based simulations. It includes:

* A histogram with a zoomed-in inset of the pressure distribution.
* A plot showing how the mean pressure varies with a scaling constant k.


#### Tumor Growth Probability Visualization
new_point_vis.py visualizes tumor growth modeling data using a variety of interactive and static plots. It analyzes and displays growth probabilities, pressure correlations, and tumor displacement data derived from biomechanical simulations. The goal is to explore regions of high-growth probability and their relationship with mechanical stress (pressure) in a tumor mesh.

## Applications
This framework is useful for:

* Preclinical studies of mesothelioma growth mechanics
* In silico simulations of tumor stress and morphology
* Augmenting radiological analysis with biomechanical insight
* Testing intervention scenarios (e.g., how tissue stiffness affects growth)
