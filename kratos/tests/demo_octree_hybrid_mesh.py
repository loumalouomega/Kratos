"""
Demo: build an adaptive OctreeHybrid hex mesh from a surface mesh (sphere skin)
and write the result to a legacy VTK file for visualisation in Paraview.

Usage (from the build directory):
    python3 <path_to_this_file>

Output:
    octree_hex_mesh.vtk   — adaptive hex mesh, coloured by "level" in Paraview
"""

import gc
import os
import sys

# ---------------------------------------------------------------------------
# 1. Bootstrap Kratos path
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir  = os.path.join(script_dir, os.pardir, os.pardir, os.pardir, "build", "Release")
build_dir  = os.path.realpath(build_dir)
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import KratosMultiphysics as KM
from KratosMultiphysics.testing.utilities import ReadModelPart

# ---------------------------------------------------------------------------
# 2. Read the surface mesh
# ---------------------------------------------------------------------------
model = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

# Coarse sphere skin from the Kratos test auxiliary files.
mdpa_path = os.path.join(
    script_dir,
    "auxiliar_files_for_python_unittest",
    "mdpa_files",
    "coarse_sphere_skin"
)

print(f"Reading surface mesh from: {mdpa_path}.mdpa")
ReadModelPart(mdpa_path, surface_mp)
print(f"  Nodes    : {surface_mp.NumberOfNodes()}")
print(f"  Elements : {surface_mp.NumberOfElements()}")
print(f"  Conditions: {surface_mp.NumberOfConditions()}")

# StlIO reads triangles into the Geometries container; ReadModelPart loads the
# .mdpa which populates Elements/Conditions.  The mesh utility reads Nodes
# for the bounding box and Geometries for intersection tests.  When the model
# part was loaded from .mdpa its triangles sit in the Conditions container,
# so we expose them as geometries by registering the condition geometries.
# The simplest path: write to STL and read back so geometries are populated.

stl_path = os.path.join(script_dir, "_demo_sphere.stl")
write_settings = KM.Parameters("""{"open_mode": "write"}""")
stl_io = KM.StlIO(stl_path, write_settings)
stl_io.WriteModelPart(surface_mp)
del stl_io  # flush and close the C++ file stream before reading below
gc.collect()
print(f"STL written to {stl_path}")

# Read the STL back: StlIO populates ModelPart.Geometries with Triangle3D3.
surface_stl_mp = model.CreateModelPart("SurfaceSTL")
surface_stl_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
read_settings = KM.Parameters("""{"open_mode": "read"}""")
stl_io2 = KM.StlIO(stl_path, read_settings)
stl_io2.ReadModelPart(surface_stl_mp)
print(f"STL read back — Geometries: {surface_stl_mp.NumberOfGeometries()}, "
      f"Nodes: {surface_stl_mp.NumberOfNodes()}")

# ---------------------------------------------------------------------------
# 3. Build the adaptive octree hex mesh
# ---------------------------------------------------------------------------
output_vtk = "octree_hex_mesh.vtk"
refinement_depth = 5

print(f"\nBuilding OctreeHybrid (depth={refinement_depth}) and writing {output_vtk} …")
KM.OctreeHybridMeshUtility.BuildAndWriteVtk(surface_stl_mp, output_vtk, refinement_depth)
print(f"Done. Open '{output_vtk}' in Paraview and colour by 'level' to see the adaptive refinement.")

# Clean up the temporary STL.
if os.path.exists(stl_path):
    os.remove(stl_path)
