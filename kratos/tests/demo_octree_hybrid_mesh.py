"""
Demo: build an adaptive OctreeHybrid hex mesh from an STL surface and write the
result to a legacy VTK file for visualisation in Paraview.

Accepts both ASCII and binary STL files.

Usage:
    python3 demo_octree_hybrid_mesh.py [path/to/surface.stl] [refinement_depth]

Defaults to Bunny-LowPoly.stl at depth 5 when no arguments are given.

Output:
    octree_hex_mesh.vtk  — adaptive hex mesh, colour by "level" in Paraview
"""

import gc
import os
import struct
import sys

# ---------------------------------------------------------------------------
# 1. Bootstrap Kratos path
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir  = os.path.realpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir,
                                            "build", "Release"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import KratosMultiphysics as KM

# ---------------------------------------------------------------------------
# 2. Helper: convert binary STL → ASCII STL
# ---------------------------------------------------------------------------

def is_binary_stl(path: str) -> bool:
    """Return True when the file is a binary (not ASCII) STL."""
    with open(path, "rb") as f:
        header = f.read(80)
    # ASCII STL files begin with the word "solid" (possibly preceded by BOM).
    try:
        return not header.lstrip().decode("ascii", errors="replace").startswith("solid")
    except Exception:
        return True


def binary_stl_to_ascii(src: str, dst: str) -> None:
    """Convert a binary STL file to ASCII format."""
    with open(src, "rb") as f:
        f.read(80)  # skip 80-byte header
        n_triangles = struct.unpack("<I", f.read(4))[0]
        with open(dst, "w") as out:
            out.write("solid converted\n")
            for _ in range(n_triangles):
                nx, ny, nz = struct.unpack("<fff", f.read(12))
                v1 = struct.unpack("<fff", f.read(12))
                v2 = struct.unpack("<fff", f.read(12))
                v3 = struct.unpack("<fff", f.read(12))
                f.read(2)  # attribute byte count
                out.write(f"  facet normal {nx} {ny} {nz}\n")
                out.write( "    outer loop\n")
                out.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                out.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                out.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
                out.write( "    endloop\n")
                out.write( "  endfacet\n")
            out.write("endsolid converted\n")
    print(f"  Converted binary STL → {dst}  ({n_triangles} triangles)")


# ---------------------------------------------------------------------------
# 3. Parse arguments
# ---------------------------------------------------------------------------
stl_input = sys.argv[1] if len(sys.argv) > 1 else os.path.join(script_dir, "Bunny-LowPoly.stl")
refinement_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8

print(f"Input STL     : {stl_input}")
print(f"Refinement    : depth {refinement_depth}")

# ---------------------------------------------------------------------------
# 4. Convert binary → ASCII if necessary
# ---------------------------------------------------------------------------
ascii_stl = stl_input
tmp_ascii  = None

if is_binary_stl(stl_input):
    tmp_ascii  = os.path.join(script_dir, "_demo_ascii_tmp.stl")
    binary_stl_to_ascii(stl_input, tmp_ascii)
    ascii_stl = tmp_ascii

# ---------------------------------------------------------------------------
# 5. Read surface mesh via StlIO (ASCII)
# ---------------------------------------------------------------------------
model      = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

stl_io = KM.StlIO(ascii_stl, KM.Parameters('{"open_mode": "read"}'))
stl_io.ReadModelPart(surface_mp)
del stl_io
gc.collect()

print(f"  Nodes      : {surface_mp.NumberOfNodes()}")
print(f"  Geometries : {surface_mp.NumberOfGeometries()}")

if tmp_ascii and os.path.exists(tmp_ascii):
    os.remove(tmp_ascii)

# ---------------------------------------------------------------------------
# 6. Build the adaptive octree hex mesh and write VTK
# ---------------------------------------------------------------------------
output_vtk = "octree_hex_mesh.vtk"

print(f"\nBuilding OctreeHybrid and writing {output_vtk} …")
KM.OctreeHybridMeshUtility.BuildAndWriteVtk(surface_mp, output_vtk, refinement_depth)

# Report mesh stats from the VTK file
with open(output_vtk) as f:
    for line in f:
        if line.startswith("POINTS") or line.startswith("CELLS "):
            print(" ", line.rstrip())
        if line.startswith("CELL_TYPES"):
            break

print(f"\nOpen '{output_vtk}' in Paraview and colour by 'level' to see adaptive refinement.")
