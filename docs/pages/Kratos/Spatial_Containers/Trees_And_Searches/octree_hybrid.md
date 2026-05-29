---
title: OctreeHybrid
keywords: search spatial_container octree adaptive hex mesh
tags: [search spatial_container octree]
sidebar: kratos_spatial_containers
summary: A complete linearised octree with 2:1 strong-balance enforcement and dual all-hex mesh extraction.
---

# OctreeHybrid

## Description

`OctreeHybrid` is a complete, linearised octree for 3D adaptive mesh refinement.
Unlike the classic Kratos `OcTree` (which is pointer-based), `OctreeHybrid` uses a
**flat array indexed by a space-filling level–grid key**, yielding O(1) cell lookup
by position, direct access to any cell's parent/children without pointer traversal,
and cache-friendly memory layout.

Its primary purpose is to serve as the primal mesh for the
[`OctreeHybridMeshUtility`](../../Utilities/General/octree_hybrid_mesh_utility.md)
which extracts a conforming all-hexahedral dual mesh — the **HybridOctree_Hex**
algorithm.

---

## Key features

- **Complete-octree storage**: the underlying array is dense at every refinement level; cells are subdivided by setting a flag.
- **2:1 strong-balance** enforcement: `StrongConstrain2To1` guarantees that adjacent leaf cells differ by at most one level, a prerequisite for the dual hex templates.
- **O(N·depth) face-adjacency**: `pGetCellNormalized` answers "which leaf cell contains a normalised point?" in O(depth) time; used by the mesh utility to build the face-adjacency graph in O(N·depth) instead of O(N²).
- **Level convention**: `level = 0` is the coarsest (root), `level = MAX_DEPTH` is the finest. Note this is the **opposite** of `OctreeBinaryCell`.
- **Surface-driven refinement**: `BuildFromSurfaceMesh` refines cells near the input triangular surface up to a requested `RefinementDepth`.

---

## Implementation

Found in:
- [`kratos/spatial_containers/octree_hybrid.h`](https://github.com/KratosMultiphysics/Kratos/blob/master/kratos/spatial_containers/octree_hybrid.h)
- [`kratos/spatial_containers/octree_hybrid_cell.h`](https://github.com/KratosMultiphysics/Kratos/blob/master/kratos/spatial_containers/octree_hybrid_cell.h)
- [`kratos/spatial_containers/octree_hybrid_configure.h`](https://github.com/KratosMultiphysics/Kratos/blob/master/kratos/spatial_containers/octree_hybrid_configure.h)

---

## Cell layout

Each cell at level `l` has an integer grid index `(gx, gy, gz)` in the range `[0, 2^l)` along each axis.
A cell at level `l` with index `(gx, gy, gz)` occupies the normalised sub-cube:

```
[gx/2^l, (gx+1)/2^l] × [gy/2^l, (gy+1)/2^l] × [gz/2^l, (gz+1)/2^l]
```

### Corner numbering

```
     7 ---- 6
    /|      /|
   4 ---- 5  |
   |  3 --|- 2
   | /    | /
   0 ---- 1
```

| Corner | (dx, dy, dz) |
|--------|-------------|
| 0      | (0, 0, 0)   |
| 1      | (1, 0, 0)   |
| 2      | (1, 1, 0)   |
| 3      | (0, 1, 0)   |
| 4      | (0, 0, 1)   |
| 5      | (1, 0, 1)   |
| 6      | (1, 1, 1)   |
| 7      | (0, 1, 1)   |

Corner 0 = `(gx, gy, gz)` (minimum); corner 6 = `(gx+1, gy+1, gz+1)` (maximum).

---

## `OctreeHybridCell` API

```cpp
int  GetLevel()  const;   // refinement level (0 = root, MAX_DEPTH = finest)
int  GetGridX()  const;   // integer x-grid index at this level
int  GetGridY()  const;   // integer y-grid index at this level
int  GetGridZ()  const;   // integer z-grid index at this level
void GetMinPointNormalized(double min_point[3]) const;  // corner 0, normalised [0,1]
void GetMaxPointNormalized(double max_point[3]) const;  // corner 6, normalised [0,1]
```

---

## `OctreeHybrid` API

```cpp
// Subdivide the cell containing normalised point `pt`
void SubdivideCell(const double pt[3]);

// Subdivide a specific cell by its id and level
void SubdivideCellByIdAndLevel(std::size_t id, int level);

// Return the leaf cell containing normalised point `pt`, or nullptr
CellType* pGetCellNormalized(const double pt[3]);

// Enforce the 2:1 balance condition (must be called before dual mesh extraction)
void StrongConstrain2To1();

// Fill `leaves` with all current leaf cells
void GetAllLeafCells(std::vector<CellType*>& leaves);
```

---

## Example usage

### C++

```cpp
#include "spatial_containers/octree_hybrid.h"
#include "spatial_containers/octree_hybrid_cell.h"
#include "spatial_containers/octree_hybrid_configure.h"

using Configure = OctreeHybridKratosConfiguration;
using Cell      = OctreeHybridCell<Configure>;
using Octree    = OctreeHybrid<Cell>;

auto p_octree = std::make_unique<Octree>();

// Refine around a point
double pt[3] = {0.3, 0.3, 0.3};
for (int i = 0; i < 5; ++i)   // 5 levels of refinement
    p_octree->SubdivideCell(pt);

// Enforce 2:1 balance
p_octree->StrongConstrain2To1();

// Iterate over leaves
std::vector<Cell*> leaves;
p_octree->GetAllLeafCells(leaves);
std::cout << "Leaves: " << leaves.size() << "\n";
```

### Python

Refinement and mesh generation are driven through `OctreeHybridMeshUtility`:

```python
import KratosMultiphysics as KM

model      = KM.Model()
surface_mp = model.CreateModelPart("Surface")
# ... populate surface_mp with triangles ...

# Build octree + enforce 2:1 balance
octree = KM.OctreeHybridMeshUtility.BuildFromSurfaceMesh(surface_mp, 8)
```

---

## 2:1 Balance (`StrongConstrain2To1`)

The 2:1 strong-balance condition requires that any two adjacent leaf cells differ by
at most one refinement level.  The algorithm iterates until convergence:

1. For every leaf cell `L`, find all 8 cells sharing its minimum-corner grid vertex.
2. If the maximum level among those 8 neighbours exceeds the level of any one cell
   `C` by more than 1, mark `C` and all its 7 siblings for subdivision.
3. Add the children of all newly-subdivided cells to the work list.
4. Remove already-subdivided cells from the leaf list and update the leaf count.
5. Repeat until no new subdivisions occur.

This is a prerequisite for correct dual hex mesh generation: the transition templates
are only defined for level differences of exactly 1.

---

## See also

- [OcTree](octree.md) — the classic pointer-based octree
- [OctreeHybridMeshUtility](../../Utilities/General/octree_hybrid_mesh_utility.md) — dual all-hex mesh extraction
