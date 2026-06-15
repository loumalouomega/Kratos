---
title: OctreeHybridMeshUtility
keywords: mesh hex hexahedral octree adaptive dual
tags: [mesh hexahedral octree utility]
sidebar: kratos_core_utilities
summary: Builds an adaptive OctreeHybrid from a surface and exports a conforming all-hexahedral dual mesh (HybridOctree_Hex algorithm).
---

# OctreeHybridMeshUtility

## Table of contents

1. [What this utility does](#1-what-this-utility-does)
2. [Background: primal vs dual mesh](#2-background-primal-vs-dual-mesh)
3. [Reference and credits](#3-reference-and-credits)
4. [Full algorithm pipeline](#4-full-algorithm-pipeline)
   - 4.1 [Stage 0 — Octree construction](#41-stage-0--octree-construction)
   - 4.2 [Stage 1 — Primal mesh with shared vertices](#42-stage-1--primal-mesh-with-shared-vertices)
   - 4.3 [Stage 2 — Face-adjacency graph](#43-stage-2--face-adjacency-graph)
   - 4.4 [Stage 3 — Plain dual hexes (regular regions)](#44-stage-3--plain-dual-hexes-regular-regions)
   - 4.5 [Stage 4 — Transition templates (mixed-level regions)](#45-stage-4--transition-templates-mixed-level-regions)
   - 4.6 [Stage 5 — Node merging and VTK output](#46-stage-5--node-merging-and-vtk-output)
5. [Transition templates — detailed geometry](#5-transition-templates--detailed-geometry)
   - 5.1 [Face numbering and orientation conventions](#51-face-numbering-and-orientation-conventions)
   - 5.2 [13-element base template (t1Id)](#52-13-element-base-template-t1id)
   - 5.3 [4-element edge template (t2Id / t22Id)](#53-4-element-edge-template-t2id--t22id)
   - 5.4 [3-element edge template (t3Id / t32Id / t33Id / t34Id)](#54-3-element-edge-template-t3id--t32id--t33id--t34id)
   - 5.5 [5-element corner template (t4Id / t42Id / t43Id / t44Id)](#55-5-element-corner-template-t4id--t42id--t43id--t44id)
   - 5.6 [Template emission rules (collectNum / consume_at)](#56-template-emission-rules-collectnum--consume_at)
6. [Implementation — Kratos port](#6-implementation--kratos-port)
   - 6.1 [Key differences from the reference code](#61-key-differences-from-the-reference-code)
   - 6.2 [Node-merge spatial hash](#62-node-merge-spatial-hash)
   - 6.3 [Face-adjacency via pGetCellNormalized](#63-face-adjacency-via-pgetcellnormalized)
   - 6.4 [Bugs found and fixed during porting](#64-bugs-found-and-fixed-during-porting)
7. [API reference](#7-api-reference)
8. [Key constants and lookup tables](#8-key-constants-and-lookup-tables)
9. [Verification and testing](#9-verification-and-testing)
10. [Usage examples](#10-usage-examples)
11. [Performance notes](#11-performance-notes)
12. [Known limitations](#12-known-limitations)
13. [The full reference pipeline and what is *not* ported](#13-the-full-reference-pipeline-and-what-is-not-ported)
    - 13.1 [Reference stage 4 — `RemoveOutsideElement`](#131-reference-stage-4--removeoutsideelement)
    - 13.2 [Reference stage 5 — `ProjectToIsoSurface`](#132-reference-stage-5--projecttoisosurface)
    - 13.3 [Reproducing the diagnosis](#133-reproducing-the-diagnosis)

---

## 1. What this utility does

`OctreeHybridMeshUtility` takes a closed, orientable triangular surface `ModelPart`
and produces an **adaptive all-hexahedral dual mesh** of the octree built around that
surface, written to a VTK file.

The mesh is **adaptive**: cells are smallest near the surface and grow larger away
from it.  All-hex means every element is a hexahedron (no pyramids, prisms, or
tetrahedra anywhere).

At refinement depth 8 on a typical CAD surface:

- ~1.5 million hex elements
- ~1.75 million nodes
- Generated in ~25 seconds (single-threaded, ARM64)

> ⚠️ **Which output do you want — block or carved object?**
> Two entry points are available:
> - **`BuildAndWriteVtk`** writes the dual-hex extraction directly — the reference's
>   intermediate `DualFullHexMeshExtraction` stage.  Its output **fills the entire
>   octree bounding box** (a solid block, fine near the surface and coarse in the
>   interior), *not* the object interior.  Opened in Paraview this looks like a box,
>   and the refinement-interface T-junctions/overlaps inherent to this stage render
>   as internal faces that look like "holes inside".
> - **`BuildCarveAndWriteVtk`** additionally runs
>   [`RemoveOutsideElement`](#13-the-full-reference-pipeline-and-what-is-not-ported)
>   (reference stage 4, inside/outside part), carving the object out of the block so
>   the output is a hex mesh of the **object interior**.  On the depth-8 Stanford
>   bunny this keeps **581,784 of 1,567,546 hexes (37 %)**, and the survivor's
>   bounding box matches the bunny surface — i.e. the bunny is carved out of the box.
>   The carved boundary is **blocky** (it follows the octree grid, not the surface).
> - **`BuildCarveProjectAndWriteVtk`** additionally runs the buffer-zone clearance
>   (paper §2.3) and **`ProjectToIsoSurface`** (reference stage 5 / paper §2.4): it
>   meshes the buffer zone out to the input triangles and runs a Jacobian-controlled
>   optimiser, so the boundary **conforms to the object surface** instead of being a
>   blocky carve.  This removes the exposed interface holes/overlaps of the raw dual
>   block.  On the depth-5 bunny the result fits the surface with **0 inverted core
>   hexes** and only a handful of residual buffer slivers (≈0.05 %).
>
> The projection reproduces the reference's scaled-Jacobian **distribution** closely:
> median ≈ 0.85 / 0.89 at depths 4 / 5 (reference ≈ 0.85 / 0.89), with the
> 10th-percentile quality matching or beating the reference and **0 inverted
> elements**.  The worst-element *floor* is lifted by a **gradual threshold
> escalation** (see [§13.2](#132-reference-stage-5--projecttoisosurface)): the minimum
> scaled Jacobian climbs from the `eps_sj = 0.01` untangling gate to ≈ 0.3 at the
> default budget and ≈ 0.46 with a larger one (reference ≈ 0.5–0.57), and keeps
> climbing with iterations — `proj_iters` is the convergence budget, not a fixed cap.

---

## 2. Background: primal vs dual mesh

The algorithm constructs two meshes:

**Primal mesh**: the octree leaves themselves are hexahedra.  They are axis-aligned
and nested; adjacent cells of different sizes share only part of their faces
(non-conforming).  The primal mesh is not output by default.

**Dual mesh**: one *dual node* is placed at the centroid of every primal cell.
Dual *elements* are formed by connecting the dual nodes of mutually adjacent primal
cells.  In a uniform octree every interior primal vertex is shared by exactly 8
cells, forming one regular hex.  At refinement level boundaries the arrangement
becomes more complex and requires special transition templates.

The fundamental insight of the HybridOctree_Hex algorithm is that the dual of a
**strongly 2:1 balanced** octree can always be tiled with hexahedra, using a
finite library of hand-crafted templates to handle every possible transition
configuration.

---

## 3. Reference and credits

> Tong, H., Halilaj, E., & Zhang, Y. J. (2024).
> **HybridOctree_Hex: Hybrid octree-based adaptive all-hexahedral mesh generation
> with Jacobian control.**
> *Journal of Computational Science*, 78, 102278.
> DOI: [10.1016/j.jocs.2024.102278](https://doi.org/10.1016/j.jocs.2024.102278)

The reference implementation is in
`external_libraries/HybridOctree_Hex/HybridOctree_Hex/` (read-only).
The Kratos port lives in `kratos/modeler/utilities/octree_hybrid_mesh_utility.h` (with the implementation in `octree_hybrid_mesh_utility.cpp`).

---

## 4. Full algorithm pipeline

```
Input: closed triangular surface ModelPart
          │
          ▼
┌──────────────────────────────────────┐
│ Stage 0 — Octree construction        │
│   BuildFromSurfaceMesh               │
│   • OctreeHybrid, initially 1 root   │
│   • For every triangle, find cells   │
│     that contain any corner; refine  │
│     cells near the surface up to     │
│     RefinementDepth                  │
│   • StrongConstrain2To1 enforces     │
│     2:1 balance across all leaves    │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│ Stage 1 — Primal mesh                │
│   WriteDualHexVtk                    │
│   • Enumerate leaf cells → leaves[]  │
│   • For every cell corner (8 per     │
│     cell) compute integer grid key   │
│     at MAX_DEPTH resolution          │
│   • Shared corners get the same id;  │
│     build vert_adj[v] = list of      │
│     (cell_index, corner_index) pairs │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│ Stage 2 — Face-adjacency graph       │
│   adj[i][j] for each leaf i, face j  │
│   • Probe 4 sub-quadrant centres     │
│     just past each face using        │
│     pGetCellNormalized               │
│   • all same cell  → count=1 (equal  │
│     size neighbour)                  │
│   • 4 distinct cells → count=4 (4   │
│     smaller neighbours → template)   │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│ Stage 3 — Transition templates       │
│   For every (i,j) where count==4:    │
│   • Compute p[0..31] reference pts   │
│   • Parity gate: (gx%2==0)&&(gy%2==0│  ← only (even,even) cell anchors
│   • Emit 13-base + up to 8 sub-tmpl  │
│   • consume_at marks the primal      │
│     vertex at the transition centre  │
│     so plain-dual pass skips it      │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│ Stage 4 — Plain dual hexes           │
│   For every primal vertex v:         │
│   • Skip if consumed[] or valence≠8  │
│   • Connect 8 adjacent cell centres  │
│     via idTransform ordering         │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────┐
│ Stage 5 — Node merge + VTK write     │
│   • Single nodes[] array seeded      │
│     with N cell centres              │
│   • Template points looked up via    │
│     27-neighbour spatial hash        │
│     (bucket = 1e-2*min_cell)         │
│   • Write POINTS + CELLS + SCALARS   │
│     (level = -1 for template cells)  │
└──────────────────────────────────────┘

Output: all-hex VTK file covering the full octree bounding box
        (the reference's intermediate DualFullHex mesh — NOT yet
         carved to the object interior; see §13)
```

> **Note on staging terminology.** Stages 0–5 above are the *internal* steps of this
> port and all together correspond to **one** stage of the reference algorithm —
> `DualFullHexMeshExtraction` (reference stage 3 of 5).  The reference's two
> remaining stages, `RemoveOutsideElement` (stage 4) and `ProjectToIsoSurface`
> (stage 5), are documented in [§13](#13-the-full-reference-pipeline-and-what-is-not-ported)
> and are not implemented here.

---

### 4.1 Stage 0 — Octree construction

```cpp
static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth);
```

1. Allocate an `OctreeHybrid` with a single root cell covering `[0,1]^3` in
   normalised coordinates.
2. Iterate over every `Triangle3D3` geometry in `rSurfaceMesh.Geometries()`.
3. For each triangle, test its three corner nodes via `pGetCellNormalized`.  If the
   cell containing a corner has `level < RefinementDepth`, subdivide it.
4. Call `StrongConstrain2To1` on the result — this recursively subdivides cells
   until no two adjacent leaves differ by more than 1 level.

The surface mesh must be triangulated (other element types are skipped silently).
The octree coordinates are normalised: node coordinates from `GetInitialPosition()`
are divided by the bounding box size (plus 1% padding on each side).

---

### 4.2 Stage 1 — Primal mesh with shared vertices

All `N` leaf cells are enumerated into `leaves[]` in the order returned by
`GetAllLeafCells`.  Each cell has 8 corners; their integer grid keys at depth
`MAX_DEPTH = 10` are computed as:

```
stride = 2^(MAX_DEPTH - level)
ix = (gx + CX[c]) * stride       where CX = {0,1,1,0,0,1,1,0}
iy = (gy + CY[c]) * stride       where CY = {0,0,1,1,0,0,1,1}
iz = (gz + CZ[c]) * stride       where CZ = {0,0,0,0,1,1,1,1}
key = iz * (R+1)^2 + iy * (R+1) + ix,   R = 2^MAX_DEPTH
```

This key uniquely identifies a grid position across all levels: two cells at
different levels sharing a physical corner will compute the same integer key.
A `vid_map[key] = vertex_id` hash-map deduplicates on-the-fly so each unique
corner gets exactly one vertex id, and `vert_adj[v]` accumulates the list of
`(cell_index, corner_index)` pairs for vertex `v`.

---

### 4.3 Stage 2 — Face-adjacency graph

For each leaf `i` and face `j` (0..5), the code probes four points just past
the face boundary using `pGetCellNormalized`.  The face geometry:

| j | Fixed axis | Low/High | Free axes |
|---|-----------|----------|-----------|
| 0 | z (axis 2) | low  | x (0), y (1) |
| 1 | y (axis 1) | low  | x (0), z (2) |
| 2 | x (axis 0) | low  | y (1), z (2) |
| 3 | x (axis 0) | high | y (1), z (2) |
| 4 | y (axis 1) | high | x (0), z (2) |
| 5 | z (axis 2) | high | x (0), y (1) |

The four probe points are placed at the four sub-quadrant centres of the face, at
offsets `{0.25, 0.75}` along each free axis within the cell's normalised extent,
and `EPS = 1e-9` past the fixed coordinate.

**Valence determination:**
- If all four probes land in the same leaf `first`:
  - `adj[i][j].count = 1`, `ids[0] = first`
  - The opposite face `adj[first][5-j]` is also marked (count=1, ids[0]=i),
    **but only if they are the same level** — if `first` is coarser, its
    transition face must not be pre-emptively claimed.
- If the four probes land in 4 distinct smaller leaves:
  - `adj[i][j].count = 4`, `ids[0..3] = found[0..3]`
  - Each smaller leaf's opposite face is marked as having `i` as its single
    same-size (coarser) neighbour.
  - The probe ordering is `Q[qi] × Q[qj]` with `qi = q&1`, `qj = (q>>1)&1`,
    giving the canonical order `(LL, HL, LH, HH)` in the `(I, J)` plane.

---

### 4.4 Stage 3 — Plain dual hexes (regular regions)

```
for every primal vertex v:
    if consumed[v]: continue         // claimed by a template
    if vert_adj[v].size() != 8: continue   // boundary or transition vertex
    hex[idTransform[c]] = cell_index_i
    emit dual hex
```

The `idTransform[8] = {6,7,4,5,2,3,0,1}` mapping converts the local corner index
`c` (which octree corner of the cell touches vertex `v`) to the VTK hexahedron
node position, ensuring the right-hand rule is satisfied for a positive Jacobian.

---

### 4.5 Stage 4 — Transition templates (mixed-level regions)

This is the core of the algorithm.  See [Section 5](#5-transition-templates--detailed-geometry) for full template geometry.

For each leaf `i` and face `j` where `adj[i][j].count == 4`:

**1. Parity gate:**

The 13-element base template covers a 2×2 footprint on the fine side.  The four
fine cells form a 2×2 grid in the `(I, J)` in-plane directions.  Only one of the
four possible anchors triggers the template — the one whose in-plane grid indices
are both even:

```cpp
const bool sI = (leaves[i]->GetGridX() % 2 == 0);   // or GetGridY/GetGridZ
const bool sJ = (leaves[i]->GetGridY() % 2 == 0);   // depending on face j
if (!(sI && sJ)) continue;
```

This is equivalent to the reference's `while (posI > 0) { posI -= delta; stepI = !stepI; }` loop, which counts how many large-cell-sized steps it takes to reduce the centre coordinate to zero — an even count means `stepI == true`.

**2. Reference point construction (p[0..31]):**

```
p[0] = centre(sLL)       p[1] = centre(sHL)        // LL = (Ilo,Jlo), HL = (+I)
p[2] = 2*p[1] - p[0]    p[3] = 2*p[2] - p[1]      // extrapolated along I
p[4] = centre(sLH)       p[5] = centre(sHH)        // LH = (+J), HH = (+I+J)
p[6] = 2*p[5] - p[4]    p[7] = 2*p[6] - p[5]
p[8] = 2*p[4] - p[0]    p[9] = 2*p[5] - p[1]      // one more step in J
p[10]= 2*p[5] - p[0]    p[11]= 2*p[10]- p[9]
p[12]= 2*p[8] - p[4]    p[13]= 2*p[9] - p[5]
p[14]= 2*p[13]- p[12]   p[15]= 2*p[14]- p[13]
```

These 16 points lie on the fine-cell grid.  The next 16 are interior/interface
points derived using empirical offsets that maximise the scaled Jacobian:

```
z = ci - 0.25*(p[0]+p[1]+p[4]+p[5])    // out-of-plane vector toward coarse centre

p[16] = p[1]  + (268/375)*z + 0.072*(p[4]-p[0])
p[17] = p[2]  + (268/375)*z + 0.072*(p[4]-p[0])
p[18] = ci                              // coarse cell centre (large cell dual node)
p[19] = p[18] + 2*(p[1]-p[0])
p[20] = p[4]  + (268/375)*z + 0.072*(p[1]-p[0])
p[21] = p[5]  + z/5 - 0.112*(p[1]-p[0]) + 0.056*(p[4]-p[0])
p[22] = p[6]  + z/5 + 0.112*(p[1]-p[0]) + 0.056*(p[4]-p[0])
p[23] = p[7]  + (268/375)*z - 0.072*(p[1]-p[0])
p[24] = p[8]  + (268/375)*z + 0.072*(p[1]-p[0])
p[25] = p[9]  + z/5 - 0.112*(p[1]-p[0]) - 0.056*(p[4]-p[0])
p[26] = p[10] + z/5 + 0.112*(p[1]-p[0]) - 0.056*(p[4]-p[0])
p[27] = p[11] + (268/375)*z - 0.072*(p[1]-p[0])
p[28] = p[18] + 2*(p[4]-p[0])
p[29] = p[18] + 2*(p[5]-p[0])
p[30] = p[13] + (268/375)*z - 0.072*(p[4]-p[0])
p[31] = p[14] + (268/375)*z - 0.072*(p[4]-p[0])
```

The constants `268/375 ≈ 0.715`, `0.072`, `0.112`, `0.056`, `1/5` were derived by
the paper authors as the values that keep all scaled Jacobians positive across all
configurations.

**3. Template variant selection:**

```cpp
const int tmpl = (j==1 || j==3 || j==5) ? 1 : 0;
```
This selects which of the two orientation sub-tables of each `tXId` array applies
(faces on the "high" side of each axis use the reversed orientation).

---

### 4.6 Stage 5 — Node merging and VTK output

A single `nodes[]` array is seeded with the `N` cell centres (the dual nodes).
Template point coordinates are looked up via a 27-neighbourhood spatial hash:

- Hash bucket: `1e-2 * min_cell` where `min_cell` is the world-space size of the
  finest octree cell.
- Merge tolerance: `1e-4 * min_cell`.
- For each template corner, if a nearby node already exists in the hash, reuse its
  id; otherwise append a new node.

This ensures template hexes and plain-dual hexes share the same node ids wherever
they meet geometrically.

The VTK file uses `VTK_HEXAHEDRON` (type 12) cells.  A `SCALARS level int 1`
cell attribute is written: `level = -1` for template-generated hexes, the leaf
cell level for plain-dual hexes.

---

## 5. Transition templates — detailed geometry

### 5.1 Face numbering and orientation conventions

Face numbering in the Kratos port:

| j | Description | Fixed axis | Direction |
|---|-------------|------------|-----------|
| 0 | z-low  (bottom) | z | − |
| 1 | y-low  (front)  | y | − |
| 2 | x-low  (left)   | x | − |
| 3 | x-high (right)  | x | + |
| 4 | y-high (back)   | y | + |
| 5 | z-high (top)    | z | + |

Opposite of face `j` is `5 - j`.

The "side-faces" adjacent to face `j` are given by:
```
pSId[j][0], pSId[j][1]   // the two faces sharing a long edge with face j
pS2Id[j][0], pS2Id[j][1] // the other two faces (sharing a short edge)
```

Full table (from `StaticVars.h`):

| j | pSId[j][0] | pSId[j][1] | pS2Id[j][0] | pS2Id[j][1] |
|---|-----------|-----------|------------|------------|
| 0 | 2 (x-lo)  | 1 (y-lo)  | 3 (x-hi)   | 4 (y-hi)   |
| 1 | 2 (x-lo)  | 0 (z-lo)  | 3 (x-hi)   | 5 (z-hi)   |
| 2 | 1 (y-lo)  | 0 (z-lo)  | 4 (y-hi)   | 5 (z-hi)   |
| 3 | 1 (y-lo)  | 0 (z-lo)  | 4 (y-hi)   | 5 (z-hi)   |
| 4 | 2 (x-lo)  | 0 (z-lo)  | 3 (x-hi)   | 5 (z-hi)   |
| 5 | 2 (x-lo)  | 1 (y-lo)  | 3 (x-hi)   | 4 (y-hi)   |

---

### 5.2 13-element base template (`t1Id`)

Triggered when `adj[i][j].count == 4` and the parity gate passes.

**Layout of the 32 reference points:**

```
Fine side (z points DOWN toward coarse cell)

Row 3 (Jhi outer):  p[12]    p[13]/p[30]    p[14]/p[31]    p[15]
                         p[28]                         p[29]
Row 2 (Jhi):        p[8]/p[24]  p[9]/p[25]   p[10]/p[26]  p[11]/p[27]

Row 1 (Jlo):        p[4]/p[20]  p[5]/p[21]   p[6]/p[22]   p[7]/p[23]
                         p[18]                         p[19]
Row 0 (Jlo outer):  p[0]     p[1]/p[16]     p[2]/p[17]     p[3]

(Ilo → Ihi across)
```

Points p[16..31] are the off-grid interface/interior points computed from the
empirical offset formulae shown in Section 4.5.  p[18] is the coarse cell centroid
(the "large dual node").

**Index table** `t1Id[orientation][13][8]`:

```
Orientation 0 (faces 0, 1, 2 — "low"):
  hex  0: {0,  1,  5,  4, 18, 16, 21, 20}
  hex  1: {1,  2,  6,  5, 16, 17, 22, 21}
  hex  2: {2,  3,  7,  6, 17, 19, 23, 22}
  hex  3: {4,  5,  9,  8, 20, 21, 25, 24}
  hex  4: {5,  6, 10,  9, 21, 22, 26, 25}
  hex  5: {6,  7, 11, 10, 22, 23, 27, 26}
  hex  6: {8,  9, 13, 12, 24, 25, 30, 28}
  hex  7: {9, 10, 14, 13, 25, 26, 31, 30}
  hex  8: {10,11, 15, 14, 26, 27, 29, 31}
  hex  9: {20,21, 25, 24, 18, 16, 30, 28}   ← connects to coarse centre
  hex 10: {22,23, 27, 26, 17, 19, 29, 31}
  hex 11: {21,22, 26, 25, 16, 17, 31, 30}
  hex 12: {16,17, 31, 30, 18, 19, 29, 28}   ← the "top cap"
```

The layout tiles a 4×4 patch of the fine grid (hexes 0–8) and links it to the
coarse dual node via four interface hexes (hexes 9–12).

**Erase point (collectNum removal):**

After emitting the 13 hexes, the mixed-level primal vertex at the transition
centre is consumed.  Its position is:

```cpp
ptmp[d] = 0.5 * (p[21][d] + p[26][d] + z[d] * 4.0/15.0)
```

This is the geometric centroid of the "star" formed by the 8 dual nodes meeting
at the transition vertex.

---

### 5.3 4-element edge template (`t2Id` / `t22Id`)

Fills the gap along one of the two long edges of the 13-element block where an
adjacent octree face also carries a transition.

**Trigger condition (side A, `pSId[j][0]` face):**

```
adj[i][pSId[j][0]].count == 1   AND
adj[nb(i, pSId[j][0], 0)][j].count == 4
```
i.e. the current cell has a single neighbour on its side face, and *that* neighbour
has 4 smaller neighbours on face `j` (meaning there is a 13-element template on
the other side of the long edge).

**Points** (p16[0..15]): 8 new points derived from the base template's `p[]`
by extrapolation away from the interface, plus 8 shared points directly from `p[]`.

New points:
```
p16[0] = 2*p[0] - p[1]                               // far corner LL
p16[1] = 2*p[18] - p[19]                             // far coarse centre
p16[2] = 2*p[4] - p[5]                               // far corner LH
p16[3] = p[20] + 1.144*(p[0] - p[1])                 // interface point LL side
p16[4] = 2*p[8] - p[9]
p16[5] = p[24] + 1.144*(p[0] - p[1])
p16[6] = 2*p[28] - p[29]
p16[7] = 2*p[12] - p[13]                             // far corner
```

Shared old points:
```
p16[8..15] = { p[0], p[4], p[8], p[12], p[18], p[20], p[24], p[28] }
```

**Index table** `t2Id[orientation][4][8]` connects p16 corners into 4 hexes.
`t22Id` covers the symmetric case on the opposite long edge (`pSId[j][1]`).

**Emission rule**: always emit (same as 13-element).
**Erase point**: `0.5*(p16[2]+p[8]) + z/3`

---

### 5.4 3-element edge template (`t3Id` / `t32Id` / `t33Id` / `t34Id`)

Fills the gap on a "long edge" between the 13-element block and the exterior of
the transition region, where neither neighbour carries its own transition.

**Trigger condition (t3Id, `pS2Id[j][0]` face):**

```
adj[i][pS2Id[j][0]].count == 1
AND adj[nb(i, pS2Id[j][0], 0)][j].count == 1          // no other template on that face
AND adj[nb(i, pS2Id[j][0], 0)][pSId[j][0]].count == 4 // but has 4-small on pSId
```

16 new points in p16, 3 hexes per instance.  The four variants (`t3Id`, `t32Id`,
`t33Id`, `t34Id`) cover the four orientations of the long-edge gap (combinations
of the two sides of both long edges).

**Emission rule**: emit only if the transition vertex at `ptmp` is **still available**
(`consume_at` returns `true`).  If a neighbour's 13-element block already claimed
that vertex, skip this template.

---

### 5.5 5-element corner template (`t4Id` / `t42Id` / `t43Id` / `t44Id`)

Fills the gap at a corner where two orthogonal 13-element blocks meet.

**Trigger condition (t4Id, `pS2Id[j][0]` face):**

```
adj[i][pS2Id[j][0]].count == 1
AND adj[nb(i, pS2Id[j][0], 0)][j].count == 4          // neighbour has transition in j
AND adj[nb(i, pS2Id[j][0], 0)][pSId[j][0]].count == 4 // ...and also in pSId direction
```

16 new points, 5 hexes per instance.  The four variants cover the four possible
corner positions.

**Emission rule**: emit only if the template introduces at least one **new node**
not yet in `nodes[]` (checked by comparing `nodes.size()` before and after).  If
all 5×8 corner positions already exist in the node array, this template would be
a pure duplicate and is skipped.

An additional geometric override: if the adjacent transition template is "far"
(the `far_trans` flag), p16[2] is adjusted:

```cpp
if (far_trans) {
    p16[2][d] += p[13][d] - p[30][d] + z[d]*2.0/3.0;
    p16[4][d] += p[13][d] - p[30][d] + z[d]*2.0/3.0;
    ptmp[d] = 0.5*(p16[2][d] + p[31][d]);
}
```

---

### 5.6 Template emission rules (collectNum / `consume_at`)

The reference code maintains `collectNum`: the set of mixed-level primal vertices
that are candidates to be replaced by template hexes.  In the Kratos port this is
`available[]` / `consumed[]`:

- `available[v]`: initially `true` for every primal vertex of valence 8 that
  touches cells of different sizes (the "irregular" or "mixed-level" vertices).
- `consume_at(ptmp)`: round `ptmp` to the integer grid, look up the vertex id in
  `vid_map`, if `available[v]` is `true` set it to `false` (consume) and return
  `true`; otherwise return `false`.

| Template | Emission condition |
|----------|-------------------|
| 13-element base | Always emit; always consume `ptmp`. |
| 4-element (t2/t22) | Always emit; always consume `ptmp`. |
| 3-element (t3/t32/t33/t34) | Emit only if `consume_at(ptmp)` returns `true`. |
| 5-element (t4/t42/t43/t44) | Emit only if `nodes.size()` grew (new node created). |

The plain-dual pass (Stage 4) then skips any vertex `v` where `consumed[v] == true`.

---

## 6. Implementation — Kratos port

### 6.1 Key differences from the reference code

| Aspect | Reference (`HexGen.cpp`) | Kratos port |
|--------|--------------------------|-------------|
| Octree structure | Complete `octreeArray` bit array + `cutArray` leaf list | `OctreeHybrid` class |
| Face adjacency | O(n²) loop over all leaf pairs (`InitiateElementValence`) | O(N·depth) `pGetCellNormalized` probes |
| Vertex dedup | O(n²) linear scan (`DIST_THRES = 1e-12`) | 27-neighbour spatial hash (`1e-4*min_cell` tolerance) |
| Node array | `hexMesh.v[]` flat C array | `std::vector<std::array<double,3>> nodes` |
| Dual-hex emission | Separate loop over `collectNum` | Integrated into template loop; `consumed[]` flag |
| Refinement criterion | Curvature + thickness adaptive (features map to absolute levels 4–8) | **Ported** (`adaptive=True`, default): same curvature + thickness criterion, `CELL_DETECT` halo and complete-octet refinement, so the dual block matches the reference **cell-for-cell** on the bunny (see §6.5).  A `adaptive=False` uniform path is kept for the synthetic template tests |
| Interior filtering | `RemoveOutsideElement` (ray-cast + manifold repair) | Carve ported (`BuildCarveAndWriteVtk`); non-manifold 146-probe repair replaced by the buffer-zone clearance (paper §2.3) in the project path |
| Surface projection | `ProjectToIsoSurface` (analytic gradient descent) | Ported (`BuildCarveProjectAndWriteVtk`): buffer-zone meshing + finite-difference Jacobian control + gated smoothing + gradual threshold escalation; reproduces the reference's quality distribution and lifts the worst element off the untangling gate (§13.2) |
| Coordinate system | Normalised to `[0,100]^3` | Octree in `[0,1]^3`; the projector renormalises to `[0,100]` internally so the reference's optimiser constants apply |
| Output format | World coordinates via `BOX_LENGTH_RATIO` | World coordinates via `ScaleBackToOriginalCoordinate` |

### 6.2 Node-merge spatial hash

The reference's O(n²) vertex deduplication is replaced by a 27-neighbourhood hash:

```cpp
// bucket = floor(coord / bucket_size) * bucket_size
auto bucket_key = [&](const std::array<double,3>& pt) -> std::size_t {
    std::size_t h = 14695981039346656037ULL;
    for (int d = 0; d < 3; ++d) {
        auto b = static_cast<long long>(std::floor(pt[d] / bucket));
        h = (h ^ static_cast<std::size_t>(b)) * 1099511628211ULL;
    }
    return h;
};
```

For a query point `pt`, all 27 neighbouring bucket cells are scanned.  If any
existing node is within `merge_eps = 1e-4 * min_cell`, its id is returned.
Otherwise a new node is appended.

`min_cell` is the world-space size of a cell at `MAX_DEPTH`:
`ScaleBackToOriginalCoordinate({1.0/R}) - ScaleBackToOriginalCoordinate({0.0})`.

### 6.3 Face-adjacency via `pGetCellNormalized`

The Kratos port builds the face-adjacency graph using `pGetCellNormalized` probes
rather than an O(n²) loop.  For each cell `i` and face `j`, four probe points are
placed just past the face, and `pGetCellNormalized` answers which leaf cell each
probe lands in.

**Critical correctness detail:** when marking the "opposite face" shortcut, the
code must only propagate to same-level neighbours.  If a fine cell sees a coarse
neighbour, the coarse cell's transition face must be discovered independently —
pre-stamping it as `count=1` would block the detection of its four fine neighbours
and suppress entire transition regions.

### 6.4 Bugs found and fixed during porting

Two correctness bugs were found and fixed by instrumenting the Kratos port against
the reference code's `DualFullHexMeshExtraction` output.  The instrumented diff
(`/tmp/diff.py`) confirmed the output is now **identical cell-for-cell** at
refinement depths 3 through 7.

**Bug 1 — Same-level guard for the opposite-face shortcut**

*Symptom*: Only half the transition templates fired.  At depth 3, my code detected
8 valence-4 faces and fired 1 template; the reference detected 12 and fired 2.  The
entire `j=5` (z-high) transition was invisible.

*Root cause*: In the `all_same` branch of the adjacency loop, when a **fine** cell
detected its single **coarse** neighbour, the code stamped `count=1` onto the coarse
cell's opposite face.  Later, when the coarse cell was processed, it hit the
`if (adj[i][j].count != 0) continue` early-exit and never detected its four fine
neighbours.

*Fix*:
```cpp
// Only propagate the opposite-face entry for a SAME-SIZE neighbour.
// A fine cell seeing a coarse neighbour must NOT mark the coarse cell's
// transition face — that face genuinely has valence 4 and must be
// discovered independently.
if (leaves[i]->GetLevel() == leaves[first]->GetLevel() &&
    adj[first][opp].count == 0) {
    adj[first][opp].count = 1;
    adj[first][opp].ids[0] = i;
}
```

**Bug 2 — Base template `collectNum` erase point**

*Symptom*: At refinement depth ≥ 6, 27 extra (unconsumed) dual hexes appeared at
the transition centres.

*Root cause*: The 13-element base erase point was computed as:
```cpp
ptmp[d] = 0.5*(p[21][d] + p[26][d]) + z[d] * 4.0/15.0;   // WRONG
```
The reference formula is:
```cpp
ptmp[0] = 0.5*(p[21][0] + p[26][0] + z[0]*4/15);           // reference
// which equals 0.5*(p[21]+p[26]) + z*2/15
```
The `z`-term was doubled (`4/15` instead of `2/15`), causing the consume to round
to the wrong grid position once cells were large enough in finest-grid units.

*Fix*:
```cpp
ptmp[d] = 0.5*(p[21][d] + p[26][d] + z[d] * 4.0/15.0);   // correct
```

### 6.5 Adaptive refinement (matches the reference cell-for-cell)

`BuildAdaptiveFromSurfaceMesh` (used by default, `adaptive=True`) ports the
reference's `ConstructOctree` criterion so the dual block is **identical** to the
reference's:

- **Curvature** — per-vertex angle defect (sum of squared dihedral-angle deviations,
  `BuildRefineSets`, the `ReadRawData` loop) on the surface normalised to a 100-unit
  cube; **thickness** — a normal-ray cast to the opposite sheet (`TriRayIntersect`).
  A surface vertex demands level `4+L` where its curvature exceeds `C_THRES[L]`, and
  a thin feature demands it where its thickness is below `H_THRES[L]`
  (`C_THRES = {0,0,0.4,0.8,1.6}`, `H_THRES = {16,8,4,2,1}`).
- **`CELL_DETECT = 1.0` halo** — a cell is tested against a box twice its size, and
- **complete-octet refinement** — when any cell of a sibling octet must subdivide,
  all eight do (the reference's `ComputeCellValue` marks all 8 children when one is
  set).  This last point is what makes the leaf set match exactly; without it the
  port under-refines by ~25 %.

The domain is the reference's centred cube (side = the surface's largest extent), so
the integer leaf grid coincides.  Verified on the low-poly bunny: the dual block
(`DualFullHexMeshExtraction`) is **3128 / 20 949 / 108 673** hexes at depths 4 / 5 / 6
— matching the reference cell-for-cell (`test_adaptive_block_matches_reference`).

> The intermediate **carve** (`RemoveOutsideElement`) stays within ~5–10 % of the
> reference's cell count: its inside/outside test consumes a pseudo-random ray
> sequence, so the surface-band classification is not bit-reproducible across
> implementations.  The stage-5 projection re-meshes that band regardless.

A `adaptive=False` path keeps the original **uniform** refinement (every cell a
triangle intersects is split to `RefinementDepth`); it is used by the synthetic
transition-template tests, whose flat patches carry no curvature and so would not
refine adaptively.

### 6.6 Modifications made to the reference code

To cross-validate against the reference on the bunny, two small changes were made to
`external_libraries/HybridOctree_Hex/` (each carries an inline `Kratos …` comment):

1. **`Initialization.h` — `VOXEL_SIZE` is now overridable** at compile time
   (`#ifndef VOXEL_SIZE … #define VOXEL_SIZE 10`).  `VOXEL_SIZE` is the octree depth
   (`hexGen` ctor: `octreeDepth = depth`); it was hard-coded to 10.  Building with
   `-DVOXEL_SIZE=5` lets the reference run at the same depth as the port.
2. **`HexGen.cpp::InitializeOctree` — `getLevel` fill bound fixed** from
   `i < octreeDepth` to `i <= octreeDepth`.  The original left the finest-level cells
   with `getLevel == 0`; whenever refinement actually reaches the finest level (the
   bunny refines almost everywhere) `StrongBalancedOctree` then computed a bogus grid
   coordinate and indexed `preVecEightCell` out of bounds, **segfaulting**.  This
   stayed latent for the shipped demos (at `VOXEL_SIZE=10` their features map to
   levels ≤ 8 and never reach the finest level).  With the fix the reference runs
   end-to-end on the bunny and produces `dualFullHex` / `dualHex` / `projHex`, which
   were used as ground truth for the carve and projection ports.

---

## 7. API reference

All methods are `static` — no instance is needed.

### `BuildFromSurfaceMesh`

```cpp
static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth,
    bool Adaptive = true);
```

Builds and balances an OctreeHybrid from the triangles in `rSurfaceMesh`.

| Parameter | Description |
|-----------|-------------|
| `rSurfaceMesh` | ModelPart with `Triangle3D3` geometries (from `StlIO::ReadModelPart`). |
| `RefinementDepth` | Max refinement level near the surface. Range: `[1, MAX_DEPTH=10]`. |
| `Adaptive` | `true` (default): the reference curvature/thickness criterion (`BuildAdaptiveFromSurfaceMesh`, §6.5), matching the reference block cell-for-cell.  `false`: uniform refinement of every surface-intersecting cell to `RefinementDepth` (used by the synthetic template tests). |

Returns a `std::unique_ptr<OctreeType>` owning the balanced octree.

---

### `WriteDualHexVtk`

```cpp
static void WriteDualHexVtk(
    OctreeType& rOctree,
    const std::string& rFilename,
    const TriangleSoup* pTriangles = nullptr);
```

Runs Stages 1–5 on an already-built octree and writes the dual all-hex mesh to
`rFilename` in VTK legacy ASCII format.  When `pTriangles` is supplied (a flat list
of surface triangles in world coordinates), the block is **carved** against that
surface before writing (see `RemoveOutsideElement`); otherwise the full
bounding-box block is written.

---

### `BuildAndWriteVtk`

```cpp
static void BuildAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rFilename,
    std::size_t RefinementDepth = 5,
    bool Adaptive = true);
```

Combines `BuildFromSurfaceMesh` and `WriteDualHexVtk` into a single call.  Writes the
**uncarved** dual block covering the whole octree bounding box.  `Adaptive` selects the
reference curvature/thickness criterion (default) or uniform refinement (§6.5).

---

### `BuildCarveAndWriteVtk`

```cpp
static void BuildCarveAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rFilename,
    std::size_t RefinementDepth = 5,
    bool Adaptive = true);
```

Like `BuildAndWriteVtk`, but **carves** the dual block against the input surface
(reference stage 4, `RemoveOutsideElement` — inside/outside part) so the output is a
hex mesh of the object interior rather than its bounding box.  A hex is kept iff at
most two corners are outside and the outside excursion is small relative to the
deepest inside corner (`n_out < 3 && d_min_neg + 0.15·d_max_pos ≥ 0`).  The surface
triangles are read from `rSurfaceMesh.Geometries()`.

In Python:
```python
KM.OctreeHybridMeshUtility.BuildCarveAndWriteVtk(surface_mp, "bunny.vtk", 8)
```

---

### `BuildCarveProjectAndWriteVtk`

```cpp
static void BuildCarveProjectAndWriteVtk(
    ModelPart& rSurfaceMesh,
    const std::string& rFilename,
    std::size_t RefinementDepth = 5,
    int ProjIters = 20000,
    int ProjSmooth = 1000,
    bool Adaptive = true);
```

Like `BuildCarveAndWriteVtk`, but after carving it **fits the mesh to the input
surface with Jacobian control** (reference stage 5, `ProjectToIsoSurface` / paper
§2.4).  The pipeline:

1. **Buffer-zone clearance** (paper §2.3): boundary hexes where the carved surface
   folds — detected as a boundary vertex whose incident face normals do not fit in
   any open hemisphere — are removed so the extruded shell cannot self-intersect.
2. **Buffer-layer meshing**: every boundary vertex is duplicated and connected to its
   duplicate by a new hex (tagged `level = -2`); the duplicate shell will be pulled
   onto the surface.
3. **Jacobian-controlled optimisation**: a gradient method ascends each element's
   quality (scaled Jacobian where valid, raw Jacobian where inverted) and, once the
   shell is valid, pulls each duplicate onto its closest surface point; smart
   Laplacian smoothing (accepted only when it keeps the scaled Jacobian above a
   rising threshold) runs every `ProjSmooth` iterations, for `ProjIters` total.

`ProjIters`/`ProjSmooth` control the run length.  The default reproduces the
reference's scaled-Jacobian **distribution** (median ≈ 0.85 / 0.89 and 0 inverted at
depths 4 / 5).  The *minimum* scaled Jacobian is driven up by a **gradual threshold
escalation** (a small per-window ramp with a best-valid snapshot, [§13.2](#132-reference-stage-5--projecttoisosurface)):
it climbs from the `eps_sj = 0.01` untangling gate toward the paper's > 0.5 floor and
**continues to climb with more `ProjIters`** (≈ 0.3 at the 20 000-iteration default,
≈ 0.46 at 300 000), so `ProjIters` is the convergence budget rather than a fixed cap.

In Python:
```python
KM.OctreeHybridMeshUtility.BuildCarveProjectAndWriteVtk(surface_mp, "bunny.vtk", 5)
```

---

### `WritePrimalVtk`

```cpp
static void WritePrimalVtk(OctreeType& rOctree, const std::string& rFilename);
```

Writes the primal (non-conforming octree cell) mesh in VTK format, one hex per
leaf cell.  Useful for debugging the octree structure.

---

### `WriteOctreeForReference`

```cpp
static void WriteOctreeForReference(
    ModelPart& rSurfaceMesh,
    const std::string& rFilename,
    std::size_t RefinementDepth);
```

Writes the balanced octree in the coordinate system expected by the reference
`HexGen.ReadOctree`: vertex positions are `grid_index * 100.0 / R` where
`R = 2^RefinementDepth`.  Used for regression testing against the reference driver.

---

## 8. Key constants and lookup tables

All lookup tables are `static constexpr` in `WriteDualHexVtk`.  They are transcribed
verbatim from `StaticVars.h` in the reference.

| Table | Dimensions | Purpose |
|-------|-----------|---------|
| `idTransform` | `[8]` | Maps cell corner index → hex node position for the regular dual hex. |
| `t1Id` | `[2][13][8]` | 13-element base template: 2 orientations × 13 hexes × 8 corners. |
| `t2Id` | `[2][4][8]` | 4-element edge template A. |
| `t22Id` | `[2][4][8]` | 4-element edge template B (opposite side). |
| `t3Id` | `[2][3][8]` | 3-element edge template A. |
| `t32Id` | `[2][3][8]` | 3-element edge template B. |
| `t33Id` | `[2][3][8]` | 3-element edge template C. |
| `t34Id` | `[2][3][8]` | 3-element edge template D. |
| `t4Id` | `[2][5][8]` | 5-element corner template A. |
| `t42Id` | `[2][5][8]` | 5-element corner template B. |
| `t43Id` | `[2][5][8]` | 5-element corner template C. |
| `t44Id` | `[2][5][8]` | 5-element corner template D. |
| `pSId` | `[6][2]` | Side-face indices adjacent to each transition face (long edges). |
| `pS2Id` | `[6][2]` | Corner-face indices adjacent to each transition face (short edges). |

```cpp
static constexpr int idTransform[8] = {6, 7, 4, 5, 2, 3, 0, 1};
```

```cpp
static constexpr int pSId[6][2] = {
    {2,1}, {2,0}, {1,0}, {1,0}, {2,0}, {2,1}
};
static constexpr int pS2Id[6][2] = {
    {3,4}, {3,5}, {4,5}, {4,5}, {3,5}, {3,4}
};
```

---

## 9. Verification and testing

### Unit test

`kratos/tests/test_octree_hybrid_dual_mesh.py` builds a small 2:1-transition surface
and runs `BuildAndWriteVtk` at depths 3–7.  For each depth it asserts:

- **0 degenerate elements** (8 distinct node ids per hex).
- **0 inverted elements** (scaled Jacobian > 0 at all 8 corners via `CORNER_TETS`).
- **Exact reference hex count** against the reference `DualFullHexMeshExtraction`
  output: `{3: 76, 4: 404, 5: 2055, 6: 5241, 7: 18450}`.

### Reference diff harness

`/tmp/diff.py DEPTH` runs `WriteOctreeForReference`, `BuildAndWriteVtk`, and the
reference driver `/tmp/refdiff`, then compares the two hex sets by converting all
coordinates to the integer grid frame and computing the symmetric difference.  At
depths 3–7 the diff is always zero: identical hex sets, zero gaps, zero extras.

### Stanford Bunny benchmark

`kratos/tests/demo_octree_hybrid_mesh.py` generates a depth-8 mesh from
`Bunny-LowPoly.stl`:
- 1,567,546 hexes / 1,751,580 points in ~25 seconds
- 0 degenerate, 0 inverted elements

---

## 10. Usage examples

### Python

```python
import KratosMultiphysics as KM

# --- 1. Read the surface mesh ---
model      = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

stl_io = KM.StlIO("my_surface.stl", KM.Parameters('{"open_mode": "read"}'))
stl_io.ReadModelPart(surface_mp)

# --- 2a. Generate the dual hex mesh (full bounding-box block) ---
KM.OctreeHybridMeshUtility.BuildAndWriteVtk(surface_mp, "hex_block.vtk", 8)

# --- 2b. ...or carve it down to the object interior (blocky boundary) ---
KM.OctreeHybridMeshUtility.BuildCarveAndWriteVtk(surface_mp, "hex_object.vtk", 8)

# --- 2c. ...or carve AND fit the boundary to the surface (Jacobian control) ---
KM.OctreeHybridMeshUtility.BuildCarveProjectAndWriteVtk(surface_mp, "hex_fitted.vtk", 5)
```

Open the VTK in Paraview and colour by the `level` scalar field to see the adaptive
refinement.  Template cells have `level = -1`; buffer-layer cells (added by
`BuildCarveProjectAndWriteVtk`) have `level = -2`.  `BuildAndWriteVtk` fills the
whole bounding box; `BuildCarveAndWriteVtk` keeps only the hexes inside/straddling
the surface (a blocky object); `BuildCarveProjectAndWriteVtk` additionally fits the
boundary to the input surface.

### Python — step-by-step (octree inspection)

```python
# Build octree separately for inspection
octree = KM.OctreeHybridMeshUtility.BuildFromSurfaceMesh(surface_mp, 6)

# Write just the primal mesh (non-conforming, one hex per leaf)
KM.OctreeHybridMeshUtility.WritePrimalVtk(octree, "primal.vtk")

# Write the dual hex mesh from the same octree
KM.OctreeHybridMeshUtility.WriteDualHexVtk(octree, "dual.vtk")
```

### Python — demo script

```bash
cd kratos/tests
python3 demo_octree_hybrid_mesh.py Bunny-LowPoly.stl 8            # full block
python3 demo_octree_hybrid_mesh.py Bunny-LowPoly.stl 8 --carve    # carved object
python3 demo_octree_hybrid_mesh.py Bunny-LowPoly.stl 5 --project  # carved + surface-fitted
```

This reads the STL (converting from binary if necessary), builds the octree, and
writes `octree_hex_mesh.vtk` in the current directory.  With `--carve` it runs
`BuildCarveAndWriteVtk` (depth-8 bunny: 1,567,546 → 581,784 hexes).

### C++

```cpp
#include "modeler/utilities/octree_hybrid_mesh_utility.h"

// surface_mp already populated with Triangle3D3 elements
KM::OctreeHybridMeshUtility::BuildAndWriteVtk(surface_mp, "block.vtk", 8);       // block
KM::OctreeHybridMeshUtility::BuildCarveAndWriteVtk(surface_mp, "object.vtk", 8); // carved
```

---

## 11. Performance notes

| Stage | Complexity | Notes |
|-------|-----------|-------|
| Octree construction | O(N·depth) | `pGetCellNormalized` is O(depth) per call. |
| Primal vertex map | O(N) | Hash-map insert per corner. |
| Face-adjacency | O(N·depth) | 4 probes × 6 faces per leaf. |
| Template loop | O(N) | Constant work per (leaf, face) pair. |
| Node merge | O(N) amortised | 27-neighbour hash, nearly O(1) per node. |
| VTK write | O(N) | Sequential file write. |

Total for a depth-8 bunny: ~25 seconds, dominated by octree construction and VTK I/O.

The reference `InitiateElementValence` is O(n²); replacing it with the probe-based
approach reduced depth-8 sphere time from 693 s to seconds.

---

## 12. Known limitations

1. **`BuildAndWriteVtk` outputs a solid bounding-box block, not the object.** The
   dual mesh covers the full octree bounding box: a fine shell near the surface plus
   a coarse interior filling the box.  Use **`BuildCarveAndWriteVtk`** to apply the
   inside/outside carve (reference stage 4, `RemoveOutsideElement`) and obtain a mesh
   of the object interior instead (see
   [§13.1](#131-reference-stage-4--removeoutsideelement)).  *Measured on the depth-8
   Stanford bunny: the block is 1,567,546 hexes filling a 110×88×109 box; the carve
   keeps 581,784 hexes whose bounding box matches the bunny surface.*  Use
   **`BuildCarveProjectAndWriteVtk`** to additionally fit the boundary to the surface
   (stage 5, [§13.2](#132-reference-stage-5--projecttoisosurface)).  Still **not
   ported**: the reference's exact non-manifold 146-probe repair (a hemisphere-based
   buffer-zone clearance is used instead) and the full minimum-scaled-Jacobian
   convergence of stage 5 (the worst-point "drag" loop).

2. **This output is the reference's intermediate `DualFullHex` stage**: the mesh is
   conforming in the node-sharing sense but carries a small number of T-junctions
   and template overlaps at the refinement interface (e.g. depth 4: 2 overlapping
   faces, 827 open boundary edges; the depth-8 bunny block sums to ~1.26× the
   bounding-box volume, i.e. it over-fills slightly because of those template
   overlaps).  The reference's own output has the *identical* counts — they are
   resolved only by the downstream `RemoveOutsideElement` and `ProjectToIsoSurface`
   stages.  This — not a missing-hex defect — is what produces the "holes inside"
   appearance when the uncarved block is viewed in Paraview: the exposed internal
   transition faces render as internal surfaces.

3. **Single-threaded**: the current implementation is single-threaded.  The main
   parallel opportunities are the octree subdivision loop and the template geometry
   computation.

4. **Carve cell count (~5–10 % below reference)**: the adaptive refinement and dual
   block now match the reference cell-for-cell (§6.5), but the intermediate carve
   (`RemoveOutsideElement`) keeps slightly fewer cells.  Its inside/outside test
   consumes a pseudo-random ray sequence in an implementation-specific order, so the
   surface-band classification is not bit-reproducible; the stage-5 projection
   re-meshes that band regardless.

5. **Stage-5 worst-element floor**: `BuildCarveProjectAndWriteVtk` reproduces the
   reference's scaled-Jacobian *distribution* (median ≈ 0.85 / 0.89 and 0 inverted at
   depths 4 / 5, with p10 matching or beating the reference).  A gradual threshold
   escalation lifts the *minimum* scaled Jacobian off the `eps_sj = 0.01` untangling
   gate toward the paper's > 0.5 floor (≈ 0.3 at the default budget, ≈ 0.46 at
   300 000 iterations and still climbing), but reaching the reference's ≈ 0.57 on the
   single worst sliver can need a large budget — and the carve set differs slightly
   from the reference (limitation 4 above), so the worst cell is not the same element
   (§13.2).

6. **`MAX_DEPTH = 10`**: dictated by `OctreeHybridKratosConfiguration`.  Finer
   meshes require increasing this constant.

---

## 13. The full reference pipeline and what is *not* ported

The reference `HexGen` driver (`Main.cpp`) runs **five** stages.  The Kratos port
now covers stages 0–5; the only pieces not reproduced exactly are the reference's
non-manifold 146-probe repair (replaced by a hemisphere-based buffer-zone clearance)
and the worst-point "drag" convergence loop of stage 5:

| # | Reference method | Output file | Ported? | Role |
|---|------------------|-------------|---------|------|
| 0 | `InitializeOctree` | `modifiedTri.vtk` | ✅ (`BuildFromSurfaceMesh`) | read surface, set bounding box |
| 1 | `ConstructOctree` | `octree.vtk` | ✅ curvature + thickness adaptive (`adaptive=True`); block matches cell-for-cell — §6.5 | refine + 2:1 balance |
| 2 | `InitiateElementValence` | — | ✅ (face-adjacency graph) | classify face valences |
| 3 | `DualFullHexMeshExtraction` | `dualFullHex.vtk` | ✅ (`BuildAndWriteVtk`) | dual hex block over the whole bbox |
| 4 | `RemoveOutsideElement` | `dualHex.vtk` | ◑ carve ported (`BuildCarveAndWriteVtk`); 146-probe repair → hemisphere clearance; ~5–10 % fewer cells (RNG ray order, §6.5) | carve the object out of the block |
| 5 | `ProjectToIsoSurface` | `projHex.vtk` | ◑ ported (`BuildCarveProjectAndWriteVtk`); matches the SJ distribution, gradual threshold escalation lifts the worst element off the gate (§13.2) | project boundary to surface, control Jacobian |

The "holes inside" reported when viewing the depth-8 bunny come from stopping after
stage 3: the **`dualFullHex` block** is solid, over the whole bounding box, and
carries the refinement-interface T-junctions/template overlaps that this stage is
*defined* to carry.  `BuildCarveAndWriteVtk` runs the stage-4 carve to recover the
(blocky) object; **`BuildCarveProjectAndWriteVtk`** runs stage 5 to fit the boundary
to the surface, which removes those exposed interface faces.

> Practical note: the reference is hardcoded for `VOXEL_SIZE = 10` and uses O(n²)
> dual-vertex deduplication and O(n²·#triangles) inside/outside tests, so running its
> own *full* (22 490-triangle) bunny end-to-end is very slow.  After the two reference
> fixes in §6.6 (configurable `VOXEL_SIZE` + the `getLevel` segfault fix), the
> **low-poly** bunny (292 triangles) runs end-to-end at `VOXEL_SIZE=5` in ~20 s,
> producing `dualFullHex.vtk` (20 949 hexes), `dualHex.vtk` (5 711) and `projHex.vtk`
> (8 889).  `projHex` reaches **0 inverted, minimum scaled Jacobian ≈ 0.48, median
> ≈ 0.89** — this is the ground truth the carve and projection ports were validated
> against.

### 13.1 Reference stage 4 — `RemoveOutsideElement`

Carves the object out of the bounding-box block and repairs non-manifold topology.
Source: `HexGen.cpp::RemoveOutsideElement` (≈ line 2562); reference doc
`doc/05_interior_extraction.md`.

> **Port status:** the **carve** (Steps 1–2 below) is ported as
> `OctreeHybridMeshUtility::RemoveOutsideElement`, exposed through
> `BuildCarveAndWriteVtk`.  The geometry kernels `Intersect` (ray/triangle) and
> `PointToTri` (closest-point-on-triangle) are ported verbatim; the per-node signed
> distance is computed in parallel (`IndexPartition`).  The **non-manifold repair**
> (Step 3) is *not* ported.

**Step 1 — signed distance per hex vertex.**
For every vertex `v` of the block mesh:
- *Inside/outside test (ray casting):* shoot a ray `v + t·dir` with a random
  direction (components bounded away from zero to avoid axis-aligned degeneracies)
  and count surface-triangle crossings with `α > 0`; an odd count ⇒ inside.  If any
  triangle is hit edge-on (`Intersect` returns `-1`), regenerate `dir` and restart.
- *Magnitude:* `deletePoint[v] = min over triangles of PointToTri(tri, v)`
  (closest-point-on-triangle distance).
- If `v` is outside, negate: `deletePoint[v] = -deletePoint[v]` ⇒ a **signed
  distance**, negative outside.

**Step 2 — keep/reject each hex.**
For hex `e`, let `k` = number of corners with `deletePoint < 0`, `tmp[0]` = max
positive signed distance, `tmp[1]` = min (most negative) signed distance.  Keep iff:

```
k < 3   AND   tmp[1] + OUT_IN_RATIO * tmp[0] >= 0          (OUT_IN_RATIO = 0.15)
```

i.e. accept hexes with at most two outside corners, and only if the outside
excursion is small relative to how deep the deepest inside corner sits.  This admits
straddling boundary hexes while discarding wholly-outside elements.

> **Verified on the port's output.** Implementing exactly this criterion against the
> depth-8 bunny block (`octree_hex_mesh.vtk`, 1,567,546 hexes) keeps **540,859 hexes
> (34.5 %)**; the kept set's bounding box is `[-23.5, 83.8] × [-41.1, 45.0] ×
> [5.5, 112.3]`, matching the bunny surface box `[-23.9, 84.2] × [-41.4, 45.2] ×
> [5.3, 112.5]`.  The block really is the bunny embedded in its bounding box.

**Step 3 — non-manifold element removal.**
Boundary faces (faces owned by a single hex) are collected.  At each boundary vertex
the outward normals of the incident boundary faces are tested against **146** probe
directions sampled on the unit sphere (`pointOnSurf`); a vertex whose faces block
*every* probe direction is a degenerate (non-manifold) configuration.  Degenerate
hexes are removed fewest-neighbours-first, re-exposing their faces, until none
remain.  This is what finally makes the boundary a clean 2-manifold (removing the
T-junctions/overlaps the `dualFullHex` stage left behind).

**Step 4 — re-index** the survivors (dedup vertices) and store back.

### 13.2 Reference stage 5 — `ProjectToIsoSurface`

The most expensive stage: it pushes the carved mesh's boundary onto the input
triangular surface while keeping every scaled Jacobian above a target threshold.
Source: `HexGen.cpp::ProjectToIsoSurface` (≈ line 2815); reference doc
`doc/06_jacobian_projection.md`.

- **Boundary-vertex split:** each boundary vertex is duplicated into an *inner* copy
  (moved by interior smoothing) and a *surface* copy (projected to the nearest
  triangle); a zero-thickness coupling hex links them.
- **Gradient-descent loop:** accumulates (a) an analytical *Jacobian* gradient for
  every evaluation point whose scaled Jacobian is `≤ 0` or below `ELEM_THRES`
  (9 evaluation points per hex: centre + 8 corners), and (b) a *drag* gradient
  pulling each surface copy toward its closest point on the surface; positions are
  updated by `LEARNING_RATE · g`.
- **Periodic Laplacian smoothing** (every `UPDATE_EVERY = 1000` iterations) with a
  random blend factor to avoid oscillation, re-projecting surface vertices each pass.
- **Progressive tightening:** `ELEM_THRES` starts at `0.01` (get everything positive
  fast), then jumps to `0.53` and climbs `+0.01` per cycle once the mesh is fully
  positive and all surface vertices are within `sqrt(1e-12)` of the surface — driving
  the minimum scaled Jacobian up to the ~0.5+ values quoted in the paper's tables.
- The worst surface vertex (`maxDistIdx`) gets periodic forced projection to escape
  local minima.

Output: the final watertight, Jacobian-controlled all-hex mesh (`projHex.vtk` /
`finalMesh.vtk`).

**What the Kratos port (`BuildCarveProjectAndWriteVtk`) does**, and how it differs:

- **Buffer-zone clearance** (`ClearBufferZone`): the reference's 146-direction
  `pointOnSurf` probe is reproduced as a hemisphere test on ~128 Fibonacci-sphere
  directions.  A boundary vertex whose incident face normals fit in no open
  hemisphere marks a fold; the most-exposed incident hex is removed and the boundary
  re-extracted until clean.  This is the key step that stops the extruded shell from
  self-intersecting — on the depth-5 bunny it cuts inverted buffer hexes from ~11 %
  to ~0.05 %.

  > **Non-destructive overloads.** `ClearBufferZone` and `ProjectToIsoSurface` also have
  > overloads that operate on a `core_cell_indices` index list instead of mutating
  > `rCells` directly: `ClearBufferZone` shrinks the index list in place, and
  > `ProjectToIsoSurface` appends any new buffer-shell hexes to `rCells`/`rCellLevel`
  > and fills a `carve_status` output vector (0 = outside, 1 = core, 2 = buffer shell).
  > These are used by `OctreeHybridMeshGeneratorModeler::ApplyRefinement` so that
  > `project_to_surface: true` never removes cells from the mesh; the destructive
  > overloads documented above remain available for `BuildCarveAndWriteVtk` /
  > `BuildCarveProjectAndWriteVtk` and their tests.

- **Buffer-layer meshing** mirrors the reference: each boundary vertex is duplicated,
  and a hex (`level = -2`) links the boundary quad to its duplicate quad, with the
  winding chosen so a small outward extrusion gives a positive Jacobian.
- **Quality metric** `ScaledJacobianMin` / `JacobianMin` are exact ports of
  `HexGen.cpp::Sj` (min scaled Jacobian over the body centre + 8 corners; raw volume
  variant for inverted elements).
- **Optimisation** follows the same energy (geometry fitting − scaled Jacobian −
  Jacobian) and update rule, but the per-element gradient is computed **numerically**
  (central differences over the incident optimisable corners) rather than with the
  reference's 1 200-line hand-expanded analytic gradient — same energy, far less code.
  The gradient only acts on cells whose scaled Jacobian is `≤ eps_sj` (untangling);
  the bulk quality comes from the gated smart-Laplacian smoothing, which accepts a
  move only when it keeps every incident element above `eps_sj`.  The shell is pulled
  onto the surface by the duplicate smoothing plus a per-window **drag** of the single
  worst-distance duplicate (the reference's `maxDistIdx` drag).
- **Normalisation:** the mesh + triangles are rescaled to a 100-unit box on entry so
  the reference's constants (`LEARNING_RATE = 5e-4`, `eps_sj = 0.01`) apply unchanged,
  then rescaled back on exit.

**Where the quality comes from.** The median scaled Jacobian is driven almost
entirely by the **gated smart-Laplacian smoothing**, not by the gradient (which only
untangles cells whose scaled Jacobian is already below `eps_sj`).  The one detail that
makes the port reach the reference's quality is the **core-boundary smoothing ring**:
each inner boundary vertex smooths toward its edge-adjacent corners across *all*
incident elements **including the buffer hexes** (the reference's `cP2`).  That pulls
the core boundary toward the on-surface duplicates and keeps the buffer prisms
regular — the buffer layer is where the quality budget is spent.  Restricting the ring
to core cells (an earlier port bug) left the buffer hexes at median ≈ 0.69 with a few
inverted; fixing it lifts the buffer to ≈ 0.81 and the whole mesh to the figures
below.

**Worst-element floor and escalation.** The reference escalates `eps_sj`
(`0.01 → 0.53 → +0.01`) once the shell is valid and seated on the surface, which lifts
the *minimum* scaled Jacobian to ≈ 0.5 on small meshes.  A single 0.01→0.53 jump
diverges under the finite-difference gradient (it makes dozens of cells bad at once and
the gated smoothing drags the duplicates off the surface to satisfy `SJ > 0.53` faster
than the gradient recovers).  The port reaches the same regime with a **gradual
escalation** built from three stabilisers:

1. a *gradual ramp* — `eps_sj` rises by `EPS_STEP = 0.03` only on a window with no
   sub-threshold cell, toward `EPS_TARGET = 0.5`; a window that cannot regain validity
   counts toward a stall budget (`STALL_MAX = 8`), after which the gate backs off a
   step, so it hovers at the best reachable threshold instead of jumping past it;
2. an *always-on surface attractor* on the duplicates (full strength when valid,
   attenuated while untangling) so escalation can never strand the shell off the
   geometry;
3. a *best-valid snapshot*, restored on exit, so escalation can only ever raise the
   worst element, never degrade the converged mesh.

The untangling gradient also gets its own larger learning rate (`LRQ = 2e-3` vs the
attractor's `LR = 5e-4`) so it lifts the worst sliver fast enough to keep up with the
ramp.  Together these climb the minimum scaled Jacobian from the `eps_sj = 0.01` gate
to ≈ 0.3 at the default budget and ≈ 0.46 at 300 000 iterations, still climbing.
Reaching the reference's ≈ 0.57 on the single worst cell can need a large budget, and
because the carve set differs slightly (§12 limitation 4) the worst cell is not the
same element as the reference's.

Measured benchmarks (port vs reference `projHex`, median / p10 / minSJ / inverted):
- depth 4, default budget: **0.85 / 0.58 / 0.30 / 0** &nbsp; (reference 0.85 / 0.57 / 0.57 / 0)
- depth 4, 300 000 iters: &nbsp;**0.85 / 0.58 / 0.46 / 0**
- depth 5, default budget: **0.89 / 0.53 / 0.28 / 0** &nbsp; (reference 0.89 / 0.54 / 0.53 / 0)

Median and p10 match the reference (the port's p10 even edges ahead) with 0 inverted;
the worst element climbs monotonically with the iteration budget.

### 13.3 Reproducing the diagnosis

The block → bunny carve used to confirm the above (no reference build required):

1. Generate the port's mesh: `python3 kratos/tests/demo_octree_hybrid_mesh.py
   kratos/tests/Bunny-LowPoly.stl 8` → `octree_hex_mesh.vtk`.
2. For every vertex, compute the signed distance to the surface (ray-cast parity for
   the sign, closest-point-on-triangle for the magnitude).
3. Keep hexes satisfying `k < 3 && tmp[1] + 0.15*tmp[0] >= 0`; re-index and write.

To run the **reference itself**, convert a surface to its `.raw` format (ASCII:
`"#points #triangles"`, then point coords, then triangle indices), place it as
`model.raw`, and build `Main.cpp + HexGen.cpp + Mesh.cpp` with `VOXEL_SIZE = 10`.
Expect long runtimes on detailed surfaces.

---

## See also

- **[OctreeHybridMeshGeneratorModeler](octree_hybrid_mesh_generator_modeler.md)** — the Kratos modeler that wraps this
  utility inside the standard Kratos pipeline; produces an in-memory `ModelPart` with
  elements, boundary conditions, and (in primal mode) hanging-node master-slave constraints.
- [OctreeHybrid spatial container](../../Spatial_Containers/Trees_And_Searches/octree_hybrid.md)
- [OcTree](../../Spatial_Containers/Trees_And_Searches/octree.md)
- Reference paper: [DOI 10.1016/j.jocs.2024.102278](https://doi.org/10.1016/j.jocs.2024.102278)
