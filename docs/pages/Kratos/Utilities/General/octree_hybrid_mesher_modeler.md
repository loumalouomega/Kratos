---
title: OctreeHybridMesherModeler
keywords: mesh hex hexahedral octree adaptive dual primal modeler hanging-node constraints
tags: [mesh hexahedral octree modeler]
sidebar: kratos_core_utilities
summary: Registry-driven modeler that wraps the OctreeHybridMeshUtility engine to produce all-hex ModelParts with optional surface projection and hanging-node constraints.
---

# OctreeHybridMesherModeler

## Table of contents

1. [What this modeler does](#1-what-this-modeler-does)
2. [Architecture: the Registry-prototype pattern](#2-architecture-the-registry-prototype-pattern)
3. [Pipeline stages](#3-pipeline-stages)
   - 3.1 [Octree generation — `BuildOctreeAndExtract`](#31-octree-generation--buildoctreeandextract)
   - 3.2 [Coloring — `coloring_settings_list`](#32-coloring--coloring_settings_list)
   - 3.3 [Entity generation — `entities_generator_list`](#33-entity-generation--entities_generator_list)
   - 3.4 [Operations — `model_part_operations`](#34-operations--model_part_operations)
4. [Shared state: OctreeHybridMesherData](#4-shared-state-octreemesherdata)
5. [Mesh topologies](#5-mesh-topologies)
   - 5.1 [Dual mesh (default)](#51-dual-mesh-default)
   - 5.2 [Primal mesh with hanging-node constraints](#52-primal-mesh-with-hanging-node-constraints)
6. [Full JSON parameters schema](#6-full-json-parameters-schema)
7. [Registered components](#7-registered-components)
   - 7.1 [Coloring components](#71-coloring-components)
   - 7.2 [Entity-generation components](#72-entity-generation-components)
   - 7.3 [Operation components](#73-operation-components)
8. [Python / JSON usage examples](#8-python--json-usage-examples)
   - 8.1 [Dual carved mesh](#81-dual-carved-mesh)
   - 8.2 [Primal mesh with hanging-node constraints](#82-primal-mesh-with-hanging-node-constraints)
   - 8.3 [Dual mesh with surface projection](#83-dual-mesh-with-surface-projection)
   - 8.4 [Boundary conditions on the exterior surface](#84-boundary-conditions-on-the-exterior-surface)
   - 8.5 [Quality report](#85-quality-report)
9. [API reference](#9-api-reference)
10. [Registration and instantiation](#10-registration-and-instantiation)
11. [Testing](#11-testing)
12. [Known limitations](#12-known-limitations)

---

## 1. What this modeler does

`OctreeHybridMesherModeler` is a Kratos `Modeler` subclass that converts a closed, orientable
triangular surface `ModelPart` into an **all-hexahedral volumetric ModelPart** using the
HybridOctree_Hex algorithm implemented in `OctreeHybridMeshUtility`.

It sits in the standard Kratos modeler pipeline (`SetupGeometryModel` →
`PrepareGeometryModel` → `SetupModelPart`) and the entire mesh generation happens inside
`SetupModelPart`.  The modeler:

1. **Builds and 2:1-balances an adaptive octree** around the input surface using
   `OctreeHybridMeshUtility::BuildFromSurfaceMesh`.
2. **Extracts a hex mesh** from the octree into an in-memory flat representation
   (node coordinates, hex connectivity, per-cell level).  Two topologies are
   available:
   - **Dual mesh** (`mesh_type: "dual"`): the conforming all-hex dual of the octree.
     Each octree vertex shared by 8 equally-refined leaves produces one regular hex;
     refinement-level boundaries are handled by a finite library of transition
     templates that keep the mesh conforming.  Optionally, the mesh can be carved
     against the input surface and its boundary projected onto the surface
     (`project_to_surface: true`).
   - **Primal mesh** (`mesh_type: "primal"`): one hexahedron per octree leaf cell.
     The mesh is non-conforming at 2:1 level transitions; the modeler records the
     hanging-node interpolation constraints so that they can be enforced via
     `LinearMasterSlaveConstraint` objects.
3. **Dispatches registered component stages** — coloring, entity generation, and
   post-processing operations — by resolving each component's `"type"` string
   against the Kratos Registry.

The design mirrors the `VoxelMeshGeneratorModeler` architecture but uses the octree
engine (adaptive refinement, conforming transition templates) instead of a Cartesian
voxel grid, and dispatches components through the Registry-prototype pattern rather
than hand-written factory maps.

---

## 2. Architecture: the Registry-prototype pattern

Every pluggable component (coloring, entity generation, operation) is a stateless C++
class that:

1. Derives from the relevant abstract base (`OctreeHybridMesherColoring`,
   `OctreeHybridMesherEntityGeneration`, or `OctreeHybridMesherOperation`).
2. Declares two `KRATOS_REGISTRY_ADD_PROTOTYPE` entries in its `private` section —
   one under the `KratosMultiphysics` sub-path and one under `All`.  These macros
   create a static `RegistryItem` that inserts a shared instance of the class into
   the global Kratos Registry at load time (before `main`).

```cpp
// Example — inside OctreeHybridClassifyCellsInsideOutside:
KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics",
                               OctreeHybridMesherColoring, OctreeHybridClassifyCellsInsideOutside)
KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All",
                               OctreeHybridMesherColoring, OctreeHybridClassifyCellsInsideOutside)
```

This registers the prototype at the path
`OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype` (and the
`KratosMultiphysics` variant).

When `SetupModelPart` processes a stage list, the private `Dispatch<TBase>` template
iterates over each entry in the JSON array, resolves the `"type"` string to a full
registry path, retrieves the shared prototype via `Registry::GetValue<TBase>`, and
calls the do-work method on it:

```cpp
template<class TBase, class TInvoke>
void Dispatch(const std::string& rRegistryRoot, Parameters StageList, TInvoke&& Invoke)
{
    for (Parameters stage_params : StageList) {
        std::string type = stage_params["type"].GetString();
        const auto segments = StringUtilities::SplitStringByDelimiter(type, '.');
        const std::string full_path = (segments.size() == 4)
            ? type
            : rRegistryRoot + ".All." + type + ".Prototype";
        const TBase& r_prototype = Registry::GetValue<TBase>(full_path);
        r_prototype.ValidateParameters(stage_params);
        Invoke(r_prototype, stage_params);
    }
}
```

**Path resolution rules:**

| `"type"` value | Resulting registry path |
|----------------|------------------------|
| `"OctreeHybridClassifyCellsInsideOutside"` | `OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype` |
| `"OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype"` (4 dots) | used as-is |

Because the retrieved prototype is a **shared, stateless object**, the do-work methods
(`Apply`, `Generate`, `Execute`) are declared `const`.  All mutable state lives in the
`OctreeHybridMesherModeler` argument passed to each call — specifically in the
`OctreeHybridMesherData` struct held by the modeler.  This makes the prototype objects
inherently thread-safe (they hold no data) and avoids any per-invocation allocation
of component objects.

---

## 3. Pipeline stages

### High-level order

```
SetupModelPart()
    │
    ├─ 1. BuildOctreeAndExtract()        [always runs; not Registry-dispatched]
    │      • build + 2:1-balance octree
    │      • extract dual or primal hex mesh
    │      • optionally carve + project to surface (dual only)
    │
    ├─ 2. Dispatch<OctreeHybridMesherColoring>  [coloring_settings_list]
    │      for each entry: r_prototype.Apply(*this, params)
    │
    ├─ 3. Dispatch<OctreeHybridMesherEntityGeneration>  [entities_generator_list]
    │      for each entry: r_prototype.Generate(*this, params)
    │
    └─ 4. Dispatch<OctreeHybridMesherOperation>  [model_part_operations]
           for each entry: r_prototype.Execute(*this, params)
```

**Stage–Base class–Dispatch list** summary:

| # | Stage | Base class | JSON key | Purpose |
|---|-------|-----------|----------|---------|
| 1 | Octree generation | *(internal)* | `octree_generator` | Build octree, extract hex mesh |
| 2 | Coloring | `OctreeHybridMesherColoring` | `coloring_settings_list` | Classify cells inside/outside |
| 3 | Entity generation | `OctreeHybridMesherEntityGeneration` | `entities_generator_list` | Create nodes, elements, conditions, constraints |
| 4 | Operations | `OctreeHybridMesherOperation` | `model_part_operations` | Post-processing (e.g. quality report) |

---

### 3.1 Octree generation — `BuildOctreeAndExtract`

This internal step is always run first and is not Registry-dispatched.  Its
parameters come from the top-level `"octree_generator"` block.

**Inputs:**

- The surface `ModelPart` identified by `octree_generator.input_model_part_name`
  (falls back to the top-level `input_model_part_name`).  The model part must contain
  `Triangle3D3` geometries (typically loaded via `StlIO::ReadModelPart`).

**What it does:**

1. Calls `OctreeHybridMeshUtility::ExtractTriangleSoup` to copy the surface triangles
   into world-space for later carving/projection/classification.
2. Calls `OctreeHybridMeshUtility::BuildFromSurfaceMesh` to build and 2:1-balance the
   adaptive octree.  When `adaptive: true` (the default), curvature and thickness
   criteria are used to determine the refinement level per surface region, matching
   the reference HybridOctree_Hex octree cell-for-cell.  When `adaptive: false`,
   every cell intersecting the surface is uniformly refined to `refinement_depth`.
3. After building, calls `StrongConstrain2To1` to enforce the 2:1 balance constraint
   across the whole tree.
4. Depending on `mesh_type`:

   **`mesh_type: "dual"` (default)**

   - Calls `OctreeHybridMeshUtility::ExtractDualHexMesh`, which runs the
     face-adjacency detection and transition-template emission pass to produce a
     conforming all-hex dual mesh stored as flat arrays in `OctreeHybridMesherData`.
   - If `project_to_surface: true` and the triangle soup is non-empty:
     - `RemoveOutsideElement`: carves away hexes whose centroids are outside the
       surface (ray-cast inside/outside parity test + signed-distance filter).
     - `ClearBufferZone`: removes boundary hexes that create non-manifold topology
       (hemisphere-probe clearance so the extruded shell cannot self-intersect).
     - `ProjectToIsoSurface`: runs the Jacobian-controlled optimiser to pull boundary
       nodes onto the input surface.  The iteration budget is controlled by
       `projection_iterations` and `projection_smoothing`.
     - Sets `OctreeHybridMesherData::mProjected = true`.

   **`mesh_type: "primal"`**

   - Calls `OctreeHybridMeshUtility::ExtractPrimalHexMesh`, which enumerates each
     octree leaf as one hex cell.  The connectivity is non-conforming at 2:1
     transitions.  Hanging-node constraint records are stored in
     `OctreeHybridMesherData::mHanging`.

5. Initialises `OctreeHybridMesherData::mNodePtrs` (size = number of nodes, all null) for
   lazy de-duplication during entity generation.

**Key parameter decisions:**

| `mesh_type` | `adaptive` | `project_to_surface` | Result |
|------------|-----------|----------------------|--------|
| `"dual"` | `true` | `false` | Conforming dual block (whole octree bbox), adaptive refinement, NOT carved |
| `"dual"` | `true` | `false` + coloring | Conforming dual block carved by coloring stage |
| `"dual"` | `true` | `true` | Surface-projected carved mesh |
| `"primal"` | `true` | `false` | One hex per leaf; hanging-node records in `mHanging` |
| `"primal"` | `false` | *(ignored)* | Uniform primal mesh; projection not supported |

> **Note:** `project_to_surface: true` is only meaningful for `mesh_type: "dual"`.
> For the primal mesh, surface projection is not implemented.

---

### 3.2 Coloring — `coloring_settings_list`

Coloring stages write an integer label into `OctreeHybridMesherData::mCellColor` (one entry
per hex cell).  Downstream entity-generation stages filter on this label.

The coloring list is processed in order; multiple coloring stages can be stacked, but
in practice a single `OctreeHybridClassifyCellsInsideOutside` entry is sufficient for most use
cases.

The canonical colour convention is:

| Label | Meaning |
|-------|---------|
| `1` | Inside the input surface |
| `0` | Outside the input surface |

When using `mesh_type: "dual"` **without** `project_to_surface`, the coloring stage is
responsible for the inside/outside carving.  When `project_to_surface: true`, the
projection pass has already removed all outside cells and set `mProjected = true`;
`OctreeHybridClassifyCellsInsideOutside` then short-circuits with a single `assign(n, 1)` call
instead of running the ray-caster.

When using `mesh_type: "primal"`, the coloring list can be left empty (all cells
included), or `OctreeHybridClassifyCellsInsideOutside` can be run to carve away outside cells
before entity generation.

---

### 3.3 Entity generation — `entities_generator_list`

Entity-generation stages transform the flat in-memory hex mesh into Kratos entities
(nodes, elements, conditions, master-slave constraints) inside one or more ModelParts.

Stages are processed in order.  Node de-duplication is shared across stages via
`OctreeHybridMesherData::mNodePtrs`: the first stage that needs a given mesh-node index
creates a `Node` object and caches the pointer; subsequent stages sharing the same
node reuse it without creating a duplicate.

This means that a single `SetupModelPart` call can populate multiple sub-ModelParts
(e.g. a volume ModelPart and a boundary ModelPart) and they will share the same
underlying node objects.

Registered entity-generation components:

| JSON `"type"` | Class | Purpose |
|--------------|-------|---------|
| `OctreeHybridGenerateHexesByCellColor` | `OctreeHybridGenerateHexesByCellColor` | Create hex elements for cells matching a colour |
| `OctreeHybridGenerateBoundaryConditionsByFace` | `OctreeHybridGenerateBoundaryConditionsByFace` | Create quad conditions on the outer surface |
| `OctreeHybridGenerateHangingNodeConstraints` | `OctreeHybridGenerateHangingNodeConstraints` | Create `LinearMasterSlaveConstraint` for primal mesh |

---

### 3.4 Operations — `model_part_operations`

Operations run after entity generation on the finished ModelPart.  They are
read-only or topology-preserving passes — they do not create or remove entities.

Registered operations:

| JSON `"type"` | Class | Purpose |
|--------------|-------|---------|
| `OctreeHybridReportMeshQuality` | `OctreeHybridReportMeshQuality` | Log min/mean scaled Jacobian and inverted-element count |

---

## 4. Shared state: OctreeHybridMesherData

`OctreeHybridMesherData` (in `kratos/modeler/internals/octree_hybrid_mesher_data.h`) is the central
shared-state struct that all pipeline stages read from and write to.  The modeler owns
it as a `std::unique_ptr<OctreeHybridMesherData>` and exposes it through `GetData()`.

| Field | Type | Written by | Read by | Description |
|-------|------|------------|---------|-------------|
| `mpOctree` | `unique_ptr<OctreeType>` | `BuildOctreeAndExtract` | Coloring | The built and 2:1-balanced octree. |
| `mTriangles` | `TriangleSoup` | `BuildOctreeAndExtract` | Coloring, projection | World-space surface triangles for carving/projection/classification. |
| `mNodes` | `vector<array<double,3>>` | `BuildOctreeAndExtract` | All stages | World-space coordinates of the hex mesh nodes. |
| `mCells` | `vector<array<int,8>>` | `BuildOctreeAndExtract` | All stages | Hex connectivity (8 node indices per cell, Hexahedra3D8 ordering). |
| `mCellLevel` | `vector<int>` | `BuildOctreeAndExtract` | Entity generation, quality report | Octree refinement level per cell (-1 for transition-template hexes). |
| `mCellColor` | `vector<int>` | Coloring stages | Entity generation | Per-cell inside(1)/outside(0) label. Empty until coloring runs. |
| `mHanging` | `vector<HangingConstraint>` | `BuildOctreeAndExtract` | `OctreeHybridGenerateHangingNodeConstraints` | Hanging-node interpolation records (primal mesh only). |
| `mNodePtrs` | `vector<Node::Pointer>` | Entity generation (lazy) | Entity generation | De-duplication cache: mesh-node index -> ModelPart Node. Null until the node is first needed. |
| `mProjected` | `bool` | `BuildOctreeAndExtract` | `OctreeHybridClassifyCellsInsideOutside` | True when surface projection has been applied; triggers the classification short-circuit. |

`IsExtracted()` returns `true` once `mCells` is non-empty (i.e. after
`BuildOctreeAndExtract` completes).

---

## 5. Mesh topologies

### 5.1 Dual mesh (default)

`mesh_type: "dual"` produces the **conforming** all-hex dual of the octree.

**How it works:**

- One *dual node* is placed at the centroid of each octree leaf cell.
- Each octree primal vertex shared by exactly 8 leaves of the same refinement level
  produces one regular hexahedron whose 8 corners are the 8 leaf centroids
  (connected in Hexahedra3D8 order via the `idTransform` permutation).
- At 2:1 refinement-level boundaries, a library of 12 hand-crafted hexahedral
  transition templates fills the geometric gap.  Each transition region is processed
  once; a `consumed[]` flag prevents duplicate coverage.

**Properties:**

- **Conforming**: adjacent hexes share complete faces or complete edges — no
  T-junctions in the raw dual block.
- **No hanging nodes**: the connectivity is conforming, so no MPC constraints are
  needed.
- **Transition templates**: hexes in a transition region receive `mCellLevel = -1`;
  regular dual hexes carry the octree level of their leaf cell.
- **Full bounding-box block**: `ExtractDualHexMesh` alone produces a hex mesh
  covering the entire octree bounding box.  The inside/outside carving is done
  separately by the coloring stage (`OctreeHybridClassifyCellsInsideOutside`) or by the
  `project_to_surface` pass.

**Refinement levels** on elements (accessible via the `REFINEMENT_LEVEL` variable
when `tag_refinement_level: true`):

| Value | Meaning |
|-------|---------|
| `1` .. `N` | Octree leaf level at which this dual hex was generated |
| `-1` | Hex produced by a transition template (at a refinement-level boundary) |
| `-2` | Buffer-layer hex added during surface projection |

---

### 5.2 Primal mesh with hanging-node constraints

`mesh_type: "primal"` produces **one hexahedron per octree leaf cell**.

**How it works:**

- Each octree leaf cell maps directly to one hex.  Corner indices are computed
  from the leaf's grid position and level at a fixed `MAX_DEPTH` integer
  resolution.  Adjacent cells of the same level share corner indices; cells of
  different levels share only a subset of their corners.
- At a 2:1 transition, a fine cell's edge-midpoint node sits on the face of a
  coarser neighbour but is not a corner of that coarser cell.  This is a
  *hanging node*.  `ExtractPrimalHexMesh` records these in `HangingConstraint`
  structs stored in `OctreeHybridMesherData::mHanging`.

**Hanging-node constraint weights:**

The linear relation enforced by each `LinearMasterSlaveConstraint` is:

```
u_slave = sum_m (w_m * u_master_m)
```

The weights are bilinear interpolation coefficients:

| Hanging-node type | Number of masters | Weights |
|------------------|--------------------|---------|
| Edge-midpoint | 2 | 0.5, 0.5 |
| Face-centre | 4 | 0.25, 0.25, 0.25, 0.25 |

One constraint is created per (hanging node, DOF variable) pair.  The `"variables"`
parameter of `OctreeHybridGenerateHangingNodeConstraints` lists which DOF variables to constrain
(default: `DISPLACEMENT_X`, `DISPLACEMENT_Y`, `DISPLACEMENT_Z`).

**Properties:**

- **Non-conforming**: the mesh has T-junctions at 2:1 refinement transitions.
- **Hanging nodes**: compatibility is enforced through MPC constraints, not through
  mesh conformity.
- **Simple topology**: one hex per octree cell means no template geometry; all
  hexes are axis-aligned.
- **No surface projection**: `project_to_surface` is silently ignored for primal
  meshes.  Carving by the coloring stage is still possible.

---

## 6. Full JSON parameters schema

The complete parameter block accepted by `OctreeHybridMesherModeler`:

```json
{
    "echo_level": 0,
    "input_model_part_name": "",
    "output_model_part_name": "",
    "octree_generator": {
        "type"                 : "generate_octree_from_surface",
        "input_model_part_name": "",
        "refinement_depth"     : 5,
        "adaptive"             : true,
        "mesh_type"            : "dual",
        "project_to_surface"   : false,
        "projection_iterations": 20000,
        "projection_smoothing" : 1000
    },
    "coloring_settings_list"  : [],
    "entities_generator_list" : [],
    "model_part_operations"   : []
}
```

**Top-level keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `echo_level` | int | `0` | Verbosity level (0 = silent, higher = more output). |
| `input_model_part_name` | string | `""` | Name of the surface ModelPart to mesh. Used as fallback when `octree_generator.input_model_part_name` is empty. |
| `output_model_part_name` | string | `""` | Reserved for future use; individual stages specify their own target ModelPart names. |
| `octree_generator` | object | see below | Octree construction and mesh-extraction settings. |
| `coloring_settings_list` | array | `[]` | Ordered list of coloring stage descriptors. |
| `entities_generator_list` | array | `[]` | Ordered list of entity-generation stage descriptors. |
| `model_part_operations` | array | `[]` | Ordered list of post-processing operation descriptors. |

**`octree_generator` keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"generate_octree_from_surface"` | Reserved; only one type is currently supported. |
| `input_model_part_name` | string | `""` | Name of the surface ModelPart.  If empty, falls back to the top-level `input_model_part_name`. |
| `refinement_depth` | int | `5` | Maximum octree refinement level near the surface.  Range: `[1, MAX_DEPTH=10]`. |
| `adaptive` | bool | `true` | `true` = curvature + thickness adaptive refinement (matches HybridOctree_Hex reference cell-for-cell); `false` = uniform refinement of all surface-intersecting cells. |
| `mesh_type` | string | `"dual"` | `"dual"` = conforming all-hex dual mesh; `"primal"` = one hex per octree leaf with hanging-node records. |
| `project_to_surface` | bool | `false` | Dual mesh only.  When `true`, additionally carves the mesh against the surface (`RemoveOutsideElement`), clears non-manifold boundary regions (`ClearBufferZone`), and runs the Jacobian-controlled surface projector (`ProjectToIsoSurface`).  Sets `mProjected = true`. |
| `projection_iterations` | int | `20000` | Number of optimiser iterations in the surface projection pass.  Higher values improve the minimum scaled Jacobian at the cost of longer runtime. |
| `projection_smoothing` | int | `1000` | Interval (in iterations) at which a gated smart-Laplacian smoothing pass is applied during surface projection. |

**Stage descriptor object (common to all three lists):**

Each entry in `coloring_settings_list`, `entities_generator_list`, and
`model_part_operations` is a JSON object whose first key must be `"type"`.  The
remaining keys are specific to the component and are described in
[§7 Registered components](#7-registered-components).

```json
{ "type": "ComponentTypeName", ... }
```

---

## 7. Registered components

### 7.1 Coloring components

#### `OctreeHybridClassifyCellsInsideOutside`

Classifies every hex cell as inside (label 1) or outside (label 0) the input surface.

**Registry path:** `OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype`

**Class:** `Kratos::OctreeHybridClassifyCellsInsideOutside`

**Header:** `kratos/modeler/coloring/octree_hybrid_classify_cells_inside_outside.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridClassifyCellsInsideOutside"` | Registry lookup key. |

**Behaviour:**

- If `OctreeHybridMesherData::mProjected == true` (surface projection was applied), every
  surviving cell is definitively inside; the method assigns `1` to all entries in
  `mCellColor` with a single `std::vector::assign` call and returns immediately,
  skipping the ray-caster entirely.
- Otherwise, calls `OctreeHybridMeshUtility::ClassifyInsideOutside`, which for each
  cell shoots a random ray from the cell centroid, counts surface-triangle crossings
  for the inside/outside sign, and uses closest-triangle distance for the magnitude.
  The result is stored in `mCellColor`.

**Example JSON:**

```json
{ "type": "OctreeHybridClassifyCellsInsideOutside" }
```

---

### 7.2 Entity-generation components

#### `OctreeHybridGenerateHexesByCellColor`

Creates one 8-noded hexahedral element per cell whose colour matches the configured
value.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateHexesByCellColor.Prototype`

**Class:** `Kratos::OctreeHybridGenerateHexesByCellColor`

**Header:** `kratos/modeler/entity_generation/octree_hybrid_generate_hexes_by_cell_color.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridGenerateHexesByCellColor"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label to emit (e.g. `1` for inside cells). |
| `properties_id` | int | `1` | Properties block ID assigned to every new element (created on demand). |
| `generated_entity` | string | `"Element3D8N"` | Registered element type name (`KratosComponents<Element>::Get`). |
| `tag_refinement_level` | bool | `true` | When `true`, stores the cell's octree refinement level in the element's `REFINEMENT_LEVEL` variable. |

**Behaviour:**

Iterates `mCells` in order.  For each cell `c` where `mCellColor[c] == color`:
1. Resolves the 8 corner mesh-node indices.
2. For each corner, calls `OctreeHybridMesherModeler::GenerateOrRetrieveNode` — creates a
   new node on the first encounter or returns the cached pointer for subsequent cells
   sharing the same node.
3. Creates an element using the registered prototype for `generated_entity`.
4. Optionally sets `REFINEMENT_LEVEL` to `mCellLevel[c]`.
5. Bulk-inserts the new nodes and elements into the target ModelPart.

Node-ordering: the hybrid octree engine stores cell corners in the order matching
Kratos `Hexahedra3D8` local-node numbering, so no remapping is required.

**Example JSON:**

```json
{
    "type"                : "OctreeHybridGenerateHexesByCellColor",
    "model_part_name"     : "FluidDomain",
    "color"               : 1,
    "properties_id"       : 1,
    "generated_entity"    : "Element3D8N",
    "tag_refinement_level": true
}
```

---

#### `OctreeHybridGenerateBoundaryConditionsByFace`

Creates quadrilateral boundary conditions on the outer surface of the coloured hex
mesh.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateBoundaryConditionsByFace.Prototype`

**Class:** `Kratos::OctreeHybridGenerateBoundaryConditionsByFace`

**Header:** `kratos/modeler/entity_generation/octree_hybrid_generate_boundary_conditions_by_face.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridGenerateBoundaryConditionsByFace"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label identifying the volume to extract the boundary of. |
| `properties_id` | int | `1` | Properties block ID assigned to every new condition. |
| `generated_entity` | string | `"SurfaceCondition3D4N"` | Registered condition type name. |

**Behaviour:**

1. Filters `mCells` to retain only cells with the requested `color`.
2. Calls `OctreeHybridMeshUtility::ExtractBoundaryFaces` on the filtered cell list
   to identify outer boundary quads (faces owned by exactly one hex).
3. For each boundary quad, creates the four corner nodes via
   `GenerateOrRetrieveNode` (reuses nodes already created by a prior hex-generation
   stage) and constructs a condition using `generated_entity`.
4. Bulk-inserts all nodes and conditions into the target ModelPart.

Winding convention: outward normals follow the convention of `ExtractBoundaryFaces`.

**Example JSON:**

```json
{
    "type"             : "OctreeHybridGenerateBoundaryConditionsByFace",
    "model_part_name"  : "Boundary",
    "color"            : 1,
    "properties_id"    : 1,
    "generated_entity" : "SurfaceCondition3D4N"
}
```

---

#### `OctreeHybridGenerateHangingNodeConstraints`

Creates `LinearMasterSlaveConstraint` objects for hanging nodes in the primal mesh.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.OctreeHybridGenerateHangingNodeConstraints.Prototype`

**Class:** `Kratos::OctreeHybridGenerateHangingNodeConstraints`

**Header:** `kratos/modeler/entity_generation/octree_hybrid_generate_hanging_node_constraints.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridGenerateHangingNodeConstraints"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | ModelPart to add constraints to (must already exist with the mesh nodes). |
| `constraint_name` | string | `"LinearMasterSlaveConstraint"` | Registered constraint type to instantiate. |
| `variables` | string array | `["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]` | Scalar DOF variables to constrain (each must be a registered `Variable<double>`). |

**Behaviour:**

For each `HangingConstraint` record in `mData.mHanging`:
1. The slave node pointer is looked up in `mData.mNodePtrs`; the record is skipped
   if the pointer is null (node was not created, e.g. because the cell was
   classified outside and not emitted).
2. All master node pointers are looked up; the record is skipped if any master
   pointer is null.
3. For each variable in `"variables"`:
   - `Node::AddDof` is called on the slave and all master nodes.
   - A `1 x N_masters` relation matrix is built from `HangingConstraint::Weights`.
   - `ModelPart::CreateNewMasterSlaveConstraint` is called with the constraint type
     name, a fresh ID from `OctreeHybridMesherModeler::NextConstraintId`, and the
     assembled DOF vectors.

The imposed linear relation is:
```
u_slave = w_0 * u_master_0 + w_1 * u_master_1 [+ w_2 * u_master_2 + w_3 * u_master_3]
```
where the weights satisfy partition-of-unity (`sum(w_i) == 1.0`).

**Prerequisite:** The primal hex-generation step (`OctreeHybridGenerateHexesByCellColor` with
`mesh_type: "primal"`) must have run first so that `mData.mNodePtrs` is populated.

**Prerequisite — coloring:** If a coloring stage (e.g. `OctreeHybridClassifyCellsInsideOutside`)
was run before hex generation, only inside cells were created.  Hanging-node records
whose slave or master nodes belong to carved-away outside cells are silently skipped;
no error is thrown.

**Example JSON:**

```json
{
    "type"            : "OctreeHybridGenerateHangingNodeConstraints",
    "model_part_name" : "FluidDomain",
    "constraint_name" : "LinearMasterSlaveConstraint",
    "variables"       : ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"]
}
```

---

### 7.3 Operation components

#### `OctreeHybridReportMeshQuality`

Logs scaled-Jacobian statistics for all hexahedral elements in a ModelPart.

**Registry path:** `OctreeHybridMesherOperation.All.OctreeHybridReportMeshQuality.Prototype`

**Class:** `Kratos::OctreeHybridReportMeshQuality`

**Header:** `kratos/modeler/operation/octree_hybrid_report_mesh_quality.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridReportMeshQuality"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the ModelPart to analyse. |

**Behaviour:**

Iterates all elements in the ModelPart (elements with fewer or more than 8 nodes are
skipped silently).  For each 8-noded element, computes the minimum scaled Jacobian
over the body centre and 8 corners using
`OctreeHybridMeshUtility::ScaledJacobianMin`.  At the end, logs the following
statistics at `INFO` level via `KRATOS_INFO("OctreeHybridReportMeshQuality")`:

- `minSJ`: minimum scaled Jacobian across all elements (worst element quality).
- `meanSJ`: arithmetic mean of per-element minimum scaled Jacobians.
- `inverted`: count and percentage of elements with scaled Jacobian `<= 0`.

This operation is purely read-only and does not modify any mesh entity.

**Example JSON:**

```json
{
    "type"            : "OctreeHybridReportMeshQuality",
    "model_part_name" : "FluidDomain"
}
```

---

## 8. Python / JSON usage examples

### 8.1 Dual carved mesh

The simplest usage: build an adaptive dual hex mesh of the object interior from an
STL surface.

```python
import KratosMultiphysics as KM

# --- Load the surface ---
model = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
KM.StlIO("my_surface.stl", KM.Parameters('{"open_mode":"read"}')).ReadModelPart(surface_mp)

# --- Configure and run the modeler ---
settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "output_model_part_name" : "Volume",
    "octree_generator" : {
        "type"              : "generate_octree_from_surface",
        "refinement_depth"  : 5,
        "adaptive"          : true,
        "mesh_type"         : "dual"
    },
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"             : "OctreeHybridGenerateHexesByCellColor",
            "model_part_name"  : "Volume",
            "color"            : 1,
            "properties_id"    : 1,
            "generated_entity" : "Element3D8N",
            "tag_refinement_level" : true
        }
    ],
    "model_part_operations" : []
}""")

modeler = KM.OctreeHybridMesherModeler(model, settings)
modeler.SetupGeometryModel()
modeler.PrepareGeometryModel()
modeler.SetupModelPart()

# --- Inspect the result ---
volume = model.GetModelPart("Volume")
print(f"Nodes   : {volume.NumberOfNodes()}")
print(f"Elements: {volume.NumberOfElements()}")
```

---

### 8.2 Primal mesh with hanging-node constraints

Use the primal topology for approaches that can handle multi-point constraints
(e.g. IGA, Nitsche, or MPC-aware linear solvers).

```python
import KratosMultiphysics as KM

model = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
KM.StlIO("my_surface.stl", KM.Parameters('{"open_mode":"read"}')).ReadModelPart(surface_mp)

settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "output_model_part_name" : "Domain",
    "octree_generator" : {
        "type"              : "generate_octree_from_surface",
        "refinement_depth"  : 4,
        "adaptive"          : true,
        "mesh_type"         : "primal"
    },
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"            : "OctreeHybridGenerateHexesByCellColor",
            "model_part_name" : "Domain",
            "color"           : 1,
            "properties_id"   : 1
        },
        {
            "type"            : "OctreeHybridGenerateHangingNodeConstraints",
            "model_part_name" : "Domain",
            "constraint_name" : "LinearMasterSlaveConstraint",
            "variables"       : ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"]
        }
    ],
    "model_part_operations" : [
        {
            "type"            : "OctreeHybridReportMeshQuality",
            "model_part_name" : "Domain"
        }
    ]
}""")

modeler = KM.OctreeHybridMesherModeler(model, settings)
modeler.SetupGeometryModel()
modeler.PrepareGeometryModel()
modeler.SetupModelPart()

domain = model.GetModelPart("Domain")
print(f"Nodes      : {domain.NumberOfNodes()}")
print(f"Elements   : {domain.NumberOfElements()}")
print(f"Constraints: {domain.NumberOfMasterSlaveConstraints()}")
```

---

### 8.3 Dual mesh with surface projection

Enable `project_to_surface` to produce a mesh whose boundary conforms to the input
surface (instead of the blocky staircase carve).  This is the highest-quality option
and is suitable for CFD and solid-mechanics simulations where surface accuracy matters.

```python
import KratosMultiphysics as KM

model = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
KM.StlIO("bunny.stl", KM.Parameters('{"open_mode":"read"}')).ReadModelPart(surface_mp)

settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "output_model_part_name" : "Volume",
    "octree_generator" : {
        "type"                 : "generate_octree_from_surface",
        "refinement_depth"     : 5,
        "adaptive"             : true,
        "mesh_type"            : "dual",
        "project_to_surface"   : true,
        "projection_iterations": 20000,
        "projection_smoothing" : 1000
    },
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"             : "OctreeHybridGenerateHexesByCellColor",
            "model_part_name"  : "Volume",
            "color"            : 1,
            "properties_id"    : 1,
            "generated_entity" : "Element3D8N"
        }
    ],
    "model_part_operations" : [
        {
            "type"            : "OctreeHybridReportMeshQuality",
            "model_part_name" : "Volume"
        }
    ]
}""")

modeler = KM.OctreeHybridMesherModeler(model, settings)
modeler.SetupGeometryModel()
modeler.PrepareGeometryModel()
modeler.SetupModelPart()
```

The `"coloring_settings_list"` entry is still required even when
`project_to_surface: true` — it assigns the colour labels that entity generation
filters on.  Because `mProjected == true`, the classification short-circuit fires and
no ray-casting occurs.

---

### 8.4 Boundary conditions on the exterior surface

Combine `OctreeHybridGenerateHexesByCellColor` and `OctreeHybridGenerateBoundaryConditionsByFace` to populate
both a volume ModelPart and a boundary ModelPart in a single `SetupModelPart` call.

```python
settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "output_model_part_name" : "Volume",
    "octree_generator" : {
        "refinement_depth" : 4,
        "adaptive"         : true,
        "mesh_type"        : "dual"
    },
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"            : "OctreeHybridGenerateHexesByCellColor",
            "model_part_name" : "Volume",
            "color"           : 1,
            "properties_id"   : 1
        },
        {
            "type"             : "OctreeHybridGenerateBoundaryConditionsByFace",
            "model_part_name"  : "Boundary",
            "color"            : 1,
            "properties_id"    : 1,
            "generated_entity" : "SurfaceCondition3D4N"
        }
    ],
    "model_part_operations" : []
}""")
```

The `Boundary` ModelPart will contain the exterior quad conditions and the shared
nodes; nodes created first by `OctreeHybridGenerateHexesByCellColor` are reused (not duplicated)
by `OctreeHybridGenerateBoundaryConditionsByFace` through the `mNodePtrs` de-duplication cache.

---

### 8.5 Quality report

Add `OctreeHybridReportMeshQuality` to the operations list to print scaled-Jacobian statistics
after mesh generation.  It runs last and does not alter the mesh.

```json
"model_part_operations" : [
    {
        "type"            : "OctreeHybridReportMeshQuality",
        "model_part_name" : "Volume"
    }
]
```

Expected log output (example, depth-4 dual box mesh):

```
[OctreeHybridReportMeshQuality] minSJ=0.472  meanSJ=0.891  inverted=0 (0.0%)
```

---

## 9. API reference

### `OctreeHybridMesherModeler` — public interface

```cpp
#include "modeler/octree_hybrid_mesher_modeler.h"
```

---

#### Constructors

```cpp
OctreeHybridMesherModeler();
OctreeHybridMesherModeler(Model& rModel, Parameters ModelerParameters = Parameters());
```

The default constructor is used internally when the Registry prototype is created.
The second constructor is the one called from Python or from `KratosModelParametersFactory`.
`ModelerParameters` is validated and defaults are assigned in the constructor.

---

#### `SetupModelPart`

```cpp
void SetupModelPart() override;
```

The main entry point.  Runs the full pipeline:
1. `BuildOctreeAndExtract` (internal).
2. Dispatch coloring stages.
3. Dispatch entity-generation stages.
4. Dispatch operation stages.

Called after `SetupGeometryModel` and `PrepareGeometryModel` (both no-ops in this
modeler).

---

#### `GetData`

```cpp
Internals::OctreeHybridMesherData& GetData();
```

Returns the shared mesher state.  Used by all pipeline component stages to read the
octree, the extracted hex mesh, and the per-cell colour array, and to write back
entity pointers.

---

#### `CreateAndGetModelPart`

```cpp
ModelPart& CreateAndGetModelPart(const std::string& rFullName);
```

Returns the ModelPart with the given full name, creating it in `mpModel` if it does
not exist yet.  Used by entity-generation stages to obtain their target ModelPart.

---

#### `GenerateOrRetrieveNode`

```cpp
Node::Pointer GenerateOrRetrieveNode(
    ModelPart& rModelPart,
    ModelPart::NodesContainerType& rNewNodes,
    int NodeIndex);
```

Returns the Kratos `Node` corresponding to mesh-node index `NodeIndex`.  On the first
call for a given `NodeIndex`:
- Creates a new `Node` with the world coordinates from `mData.mNodes[NodeIndex]`.
- Sets the solution-step variables list and buffer size from `rModelPart`.
- Caches the pointer in `mData.mNodePtrs[NodeIndex]`.
- Appends the node to `rNewNodes`.

On subsequent calls for the same index, returns the cached pointer immediately without
creating a new node.  The node ID is consumed from the internal `mStartNodeId` counter
(incremented atomically per call to this function).

---

#### `SetStartIds`

```cpp
void SetStartIds(ModelPart& rModelPart);
```

Seeds the four ID counters (`mStartNodeId`, `mStartElementId`, `mStartConditionId`,
`mStartConstraintId`) from the current highest IDs in the root model part, ensuring
new entities get IDs that are strictly greater than any pre-existing entity.  Entity-
generation stages call this before creating the first entity in a new ModelPart.

---

#### ID counter accessors

```cpp
std::size_t NextElementId();    // returns mStartElementId++
std::size_t NextConditionId();  // returns mStartConditionId++
std::size_t NextConstraintId(); // returns mStartConstraintId++
```

Used by entity-generation stages to obtain unique, monotonically increasing IDs for
elements, conditions, and constraints respectively.

---

#### `GetDefaultParameters`

```cpp
const Parameters GetDefaultParameters() const override;
```

Returns the full default parameter schema (see [§6](#6-full-json-parameters-schema)).
Called during construction to fill in any missing keys.

---

## 10. Registration and instantiation

### Registering the modeler

`OctreeHybridMesherModeler` is registered in `KratosApplication` via:

```cpp
// kratos/sources/kratos_application.cpp
KRATOS_REGISTER_MODELER("OctreeHybridMesherModeler", mOctreeHybridMesherModeler);
```

where `mOctreeHybridMesherModeler` is a `const OctreeHybridMesherModeler` data member of
`KratosApplication` (declared in `kratos/includes/kratos_application.h`).

`KRATOS_REGISTER_MODELER` calls `KratosComponents<Modeler>::Add`, which inserts the
prototype under the key `"OctreeHybridMesherModeler"` in the global component database.

### Instantiation from JSON

The standard way to instantiate a modeler from a JSON file in a Kratos simulation
script is through the modeler factory:

```python
import KratosMultiphysics as KM

model = KM.Model()
modeler = KM.CreateModeler(model, KM.Parameters("""{
    "modeler_name" : "OctreeHybridMesherModeler",
    "Parameters"   : {
        "input_model_part_name" : "MySurface",
        ...
    }
}"""))
modeler.SetupGeometryModel()
modeler.PrepareGeometryModel()
modeler.SetupModelPart()
```

Alternatively, construct it directly in Python:

```python
modeler = KM.OctreeHybridMesherModeler(model, settings)
```

The Python binding is registered in `kratos/python/add_modeler_to_python.cpp`:

```cpp
py::class_<OctreeHybridMesherModeler, OctreeHybridMesherModeler::Pointer, Modeler>(m, "OctreeHybridMesherModeler")
    .def(py::init<Model&, Parameters>())
;
```

### Registering custom components

To add a new coloring stage, entity-generation component, or operation without
modifying the core Kratos source tree, create a class that derives from
`OctreeHybridMesherColoring`, `OctreeHybridMesherEntityGeneration`, or `OctreeHybridMesherOperation`
respectively, add the two `KRATOS_REGISTRY_ADD_PROTOTYPE` macros in its `private`
section, and ensure its translation unit is compiled and linked.  The macros register
the prototype at static-initialisation time (before `main`), so no additional
registration call is needed.

**Coloring example skeleton:**

```cpp
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"

class MyCustomColoring : public Kratos::OctreeHybridMesherColoring {
public:
    MyCustomColoring() = default;
    MyCustomColoring(MyCustomColoring const&) {}

    void Apply(Kratos::OctreeHybridMesherModeler& rModeler,
               Kratos::Parameters ColoringParameters) const override
    {
        // Populate rModeler.GetData().mCellColor here.
    }

    const Kratos::Parameters GetDefaultParameters() const override {
        return Kratos::Parameters(R"({ "type" : "MyCustomColoring" })");
    }

private:
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics",
                                   Kratos::OctreeHybridMesherColoring, MyCustomColoring)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All",
                                   Kratos::OctreeHybridMesherColoring, MyCustomColoring)
};
```

---

## 11. Testing

### Python tests

The Python test suite lives at:

```
kratos/tests/test_octree_hybrid_mesher_modeler.py
```

It can be run directly:

```bash
PYTHONPATH=/path/to/build/Release python3 kratos/tests/test_octree_hybrid_mesher_modeler.py
```

or under the Kratos test runner.  The file contains nine test classes.

#### `TestOctreeHybridMesherModelerDual`

Tests for the dual (conforming) hex mesh path using a synthetic closed-box surface.

| Test | Assertion |
|------|-----------|
| `test_dual_mesh_elements_created` | `SetupModelPart` produces a non-empty hex ModelPart (elements > 0, nodes > 0). |
| `test_dual_mesh_zero_inverted` | All hexes have minimum scaled Jacobian > 0 (no inverted elements). |
| `test_dual_mesh_carve_bbox_inside_surface` | Output node bounding box lies within one half-cell margin of the input surface box (carve respected). |
| `test_dual_mesh_refinement_level_tagged` | At least one element has `REFINEMENT_LEVEL > 0` (regular dual hexes carry their octree level). |
| `test_boundary_conditions_created` | `OctreeHybridGenerateBoundaryConditionsByFace` creates a non-empty `Boundary` ModelPart alongside the volume. |
| `test_quality_report_operation` | `OctreeHybridReportMeshQuality` runs without error; the resulting ModelPart is non-empty. |

#### `TestOctreeHybridMesherModelerPrimal`

Tests for the primal (leaf-hex + hanging-node constraints) path.

| Test | Assertion |
|------|-----------|
| `test_primal_elements_created` | Primal mesh with `adaptive: true` produces elements and nodes. |
| `test_primal_constraints_count` | At least one hanging-node constraint is generated at 2:1 transitions. |
| `test_primal_constraint_row_sum` | Every constraint's relation matrix sums to 1.0 (partition of unity), verified to within `1e-10`. |
| `test_primal_constraint_master_counts` | Every constraint has exactly 2 masters (edge-midpoint) or 4 masters (face-centre). |

#### `TestOctreeHybridMesherModelerBunny`

Tests using the low-poly Stanford Bunny surface (`Bunny-LowPoly.stl`).  Automatically skipped when the STL is absent.

| Test | Assertion |
|------|-----------|
| `test_dual_bunny_zero_inverted` | Dual mesh at depth 4 produces a non-empty ModelPart with 0 inverted hexes. |
| `test_primal_bunny_constraints_row_sum` | Primal mesh of the bunny at depth 4: all hanging-node constraints satisfy partition of unity. |

#### `TestClassifyCellsInsideOutside`

Unit tests for the `OctreeHybridClassifyCellsInsideOutside` coloring component.

| Test | Assertion |
|------|-----------|
| `test_produces_inside_and_outside_colors` | Running the classifier reduces the element count vs. the unfiltered full block; the inside set is strictly smaller but non-empty. |
| `test_projected_shortcut_all_cells_inside` | With `project_to_surface: true` the short-circuit path fires: all surviving cells receive `color = 1`. |
| `test_default_type_name` | The Registry path `OctreeHybridMesherColoring.All.OctreeHybridClassifyCellsInsideOutside.Prototype` exists. |
| `test_unknown_coloring_type_raises` | A non-existent coloring type name triggers a clear error. |

#### `TestGenerateHexesByCellColor`

Unit tests for the `OctreeHybridGenerateHexesByCellColor` entity-generation component.

| Test | Assertion |
|------|-----------|
| `test_positive_element_count` | At least one element and one node are created. |
| `test_color_filter_inside` | `color = 1` produces fewer elements than omitting the coloring stage (the inside-only carve is smaller than the full block). |
| `test_zero_inverted_elements` | All hexes have positive scaled Jacobian. |
| `test_refinement_level_tagged` | With `tag_refinement_level: true` at least one element has `REFINEMENT_LEVEL > 0`. |
| `test_refinement_level_not_tagged` | With `tag_refinement_level: false` all elements show the default value `0`. |
| `test_node_deduplication` | Node count is strictly less than 8 × element count (nodes are shared across adjacent elements). |
| `test_node_ids_contiguous_from_one` | Node ids are unique and start from 1. |
| `test_element_ids_contiguous_from_one` | Element ids are unique and start from 1. |
| `test_registry_path_exists` | The Registry path for `OctreeHybridGenerateHexesByCellColor` exists. |
| `test_unknown_entity_type_raises` | A non-existent `generated_entity` type name triggers a clear error. |

#### `TestGenerateBoundaryConditionsByFace`

Unit tests for the `OctreeHybridGenerateBoundaryConditionsByFace` entity-generation component.

| Test | Assertion |
|------|-----------|
| `test_conditions_created` | At least one boundary condition is created. |
| `test_boundary_nodes_populated` | The boundary ModelPart contains at least one node. |
| `test_boundary_nodes_subset_of_volume_nodes` | Every boundary node id also exists in the volume mesh (nodes are shared, not duplicated). |
| `test_condition_ids_contiguous` | Condition ids are unique and start from 1. |
| `test_each_condition_has_four_nodes` | Every boundary condition has exactly 4 nodes (is a quad). |
| `test_boundary_faces_lt_6_times_elements` | Boundary face count is less than 6 × element count (interior faces are not counted). |
| `test_registry_path_exists` | The Registry path for `OctreeHybridGenerateBoundaryConditionsByFace` exists. |

#### `TestGenerateHangingNodeConstraints`

Unit tests for the `OctreeHybridGenerateHangingNodeConstraints` entity-generation component (primal mesh).

| Test | Assertion |
|------|-----------|
| `test_constraints_generated` | At least one hanging-node constraint is produced on the transition surface. |
| `test_partition_of_unity` | Every constraint row sums to 1.0. |
| `test_master_counts_two_or_four` | Only 2-master (edge-midpoint) and 4-master (face-centre) constraints exist. |
| `test_face_centre_constraints_present` | At least one 4-master (face-centre) constraint is present. |
| `test_multiple_variables` | Requesting 3 variables multiplies the constraint count by exactly 3. |
| `test_each_constraint_has_one_slave_dof` | Every constraint has exactly one slave DOF. |
| `test_registry_path_exists` | The Registry path for `OctreeHybridGenerateHangingNodeConstraints` exists. |
| `test_dual_mesh_no_hanging_constraints` | The dual mesh path produces zero hanging-node constraints (it is conforming by construction). |

#### `TestReportMeshQuality`

Unit tests for the `OctreeHybridReportMeshQuality` operation component.

| Test | Assertion |
|------|-----------|
| `test_runs_without_error` | The operation completes without raising an exception. |
| `test_zero_inverted_box` | A carved box mesh at depth 4 has zero inverted elements after the quality report runs. |
| `test_empty_model_part_does_not_crash` | Running the quality report on an empty ModelPart logs nothing and does not throw. |
| `test_registry_path_exists` | The Registry path for `OctreeHybridReportMeshQuality` exists. |

#### `TestRegistryDispatch`

Tests for the Registry-prototype dispatch mechanism inside `OctreeHybridMesherModeler`.

| Test | Assertion |
|------|-----------|
| `test_all_base_prototypes_registered` | All three abstract base-class prototypes (`OctreeHybridMesherColoring`, `OctreeHybridMesherEntityGeneration`, `OctreeHybridMesherOperation`) are in the Registry. |
| `test_all_concrete_prototypes_registered` | All five concrete components are in the Registry. |
| `test_full_path_dispatch_works` | A four-segment dot-separated full Registry path in the `"type"` field is accepted and dispatched correctly. |
| `test_unknown_operation_type_raises` | An unknown operation type name triggers a Registry-not-found error. |
| `test_base_type_invocation_raises` | Invoking the abstract base `OctreeHybridMesherOperation` prototype directly raises a clear error (the base does not implement the do-work virtual). |

---

### C++ tests

The C++ tests use the GTest framework and live at:

```
kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp
```

All tests are registered in `KratosCoreFastSuite`.  Run them with:

```bash
# From the build directory
./KratosCore.Tests --gtest_filter="*OctreeHybridMesher*"
# or via the VS Code "Run C++ Test Suite Filtered" task with the pattern OctreeHybridMesher
```

The C++ suite mirrors the Python suite, covering the same seven functional groups:

| Group | C++ test names |
|-------|---------------|
| Top-level modeler | `OctreeHybridMesherModelerDualElementsCreated`, `…DualZeroInverted`, `…DualCarveBbox`, `…DefaultParametersValid`, `…InfoString`, `…UnknownOperationThrows` |
| `OctreeHybridClassifyCellsInsideOutside` | `OctreeHybridMesherClassifyReducesCellCount`, `…ClassifyRegistered` |
| `OctreeHybridGenerateHexesByCellColor` | `OctreeHybridMesherGenerateHexesRegistered`, `…GenerateHexesNodeDeduplication`, `…GenerateHexesRefinementLevelTagged`, `…GenerateHexesNoLevelWhenDisabled`, `…GenerateHexesUniqueIds` |
| `OctreeHybridGenerateBoundaryConditionsByFace` | `OctreeHybridMesherBoundaryConditionsRegistered`, `…BoundaryConditionsCreated`, `…BoundaryConditionsQuadNodes`, `…BoundaryConditionsFewerthanSixTimesElements`, `…BoundaryNodesSubsetOfVolume` |
| `OctreeHybridGenerateHangingNodeConstraints` | `OctreeHybridMesherHangingNodeConstraintsRegistered`, `…PrimalMeshConstraintsGenerated`, `…PrimalConstraintsPartitionOfUnity`, `…PrimalConstraintsMasterCountsValid`, `…PrimalMultiVariableConstraints`, `…DualMeshNoHangingConstraints` |
| `OctreeHybridReportMeshQuality` | `OctreeHybridMesherReportMeshQualityRegistered`, `…ReportMeshQualityRunsWithoutError`, `…ReportMeshQualityEmptyModelPart` |
| Registry dispatch | `OctreeHybridMesherRegistryBasePrototypesPresent`, `…RegistryKratosMultiphysicsPaths`, `…RegistryFullPathDispatchWorks`, `…RegistryBaseColoringInvocationThrows`, `…RegistryBaseOperationInvocationThrows` |

---

### Example notebook

An interactive Jupyter notebook demonstrating the full modeler pipeline (with PyVista 3-D
visualisation) is provided at:

```
kratos/python_scripts/notebooks/octree_hybrid_mesher_modeler_example.ipynb
```

The notebook walks through:
- Loading a surface from a closed-box or STL geometry.
- Running the dual and primal mesh pipelines with annotated JSON settings blocks.
- Visualising the octree adaptive refinement (level scalar field) and the hex mesh quality.
- Comparing the uncarved block, the coloring-carved mesh, and the surface-projected mesh.
- Generating hanging-node constraints for the primal mesh and inspecting their partition-of-unity property.

It requires `KratosMultiphysics`, `pyvista`, and optionally `trame`/`ipywidgets` for interactive rendering.

---

## 12. Known limitations

1. **No surface projection for the primal mesh.** `project_to_surface: true` is
   silently ignored when `mesh_type: "primal"`.  The boundary of a primal mesh
   follows the octree grid and is not projected onto the input surface.  Use
   `mesh_type: "dual"` with `project_to_surface: true` for a surface-fitted boundary.

2. **Classification carve count differs slightly from the reference.** The
   `OctreeHybridClassifyCellsInsideOutside` pass uses a pseudo-random ray direction for the
   inside/outside parity test.  The ray sequence is not bit-reproducible across
   implementations (the retry logic regenerates the direction when a triangle is hit
   edge-on), so the carve keeps slightly fewer or more surface-band cells than the
   reference `RemoveOutsideElement`.  This is a known non-determinism in the
   inside/outside stage, not a defect.  The stage-5 surface projection re-meshes the
   surface band regardless.

3. **Non-manifold repair not fully ported.** The reference's
   `RemoveOutsideElement` includes a 146-probe non-manifold repair pass that removes
   degenerate boundary configurations after carving.  The Kratos port replaces this
   with the `ClearBufferZone` hemisphere-probe clearance.  The result produces a
   clean 2-manifold boundary on all tested models, but the hemisphere-probe algorithm
   is not the bit-for-bit equivalent of the reference's approach.

4. **`MAX_DEPTH = 10`.** The maximum octree refinement depth is fixed at 10 by
   `OctreeHybridKratosConfiguration`.  Meshes requiring finer resolution need this
   constant increased.

5. **Single-threaded pipeline.** All pipeline stages (octree construction, dual
   extraction, coloring, entity generation) are single-threaded.  The most expensive
   stages (face-adjacency probing, entity creation) are the main parallel opportunity.

6. **`project_to_surface` worst-element floor.** With the default budget
   (`projection_iterations: 20000`), `BuildCarveProjectAndWriteVtk` reproduces the
   reference's scaled-Jacobian distribution (median ~0.85, 0 inverted), but the
   minimum scaled Jacobian is ~0.3.  Reaching the reference's ~0.5+ floor requires a
   larger iteration budget (set `projection_iterations` to 300 000 for ~0.46
   minimum, still climbing).  See the `OctreeHybridMeshUtility` documentation,
   §13.2 for a detailed explanation.

7. **Coloring required before entity generation.** Even when `project_to_surface:
   true` (which already removes outside cells), `OctreeHybridClassifyCellsInsideOutside` must
   still appear in `coloring_settings_list` so that `mCellColor` is populated.
   `OctreeHybridGenerateHexesByCellColor` filters on `mCellColor` and will produce no elements
   if the color array is empty.

---

## See also

- [OctreeHybridMeshUtility](octree_hybrid_mesh_utility.md) — the underlying dual-hex
  meshing engine: transition templates, node-merge hash, surface projection details.
- [OctreeHybrid spatial container](../../Spatial_Containers/Trees_And_Searches/octree_hybrid.md)
- Reference paper:
  Tong, H., Halilaj, E., & Zhang, Y. J. (2024).
  **HybridOctree_Hex: Hybrid octree-based adaptive all-hexahedral mesh generation
  with Jacobian control.**
  *Journal of Computational Science*, 78, 102278.
  DOI: [10.1016/j.jocs.2024.102278](https://doi.org/10.1016/j.jocs.2024.102278)
