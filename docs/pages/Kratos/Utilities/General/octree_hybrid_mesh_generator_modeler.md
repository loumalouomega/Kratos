---
title: OctreeHybridMeshGeneratorModeler
keywords: mesh hex hexahedral tetrahedral octree adaptive dual primal modeler hanging-node constraints BCC Freudenthal
tags: [mesh hexahedral tetrahedral octree modeler]
sidebar: kratos_core_utilities
summary: Registry-driven modeler that wraps the OctreeHybridMeshUtility engine to produce all-hex or all-tet ModelParts with optional surface projection and hanging-node constraints.
---

# OctreeHybridMeshGeneratorModeler

## Table of contents

1. [What this modeler does](#1-what-this-modeler-does)
2. [Architecture: the Registry-prototype pattern](#2-architecture-the-registry-prototype-pattern)
3. [Pipeline stages](#3-pipeline-stages)
   - 3.1 [Octree generation and refinement — `refinement_settings_list`](#31-octree-generation-and-refinement--refinement_settings_list)
   - 3.2 [Coloring — `coloring_settings_list`](#32-coloring--coloring_settings_list)
   - 3.3 [Entity generation — `entities_generator_list`](#33-entity-generation--entities_generator_list)
   - 3.4 [Operations — `model_part_operations`](#34-operations--model_part_operations)
4. [Shared state: OctreeHybridMesherData](#4-shared-state-octreemesherdata)
5. [Mesh topologies](#5-mesh-topologies)
   - 5.1 [Dual mesh (default)](#51-dual-mesh-default)
   - 5.2 [Primal mesh with hanging-node constraints](#52-primal-mesh-with-hanging-node-constraints)
6. [Full JSON parameters schema](#6-full-json-parameters-schema)
7. [Registered components](#7-registered-components)
   - 7.0 [Refine operations](#70-refine-operations)
   - 7.1 [Coloring components](#71-coloring-components)
   - 7.2 [Entity-generation components](#72-entity-generation-components)
   - 7.3 [Operation components](#73-operation-components)
8. [Python / JSON usage examples](#8-python--json-usage-examples)
   - 8.1 [Dual carved mesh](#81-dual-carved-mesh)
   - 8.2 [Primal mesh with hanging-node constraints](#82-primal-mesh-with-hanging-node-constraints)
   - 8.3 [Dual mesh with surface projection](#83-dual-mesh-with-surface-projection)
   - 8.4 [Boundary conditions on the exterior surface](#84-boundary-conditions-on-the-exterior-surface)
   - 8.5 [Quality report](#85-quality-report)
   - 8.6 [Tetrahedral mesh (BCC Freudenthal decomposition)](#86-tetrahedral-mesh-bcc-freudenthal-decomposition)
9. [API reference](#9-api-reference)
10. [Registration and instantiation](#10-registration-and-instantiation)
11. [Testing](#11-testing)
12. [Known limitations](#12-known-limitations)

---

## 1. What this modeler does

`OctreeHybridMeshGeneratorModeler` is a Kratos `Modeler` subclass that converts a closed, orientable
triangular surface `ModelPart` into a volumetric ModelPart — either **all-hexahedral** or
**all-tetrahedral** — using the HybridOctree_Hex algorithm implemented in `OctreeHybridMeshUtility`.

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
`OctreeHybridMeshGeneratorModeler` argument passed to each call — specifically in the
`OctreeHybridMesherData` struct held by the modeler.  This makes the prototype objects
inherently thread-safe (they hold no data) and avoids any per-invocation allocation
of component objects.

---

## 3. Pipeline stages

### High-level order

```
SetupModelPart()
    │
    ├─ 1. BuildOctreeAndExtract()
    │      Dispatch<OctreeHybridRefineOperation>  [refinement_settings_list]
    │      │  first entry must be OctreeHybridRefineInterfaceCells:
    │      │    • build + 2:1-balance octree
    │      │    • store mesh_type / projection settings in OctreeHybridMesherData
    │      │  subsequent entries (optional): further adaptive / interface refinement
    │      then:
    │        • StrongConstrain2To1
    │        • extract dual or primal hex mesh
    │        • optionally carve + project to surface (dual only)
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
| 1 | Octree generation + refinement | `OctreeHybridRefineOperation` | `refinement_settings_list` | Build octree (via `OctreeHybridRefineInterfaceCells`), optional further refinement |
| 2 | Coloring | `OctreeHybridMesherColoring` | `coloring_settings_list` | Classify cells inside/outside |
| 3 | Entity generation | `OctreeHybridMesherEntityGeneration` | `entities_generator_list` | Create nodes, elements, conditions, constraints |
| 4 | Operations | `OctreeHybridMesherOperation` | `model_part_operations` | Post-processing (e.g. quality report) |

---

### 3.1 Octree generation and refinement — `refinement_settings_list`

`BuildOctreeAndExtract` dispatches the full `refinement_settings_list` before extracting the hex mesh.
The **first entry must be `OctreeHybridRefineInterfaceCells`** (which replaces the former `octree_generator` block);
subsequent entries are optional and can apply additional refinement (e.g. `OctreeHybridRefineUniform`,
`OctreeHybridRefineInterfaceCells`).

**Inputs:**

- The surface `ModelPart` identified by `input_model_part_name` inside the
  `OctreeHybridRefineInterfaceCells` entry (falls back to the top-level `input_model_part_name`).
  The model part must contain `Triangle3D3` geometries (typically loaded via `StlIO::ReadModelPart`).

**What it does:**

1. Dispatches the `refinement_settings_list`.  The first entry (`OctreeHybridRefineInterfaceCells`):
   - Calls `OctreeHybridMeshUtility::ExtractTriangleSoup` to copy the surface triangles
     into world-space for later carving/projection/classification.
   - Calls `OctreeHybridMeshUtility::BuildFromSurfaceMesh` to build the adaptive octree.
     When `adaptive: true` (the default), curvature and thickness criteria determine the
     refinement level per surface region, matching the reference HybridOctree_Hex octree
     cell-for-cell.  When `adaptive: false`, every cell intersecting the surface is uniformly
     refined to `refinement_depth`.
   - Stores `mesh_type`, `project_to_surface`, `projection_iterations`, and
     `projection_smoothing` into `OctreeHybridMesherData` for use by the extraction pass.
   - Any subsequent entries in `refinement_settings_list` (e.g. `OctreeHybridRefineUniform`)
     further subdivide octree cells.
2. Calls `StrongConstrain2To1` to enforce the 2:1 balance constraint across the whole tree.
3. Depending on `mesh_type`:

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

4. Initialises `OctreeHybridMesherData::mNodePtrs` (size = number of nodes, all null) for
   lazy de-duplication during entity generation.

**Key parameter decisions (set inside `OctreeHybridRefineInterfaceCells`):**

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

The coloring list is processed in order; multiple stages can be stacked to build
composite masks (e.g. classify inside/outside, then further mark interface cells or
level-specific regions).

The canonical colour convention is:

| Label | Meaning |
|-------|---------|
| `1` | Inside the input surface |
| `0` | Outside the input surface |
| other | User-defined sub-region |

Registered coloring components:

| JSON `"type"` | Purpose |
|--------------|---------|
| `OctreeHybridClassifyCellsInsideOutside` | Ray-cast inside/outside classification |
| `OctreeHybridColorCellsInTouch` | Color cells whose AABB intersects input-model-part geometry |
| `OctreeHybridColorConnectedCellsInTouch` | Flood-fill connected cells touching input geometry |
| `OctreeHybridColorCellsByLevel` | Color cells by octree refinement level |

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
| `GenerateHybridOctreeHexahedraElementsWithCellColor` | `GenerateHybridOctreeHexahedraElementsWithCellColor` | Create hex elements for cells matching a colour; optionally generates a `"constraint_type"` constraint for primal mesh hanging nodes when `"constraint_type"` and `"constrained_variables"` are non-empty |
| `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` | `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` | Create quad conditions on the outer surface; optionally generates a `"constraint_type"` constraint for primal mesh hanging nodes when `"constraint_type"` and `"constrained_variables"` are non-empty |
| `GenerateHybridOctreeTetrahedraElementsWithCellColor` | `GenerateHybridOctreeTetrahedraElementsWithCellColor` | Decompose each colour-matched hex into 6 tetrahedral elements using the Freudenthal–Kuhn scheme (BCC lattice, §3.3.3) |
| `GenerateHybridOctreeTriangularConditionsWithFaceColor` | `GenerateHybridOctreeTriangularConditionsWithFaceColor` | Create triangular conditions on the outer surface by splitting each boundary quad along the `(n₀, n₂)` diagonal (2 triangles per quad) |
| `GenerateOctreeHybridConstraints` | `GenerateOctreeHybridConstraints` | Create one master-slave constraint per (master node x constrained variable) for every primal-mesh 2:1 transition hanging node, optionally filtered by `"color"` and/or `"face_color"` |
| `GenerateOctreeHybridConstraintsBetweenColors` | `GenerateOctreeHybridConstraintsBetweenColors` | Tie geometrically coincident, non-conforming nodes between two colour blocks (`"color_block_a"` / `"color_block_b"`) with one master-slave constraint per (node pair x constrained variable) |

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
| `mHanging` | `vector<HangingConstraint>` | `BuildOctreeAndExtract` | `GenerateHybridOctreeHexahedraElementsWithCellColor` (when `"constraint_type"` and `"constrained_variables"` are non-empty), `GenerateOctreeHybridConstraints` | Hanging-node interpolation records (primal mesh only). |
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

Each `LinearMasterSlaveConstraint` uses the **1-1 form** (one master DOF, one slave DOF),
matching the format written and read by `ModelPartIO`.  The full bilinear interpolation:

```
u_slave = sum_m (w_m * u_master_m)
```

is recovered by the builder-and-solver when it accumulates all constraints sharing the
same slave DOF.  One constraint is created per **(hanging node × master node × DOF
variable)** triple:

| Hanging-node type | Constraints per (node × variable) | Weight per constraint |
|------------------|-----------------------------------|----------------------|
| Edge-midpoint | 2 | 0.5 |
| Face-centre | 4 | 0.25 |

The `"constrained_variables"` parameter of `GenerateHybridOctreeHexahedraElementsWithCellColor` lists which
DOF variables to constrain at 2:1 transitions (default: `[]` — no constraints).
Pass the desired scalar DOF names (e.g. `["DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"]`)
together with a non-empty `"constraint_type"` (e.g. `"LinearMasterSlaveConstraint"`)
to enable constraint generation in the same pass as element creation.

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

The complete parameter block accepted by `OctreeHybridMeshGeneratorModeler`:

```json
{
    "refinement_settings_list": [
        {
            "type"                 : "OctreeHybridRefineInterfaceCells",
            "input_model_part_name": "",
            "refinement_depth"     : 5,
            "adaptive"             : true,
            "mesh_type"            : "dual",
            "project_to_surface"   : false,
            "projection_iterations": 20000,
            "projection_smoothing" : 1000
        }
    ],
    "coloring_settings_list"  : [],
    "entities_generator_list" : [],
    "model_part_operations"   : [],
    "mdpa_file_name"          : "",
    "input_model_part_name"   : "",
    "default_outside_color"   : 1,
    "remove_orphan_nodes"     : true,
    "echo_level"              : 1
}
```

**Top-level keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `refinement_settings_list` | array | `[]` | Ordered list of refine-operation descriptors. The first entry must be `OctreeHybridRefineInterfaceCells`; optional further entries refine the octree before mesh extraction. |
| `coloring_settings_list` | array | `[]` | Ordered list of coloring stage descriptors. |
| `entities_generator_list` | array | `[]` | Ordered list of entity-generation stage descriptors. |
| `model_part_operations` | array | `[]` | Ordered list of post-processing operation descriptors. |
| `mdpa_file_name` | string | `""` | Optional path to an `.mdpa` file used to populate `input_model_part_name` during `Initialize`. If empty, the input ModelPart is assumed to be populated already. |
| `input_model_part_name` | string | `""` | Name of the surface ModelPart to mesh. Used as fallback when `OctreeHybridRefineInterfaceCells.input_model_part_name` is empty. |
| `default_outside_color` | int | `1` | Reserved for future use (mirrors `VoxelMeshGeneratorModeler`); not yet consumed by the coloring pipeline. |
| `remove_orphan_nodes` | bool | `true` | Reserved: intended to remove nodes not belonging to any generated element, condition, or constraint after the entity-generation and operation stages. Not yet implemented. |
| `echo_level` | int | `1` | Verbosity level (0 = silent, higher = more output). |

**`OctreeHybridRefineInterfaceCells` keys** (first entry of `refinement_settings_list`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridRefineInterfaceCells"` | Registry lookup key. |
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

### 7.0 Refine operations

#### `OctreeHybridRefineInterfaceCells`

On the **first call** (no octree built yet): extracts a triangle soup from the surface,
builds the adaptive or uniform octree via `BuildFromSurfaceMesh`, and stores `mesh_type`
and projection settings for later use by `BuildOctreeAndExtract`.  **Must appear as the
first entry** of `refinement_settings_list`.

On **subsequent calls**: selectively subdivides octree cells near the interface surface up
to `refinement_depth`.  Multiple entries may target different model parts for
feature-specific resolution.

**Registry path:** `OctreeHybridRefineOperation.All.OctreeHybridRefineInterfaceCells.Prototype`

**Class:** `Kratos::OctreeHybridRefineInterfaceCells`

**Header:** `kratos/modeler/refine_operations/refine_interface_cells_hybrid_octree.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridRefineInterfaceCells"` | Registry lookup key. |
| `input_model_part_name` | string | `""` | Surface ModelPart name. First call: empty → falls back to modeler's top-level `input_model_part_name`. Subsequent calls: empty → reuse main triangle soup. |
| `refinement_depth` | int | `5` | Build depth (first call) or maximum refinement depth for interface cells (subsequent calls). |
| `refined_cell_size` | double | `0.0` | Subsequent calls only: desired cell size (world-space). When > 0, overrides `refinement_depth` via `ElementSizeToDepth`. |
| `adaptive` | bool | `true` | First call only: `true` = adaptive near-surface refinement; `false` = uniform to `refinement_depth`. |
| `mesh_type` | string | `"dual"` | First call only: `"dual"` = conforming all-hex dual mesh; `"primal"` = one hex per leaf with hanging-node records. |
| `project_to_surface` | bool | `false` | First call only. Dual mesh: when `true`, the built mesh is projected onto the iso-surface. |
| `projection_iterations` | int | `20000` | First call only: optimiser iteration budget for the projection pass. |
| `projection_smoothing` | int | `1000` | First call only: smart-Laplacian smoothing interval during projection. |

**Behaviour (first call — octree build):**

1. Extracts the triangle soup from the named surface `ModelPart`.
2. Builds the octree via `OctreeHybridMeshUtility::BuildFromSurfaceMesh`.
3. Writes `mesh_type`, `project_to_surface`, `projection_iterations`, and
   `projection_smoothing` into `OctreeHybridMesherData`.

**Behaviour (subsequent calls — interface refinement):**

1. Resolves the triangle soup (named model part or cached `mTriangles`).
2. Calls `OctreeHybridMeshUtility::RefineInterfaceCells` to subdivide cells near the surface.

**Example JSON (minimal dual mesh):**

```json
{
    "type"             : "OctreeHybridRefineInterfaceCells",
    "refinement_depth" : 5,
    "adaptive"         : true,
    "mesh_type"        : "dual"
}
```

**Example JSON (primal mesh):**

```json
{
    "type"             : "OctreeHybridRefineInterfaceCells",
    "refinement_depth" : 4,
    "adaptive"         : true,
    "mesh_type"        : "primal"
}
```

**Example JSON (dual with surface projection):**

```json
{
    "type"                 : "OctreeHybridRefineInterfaceCells",
    "refinement_depth"     : 5,
    "adaptive"             : true,
    "mesh_type"            : "dual",
    "project_to_surface"   : true,
    "projection_iterations": 20000,
    "projection_smoothing" : 1000
}
```

---

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

#### `OctreeHybridColorCellsInTouch`

Colors every hex cell whose axis-aligned bounding box (AABB) intersects any geometry
from the specified ModelPart.

**Registry path:** `OctreeHybridMesherColoring.All.OctreeHybridColorCellsInTouch.Prototype`

**Class:** `Kratos::OctreeHybridColorCellsInTouch`

**Header:** `kratos/modeler/coloring/octree_hybrid_color_cells_in_touch.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridColorCellsInTouch"` | Registry lookup key. |
| `model_part_name` | string | `""` | Name of the ModelPart whose geometry is tested. |
| `color` | int | `1` | Label to write for cells in touch with the geometry. |
| `input_entities` | string | `"geometries"` | Which entities to iterate: `"geometries"`, `"elements"`, or `"conditions"`. |

**Behaviour:**

For each geometry in the ModelPart the operation computes the geometry's AABB, uses
it to quick-reject cells whose own AABB does not overlap, then calls
`Geometry::HasIntersection(cell_min, cell_max)` for the remaining candidates.  Cells
that pass are assigned the configured `color`.  If `mCellColor` has not been
initialised it is resized and filled with `0` first.

**Example JSON:**

```json
{
    "type"            : "OctreeHybridColorCellsInTouch",
    "model_part_name" : "MySurface",
    "color"           : 2,
    "input_entities"  : "geometries"
}
```

---

#### `OctreeHybridColorConnectedCellsInTouch`

Flood-fills all hex cells that are face-adjacent to cells touching the input geometry
and carry a specified seed colour.

**Registry path:** `OctreeHybridMesherColoring.All.OctreeHybridColorConnectedCellsInTouch.Prototype`

**Class:** `Kratos::OctreeHybridColorConnectedCellsInTouch`

**Header:** `kratos/modeler/coloring/octree_hybrid_color_connected_cells_in_touch.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridColorConnectedCellsInTouch"` | Registry lookup key. |
| `model_part_name` | string | `""` | Name of the ModelPart whose geometry seeds the flood-fill. |
| `color` | int | `1` | Label to write to every reached cell. |
| `cell_color` | int | `0` | Only cells currently carrying this label are traversed. |
| `input_entities` | string | `"geometries"` | Which entities to iterate: `"geometries"`, `"elements"`, or `"conditions"`. |

**Behaviour:**

1. Builds a face-adjacency graph from `mCells` (two cells are adjacent when they
   share a sorted 4-tuple of global node indices).
2. Seeds: all cells with colour `cell_color` that pass the AABB + `HasIntersection`
   test against any geometry in the ModelPart.
3. BFS from seeds through neighbours with `cell_color`; each visited cell is
   assigned `color`.

A typical use-case is labelling the connected exterior region after
`OctreeHybridClassifyCellsInsideOutside`: run with `cell_color=0` to mark the
outer shell as a distinct colour (e.g. 2), separating it from interior voids that
also received label 0.

**Example JSON:**

```json
{
    "type"            : "OctreeHybridColorConnectedCellsInTouch",
    "model_part_name" : "MySurface",
    "color"           : 2,
    "cell_color"      : 0,
    "input_entities"  : "geometries"
}
```

---

#### `OctreeHybridColorCellsByLevel`

Colors cells whose octree refinement level falls within a specified inclusive range.

**Registry path:** `OctreeHybridMesherColoring.All.OctreeHybridColorCellsByLevel.Prototype`

**Class:** `Kratos::OctreeHybridColorCellsByLevel`

**Header:** `kratos/modeler/coloring/octree_hybrid_color_cells_by_level.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"OctreeHybridColorCellsByLevel"` | Registry lookup key. |
| `color` | int | `1` | Label to write for cells in the specified level range. |
| `min_level` | int | `1` | Minimum octree level (inclusive). Use `-1` to include transition-template hexes; `-2` for buffer-layer hexes. |
| `max_level` | int | `100` | Maximum octree level (inclusive). |

**Behaviour:**

Iterates `mCellLevel` and writes `color` to every entry where
`min_level <= mCellLevel[i] <= max_level`.  Level conventions match
`OctreeHybridMesherData::mCellLevel`:

| Value | Meaning |
|-------|---------|
| 1 … N | Octree leaf level |
| -1 | Transition-template hex (dual mesh only) |
| -2 | Buffer-layer hex (projection path only) |

**Example JSON:**

```json
{
    "type"      : "OctreeHybridColorCellsByLevel",
    "color"     : 2,
    "min_level" : 4,
    "max_level" : 4
}
```

---

### 7.2 Entity-generation components

#### `GenerateHybridOctreeHexahedraElementsWithCellColor`

Creates one 8-noded hexahedral element per cell whose colour matches the configured
value.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeHexahedraElementsWithCellColor.Prototype`

**Class:** `Kratos::GenerateHybridOctreeHexahedraElementsWithCellColor`

**Header:** `kratos/modeler/entity_generation/generate_hybrid_octree_hexahedra_elements_with_cell_color.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateHybridOctreeHexahedraElementsWithCellColor"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label to emit (e.g. `1` for inside cells). |
| `properties_id` | int | `1` | Properties block ID assigned to every new element (created on demand). |
| `generated_entity` | string | `"Element3D8N"` | Registered element type name (`KratosComponents<Element>::Get`). |
| `tag_refinement_level` | bool | `true` | When `true`, stores the cell's octree refinement level in the element's `REFINEMENT_LEVEL` variable. |
| `constraint_type` | string | `""` | Registered constraint type for hanging-node MPC (e.g. `"LinearMasterSlaveConstraint"`). Empty (default) = no constraints generated. |
| `constrained_variables` | string array | `[]` | Scalar DOF variable names to constrain at 2:1 transitions. Used only when `"constraint_type"` is non-empty. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto (continue from existing model). |
| `initial_element_id` | int | `0` | Explicit first element ID; `0` = auto. |
| `initial_constraint_id` | int | `0` | Explicit first constraint ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Behaviour:**

Iterates `mCells` in order.  For each cell `c` where `mCellColor[c] == color`:
1. Resolves the 8 corner mesh-node indices.
2. For each corner, calls `OctreeHybridMeshGeneratorModeler::GenerateOrRetrieveNode` — creates a
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
    "type"                : "GenerateHybridOctreeHexahedraElementsWithCellColor",
    "model_part_name"     : "FluidDomain",
    "color"               : 1,
    "properties_id"       : 1,
    "generated_entity"    : "Element3D8N",
    "tag_refinement_level": true
}
```

---

#### `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor`

Creates quadrilateral boundary conditions on the outer surface of the coloured hex
mesh.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeQuadrilateralConditionsWithFaceColor.Prototype`

**Class:** `Kratos::GenerateHybridOctreeQuadrilateralConditionsWithFaceColor`

**Header:** `kratos/modeler/entity_generation/generate_hybrid_octree_quadrilateral_conditions_with_face_color.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateHybridOctreeQuadrilateralConditionsWithFaceColor"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label identifying the volume to extract the boundary of. |
| `properties_id` | int | `1` | Properties block ID assigned to every new condition. |
| `generated_entity` | string | `"SurfaceCondition3D4N"` | Registered condition type name. |
| `constraint_type` | string | `""` | Registered constraint type for hanging-node MPC (e.g. `"LinearMasterSlaveConstraint"`). Empty (default) = no constraints generated. |
| `constrained_variables` | string array | `[]` | Scalar DOF variable names to constrain at 2:1 transitions. Used only when `"constraint_type"` is non-empty. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto. |
| `initial_condition_id` | int | `0` | Explicit first condition ID; `0` = auto. |
| `initial_constraint_id` | int | `0` | Explicit first constraint ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Behaviour:**

1. Filters `mCells` to retain only cells with the requested `color`.
2. Calls `OctreeHybridMeshUtility::ExtractBoundaryFaces` on the filtered cell list
   to identify outer boundary quads (faces owned by exactly one hex).
3. For each boundary quad, creates the four corner nodes via
   `GenerateOrRetrieveNode` (reuses nodes already created by a prior hex-generation
   stage) and constructs a condition using `generated_entity`.
4. Bulk-inserts all nodes and conditions into the target ModelPart.
5. If `"constraint_type"` and `"constrained_variables"` are non-empty and `mHanging` is
   non-empty, generates `"constraint_type"` objects for all 2:1 hanging-node transitions
   (same logic as `GenerateHybridOctreeHexahedraElementsWithCellColor`).

Winding convention: outward normals follow the convention of `ExtractBoundaryFaces`.

**Example JSON (primal mesh with hanging-node constraints on the boundary):**

```json
{
    "type"                  : "GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
    "model_part_name"       : "Boundary",
    "color"                 : 1,
    "properties_id"         : 1,
    "generated_entity"      : "SurfaceCondition3D4N",
    "constraint_type"       : "LinearMasterSlaveConstraint",
    "constrained_variables" : ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"]
}
```

---

#### `GenerateHybridOctreeTetrahedraElementsWithCellColor`

Decomposes each colour-matched hex cell into **6 tetrahedra** using the Freudenthal–Kuhn
decomposition along the main diagonal (local nodes 0 → 6).  This implements the BCC-lattice
tetrahedral pattern described in TACS1de1.pdf §3.3.3: dual-hex cell centres act as BCC body
positions, yielding tetrahedra with a minimum dihedral angle of **45°**.

The resulting tet mesh is **conforming**: `OctreeHybridMeshUtility::ExtractDualHexMesh` assigns
local node indices via the fixed `idTransform` map, so adjacent hexes always share the same
physical node at the same logical corner — the shared-face diagonal is therefore identical in
both hexes, and no hanging nodes are produced.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeTetrahedraElementsWithCellColor.Prototype`

**Class:** `Kratos::GenerateHybridOctreeTetrahedraElementsWithCellColor`

**Header:** `kratos/modeler/entity_generation/generate_hybrid_octree_tetrahedra_elements_with_cell_color.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateHybridOctreeTetrahedraElementsWithCellColor"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label to decompose (e.g. `1` for inside cells). |
| `properties_id` | int | `1` | Properties block ID assigned to every new element. |
| `generated_entity` | string | `"Element3D4N"` | Registered tetrahedral element type name. |
| `tag_refinement_level` | bool | `true` | When `true`, each element carries the parent hex's `REFINEMENT_LEVEL` value. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto. |
| `initial_element_id` | int | `0` | Explicit first element ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Freudenthal 6-tet connectivity** (local hex node indices, Hexahedra3D8 ordering):

| Tet | Nodes |
|-----|-------|
| 0 | 0, 3, 6, 2 |
| 1 | 3, 6, 7, 0 |
| 2 | 4, 7, 6, 0 |
| 3 | 0, 4, 5, 6 |
| 4 | 0, 1, 2, 6 |
| 5 | 1, 5, 6, 0 |

**Example JSON:**

```json
{
    "type"                 : "GenerateHybridOctreeTetrahedraElementsWithCellColor",
    "model_part_name"      : "TetDomain",
    "color"                : 1,
    "properties_id"        : 1,
    "generated_entity"     : "Element3D4N",
    "tag_refinement_level" : true
}
```

---

#### `GenerateHybridOctreeTriangularConditionsWithFaceColor`

Creates **triangular boundary conditions** on the outer surface of the coloured hex mesh.
Each boundary quad `{n₀, n₁, n₂, n₃}` from `OctreeHybridMeshUtility::ExtractBoundaryFaces`
is split along the `(n₀, n₂)` diagonal into two triangles:
- triangle 1: `{n₀, n₁, n₂}`
- triangle 2: `{n₀, n₂, n₃}`

This diagonal is consistent with the `(0, 6)` main-diagonal Freudenthal decomposition used by
`GenerateHybridOctreeTetrahedraElementsWithCellColor`, so every boundary triangle is an exposed face of an
interior tetrahedron.  A tet mesh + triangle BC mesh pair generated from the same `entities_generator_list`
is therefore **conforming** at the boundary.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateHybridOctreeTriangularConditionsWithFaceColor.Prototype`

**Class:** `Kratos::GenerateHybridOctreeTriangularConditionsWithFaceColor`

**Header:** `kratos/modeler/entity_generation/generate_hybrid_octree_triangular_conditions_with_face_color.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateHybridOctreeTriangularConditionsWithFaceColor"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `1` | Cell-colour label identifying the volume to extract the boundary of. |
| `properties_id` | int | `1` | Properties block ID assigned to every new condition. |
| `generated_entity` | string | `"SurfaceCondition3D3N"` | Registered triangular condition type name. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto. |
| `initial_condition_id` | int | `0` | Explicit first condition ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Example JSON:**

```json
{
    "type"             : "GenerateHybridOctreeTriangularConditionsWithFaceColor",
    "model_part_name"  : "TetDomain.Boundary",
    "color"            : 1,
    "properties_id"    : 1,
    "generated_entity" : "SurfaceCondition3D3N"
}
```

---

#### `GenerateOctreeHybridConstraints`

Standalone entity-generation stage that creates the **2:1 transition hanging-node**
master-slave constraints of the **primal** mesh. It extracts the hanging-node-constraint
logic that the per-entity generators (e.g.
`GenerateHybridOctreeHexahedraElementsWithCellColor`) can optionally run inline, into its
own pipeline entry, so that constraints can be generated independently of (and after) the
element/condition generation stages.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateOctreeHybridConstraints.Prototype`

**Class:** `Kratos::GenerateOctreeHybridConstraints`

**Header:** `kratos/modeler/entity_generation/generate_octree_hybrid_constraints.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateOctreeHybridConstraints"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `-1` | Restrict to hanging slaves that are corners of cells with this colour (`-1` = no filter). |
| `face_color` | int | `-1` | Restrict to hanging slaves on the boundary of the cells with this colour, per `OctreeHybridMeshUtility::ExtractBoundaryFaces` (`-1` = no filter). |
| `generated_entity` | string | `"LinearMasterSlaveConstraint"` | Registered MasterSlaveConstraint type name. |
| `constrained_variables` | string array | `["TEMPERATURE"]` | Scalar DOF variable names to constrain at 2:1 transitions. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto. |
| `initial_constraint_id` | int | `0` | Explicit first constraint ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Behaviour:**

1. For every record `hc` in `mHanging` (populated only for the primal mesh):
   - If `color != -1` and `mCellColor` is non-empty, `hc.SlaveNode` must be a corner of
     at least one cell coloured `color`; otherwise the record is skipped.
   - If `face_color != -1`, `hc.SlaveNode` must additionally lie on a boundary face
     (per `ExtractBoundaryFaces`) of the cells coloured `face_color` (or of all cells if
     `mCellColor` is empty); otherwise the record is skipped.
2. `hc.SlaveNode` and every entry of `hc.MasterNodes` must already have a non-null
   `mNodePtrs` entry (i.e. created by a prior entity-generation stage); otherwise the
   record is skipped silently.
3. For each variable in `constrained_variables` and each master `m`, registers the DOF
   on both the slave and master `m` nodes (`Node::AddDof`, no-op if already present) and
   creates one `generated_entity` constraint via `ModelPart::CreateNewMasterSlaveConstraint`
   with master `m` as master, the slave node as slave, weight `hc.Weights[m]` and
   constant `0.0`.
4. If `mHanging` is empty (dual mesh, or no 2:1 transitions) or `constrained_variables`
   is empty, no constraints are generated.

**Example JSON:**

```json
{
    "type"                  : "GenerateOctreeHybridConstraints",
    "model_part_name"       : "StructureDomain",
    "color"                 : 1,
    "face_color"            : -1,
    "generated_entity"      : "LinearMasterSlaveConstraint",
    "constrained_variables" : ["TEMPERATURE"]
}
```

---

#### `GenerateOctreeHybridConstraintsBetweenColors`

Entity-generation stage that ties together two **non-conforming** colour blocks of the
hybrid octree mesh through master-slave constraints.

`GenerateOrRetrieveNode` de-duplicates nodes through the shared `mNodePtrs` map keyed by
mesh-node index, so two cells of different colours that share a mesh-node index already
share the same `Node` — there is nothing to tie in that (conforming) case. This stage
instead targets pairs of mesh-node indices that belong to different colour blocks
(`color_block_a` / `color_block_b`), have **different** mesh-node indices, but are
**geometrically coincident** (same `mNodes` coordinates within a fixed tolerance). For
every such pair, one `generated_entity` constraint per `constrained_variables` entry is
created, with the `color_block_a` node as master and the `color_block_b` node as slave
(weight `1.0`, constant `0.0`, i.e. `u_slave = u_master`).

Mesh-node indices shared by both blocks (the conforming case) and indices belonging to
only one of the two blocks are ignored — they produce zero constraints. With the current
dual/primal extraction (which always shares mesh-node indices at conforming interfaces),
this stage is a no-op safeguard intended for future non-conforming meshes.

**Registry path:** `OctreeHybridMesherEntityGeneration.All.GenerateOctreeHybridConstraintsBetweenColors.Prototype`

**Class:** `Kratos::GenerateOctreeHybridConstraintsBetweenColors`

**Header:** `kratos/modeler/entity_generation/generate_octree_hybrid_constraints_between_colors.h`

**Parameter schema:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | string | `"GenerateOctreeHybridConstraintsBetweenColors"` | Registry lookup key. |
| `model_part_name` | string | `"Undefined"` | Name of the target ModelPart (created if absent). |
| `color` | int | `-1` | Restrict matching to nodes that are also corners of cells with this colour (`-1` = no filter). |
| `color_block_a` | int | `-1` | Colour of the master-side block. |
| `color_block_b` | int | `-1` | Colour of the slave-side block. |
| `generated_entity` | string | `""` | Registered MasterSlaveConstraint type name (**required**, no default). |
| `constrained_variables` | string array | `[]` | Scalar DOF variable names to constrain across the interface. |
| `initial_node_id` | int | `0` | Explicit first node ID; `0` = auto. |
| `initial_constraint_id` | int | `0` | Explicit first constraint ID; `0` = auto. |
| `echo_level` | int | `0` | Verbosity of `KRATOS_INFO` logging (`0` = silent). |

**Behaviour:**

1. **Membership sets** — for every cell `c`, the indices of its 8 corner nodes are added
   to `nodes_a` if `mCellColor[c] == color_block_a`, and to `nodes_b` if
   `mCellColor[c] == color_block_b`. If `color != -1`, a third set `nodes_region`
   collects corners of cells coloured `color`.
2. **Candidate filtering** — `nodes_a_only = nodes_a \ nodes_b` and
   `nodes_b_only = nodes_b \ nodes_a` (indices in both sets are already shared `Node`s
   and are dropped). When `color != -1`, both sets are further intersected with
   `nodes_region`.
3. **Geometric matching** — `nodes_a_only` and `nodes_b_only` indices are bucketed by
   their (rounded) `mNodes` coordinates. For every coincident pair `(ia, ib)`, the
   corresponding nodes are retrieved (or created) via `GenerateOrRetrieveNode`.
4. **Constraint creation** — for each variable in `constrained_variables`, registers the
   DOF on both nodes (`Node::AddDof`) and creates one `generated_entity` constraint with
   `ia`'s node as master, `ib`'s node as slave, weight `1.0` and constant `0.0`.
5. **Finalisation** — any newly created nodes are bulk-inserted into the target
   ModelPart via `ModelPartUtils::AddNodesFromOrderedContainer`.

If `mCellColor` is empty (no colouring stage has run) or `constrained_variables` is
empty, no constraints are generated. `generated_entity` has no default and **must** be
set explicitly; an empty value raises an error.

**Example JSON:**

```json
{
    "type"                  : "GenerateOctreeHybridConstraintsBetweenColors",
    "model_part_name"       : "Interface",
    "color_block_a"         : 1,
    "color_block_b"         : 2,
    "generated_entity"      : "LinearMasterSlaveConstraint",
    "constrained_variables" : ["TEMPERATURE"]
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
    "refinement_settings_list" : [
        {
            "type"             : "OctreeHybridRefineInterfaceCells",
            "refinement_depth" : 5,
            "adaptive"         : true,
            "mesh_type"        : "dual"
        }
    ],
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"             : "GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name"  : "Volume",
            "color"            : 1,
            "properties_id"    : 1,
            "generated_entity" : "Element3D8N",
            "tag_refinement_level" : true
        }
    ],
    "model_part_operations" : []
}""")

modeler = KM.OctreeHybridMeshGeneratorModeler(model, settings)
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
    "refinement_settings_list" : [
        {
            "type"             : "OctreeHybridRefineInterfaceCells",
            "refinement_depth" : 4,
            "adaptive"         : true,
            "mesh_type"        : "primal"
        }
    ],
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"                  : "GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name"       : "Domain",
            "color"                 : 1,
            "properties_id"         : 1,
            "constraint_type"       : "LinearMasterSlaveConstraint",
            "constrained_variables" : ["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"]
        }
    ],
    "model_part_operations" : [
        {
            "type"            : "OctreeHybridReportMeshQuality",
            "model_part_name" : "Domain"
        }
    ]
}""")

modeler = KM.OctreeHybridMeshGeneratorModeler(model, settings)
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
    "refinement_settings_list" : [
        {
            "type"                 : "OctreeHybridRefineInterfaceCells",
            "refinement_depth"     : 5,
            "adaptive"             : true,
            "mesh_type"            : "dual",
            "project_to_surface"   : true,
            "projection_iterations": 20000,
            "projection_smoothing" : 1000
        }
    ],
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"             : "GenerateHybridOctreeHexahedraElementsWithCellColor",
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

modeler = KM.OctreeHybridMeshGeneratorModeler(model, settings)
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

Combine `GenerateHybridOctreeHexahedraElementsWithCellColor` and `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` to populate
both a volume ModelPart and a boundary ModelPart in a single `SetupModelPart` call.

```python
settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "refinement_settings_list" : [
        {
            "type"             : "OctreeHybridRefineInterfaceCells",
            "refinement_depth" : 4,
            "adaptive"         : true,
            "mesh_type"        : "dual"
        }
    ],
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"            : "GenerateHybridOctreeHexahedraElementsWithCellColor",
            "model_part_name" : "Volume",
            "color"           : 1,
            "properties_id"   : 1
        },
        {
            "type"             : "GenerateHybridOctreeQuadrilateralConditionsWithFaceColor",
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
nodes; nodes created first by `GenerateHybridOctreeHexahedraElementsWithCellColor` are reused (not duplicated)
by `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` through the `mNodePtrs` de-duplication cache.

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

### 8.6 Tetrahedral mesh (BCC Freudenthal decomposition)

`GenerateHybridOctreeTetrahedraElementsWithCellColor` decomposes each hex cell into 6 tetrahedra.
Pair it with `GenerateHybridOctreeTriangularConditionsWithFaceColor` to obtain a conforming
tet + triangle-BC mesh in a single `SetupModelPart` call.

The tet count is always exactly **6 × hex count** for the same settings, and the
triangle count is exactly **2 × quad-BC count**.

```python
import KratosMultiphysics as KM

model = KM.Model()
surface_mp = model.CreateModelPart("Surface")
surface_mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
KM.StlIO("my_surface.stl", KM.Parameters('{"open_mode":"read"}')).ReadModelPart(surface_mp)

settings = KM.Parameters("""{
    "input_model_part_name"  : "Surface",
    "refinement_settings_list" : [
        {
            "type"             : "OctreeHybridRefineInterfaceCells",
            "refinement_depth" : 4,
            "adaptive"         : false,
            "mesh_type"        : "dual"
        }
    ],
    "coloring_settings_list" : [
        { "type" : "OctreeHybridClassifyCellsInsideOutside" }
    ],
    "entities_generator_list" : [
        {
            "type"                 : "GenerateHybridOctreeTetrahedraElementsWithCellColor",
            "model_part_name"      : "TetDomain",
            "color"                : 1,
            "properties_id"        : 1,
            "generated_entity"     : "Element3D4N",
            "tag_refinement_level" : true
        },
        {
            "type"             : "GenerateHybridOctreeTriangularConditionsWithFaceColor",
            "model_part_name"  : "TetDomain.Boundary",
            "color"            : 1,
            "properties_id"    : 1,
            "generated_entity" : "SurfaceCondition3D3N"
        }
    ],
    "model_part_operations" : []
}""")

modeler = KM.OctreeHybridMeshGeneratorModeler(model, settings)
modeler.SetupModelPart()

tet_mp   = model.GetModelPart("TetDomain")
bound_mp = model.GetModelPart("TetDomain.Boundary")
print(f"Tet elements : {tet_mp.NumberOfElements()}")
print(f"Tri BCs      : {bound_mp.NumberOfConditions()}")
```

> **Note:** `"TetDomain.Boundary"` is a sub-model-part of `"TetDomain"`.  Node IDs are
> shared: every boundary triangle node also appears in the volume mesh.

---

## 9. API reference

### `OctreeHybridMeshGeneratorModeler` — public interface

```cpp
#include "modeler/octree_hybrid_mesh_generator_modeler.h"
```

---

#### Constructors

```cpp
OctreeHybridMeshGeneratorModeler();
OctreeHybridMeshGeneratorModeler(Model& rModel, Parameters ModelerParameters = Parameters());
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

`OctreeHybridMeshGeneratorModeler` is registered in `KratosApplication` via:

```cpp
// kratos/sources/kratos_application.cpp
KRATOS_REGISTER_MODELER("OctreeHybridMeshGeneratorModeler", mOctreeHybridMeshGeneratorModeler);
```

where `mOctreeHybridMeshGeneratorModeler` is a `const OctreeHybridMeshGeneratorModeler` data member of
`KratosApplication` (declared in `kratos/includes/kratos_application.h`).

`KRATOS_REGISTER_MODELER` calls `KratosComponents<Modeler>::Add`, which inserts the
prototype under the key `"OctreeHybridMeshGeneratorModeler"` in the global component database.

### Instantiation from JSON

The standard way to instantiate a modeler from a JSON file in a Kratos simulation
script is through the modeler factory:

```python
import KratosMultiphysics as KM

model = KM.Model()
modeler = KM.CreateModeler(model, KM.Parameters("""{
    "modeler_name" : "OctreeHybridMeshGeneratorModeler",
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
modeler = KM.OctreeHybridMeshGeneratorModeler(model, settings)
```

The Python binding is registered in `kratos/python/add_modeler_to_python.cpp`:

```cpp
py::class_<OctreeHybridMeshGeneratorModeler, OctreeHybridMeshGeneratorModeler::Pointer, Modeler>(m, "OctreeHybridMeshGeneratorModeler")
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

    void Apply(Kratos::OctreeHybridMeshGeneratorModeler& rModeler,
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
kratos/tests/test_octree_hybrid_mesh_generator_modeler.py
```

It can be run directly:

```bash
PYTHONPATH=/path/to/build/Release python3 kratos/tests/test_octree_hybrid_mesh_generator_modeler.py
```

or under the Kratos test runner.  The file contains twelve test classes.

#### `TestOctreeHybridMeshGeneratorModelerDual`

Tests for the dual (conforming) hex mesh path using a synthetic closed-box surface.

| Test | Assertion |
|------|-----------|
| `test_dual_mesh_elements_created` | `SetupModelPart` produces a non-empty hex ModelPart (elements > 0, nodes > 0). |
| `test_dual_mesh_zero_inverted` | All hexes have minimum scaled Jacobian > 0 (no inverted elements). |
| `test_dual_mesh_carve_bbox_inside_surface` | Output node bounding box lies within one half-cell margin of the input surface box (carve respected). |
| `test_dual_mesh_refinement_level_tagged` | At least one element has `REFINEMENT_LEVEL > 0` (regular dual hexes carry their octree level). |
| `test_boundary_conditions_created` | `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` creates a non-empty `Boundary` ModelPart alongside the volume. |
| `test_quality_report_operation` | `OctreeHybridReportMeshQuality` runs without error; the resulting ModelPart is non-empty. |

#### `TestOctreeHybridMeshGeneratorModelerPrimal`

Tests for the primal (leaf-hex + hanging-node constraints) path.

| Test | Assertion |
|------|-----------|
| `test_primal_elements_created` | Primal mesh with `adaptive: true` produces elements and nodes. |
| `test_primal_constraints_count` | At least one hanging-node constraint is generated at 2:1 transitions. |
| `test_primal_constraint_row_sum` | Every constraint is 1×1 (one master DOF per constraint). |
| `test_primal_constraint_master_counts` | Every constraint has exactly 1 master DOF (1-1 form). |

#### `TestOctreeHybridMeshGeneratorModelerBunny`

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

Unit tests for the `GenerateHybridOctreeHexahedraElementsWithCellColor` entity-generation component.

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
| `test_registry_path_exists` | The Registry path for `GenerateHybridOctreeHexahedraElementsWithCellColor` exists. |
| `test_unknown_entity_type_raises` | A non-existent `generated_entity` type name triggers a clear error. |

#### `TestGenerateBoundaryConditionsByFace`

Unit tests for the `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` entity-generation component.

| Test | Assertion |
|------|-----------|
| `test_conditions_created` | At least one boundary condition is created. |
| `test_boundary_nodes_populated` | The boundary ModelPart contains at least one node. |
| `test_boundary_nodes_subset_of_volume_nodes` | Every boundary node id also exists in the volume mesh (nodes are shared, not duplicated). |
| `test_condition_ids_contiguous` | Condition ids are unique and start from 1. |
| `test_each_condition_has_four_nodes` | Every boundary condition has exactly 4 nodes (is a quad). |
| `test_boundary_faces_lt_6_times_elements` | Boundary face count is less than 6 × element count (interior faces are not counted). |
| `test_registry_path_exists` | The Registry path for `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` exists. |
| `test_primal_boundary_constraints_generated` | With `"constraint_type"` and `"constrained_variables"` set on a primal mesh, constraints are created in the boundary ModelPart. |

#### `TestGenerateHangingNodeConstraints`

Tests for hanging-node constraint generation via the `"constraint_type"` and `"constrained_variables"` parameters of
`GenerateHybridOctreeHexahedraElementsWithCellColor` (primal mesh).

| Test | Assertion |
|------|-----------|
| `test_constraints_generated` | At least one hanging-node constraint is produced on the transition surface. |
| `test_partition_of_unity` | Every constraint is 1×1 (one master DOF per constraint). |
| `test_master_counts_two_or_four` | Every constraint has exactly 1 master DOF (1-1 form). |
| `test_face_centre_constraints_present` | At least one constraint has weight ≈ 0.25 (from a face-centre hanging node). |
| `test_multiple_variables` | Requesting 3 variables multiplies the constraint count by exactly 3. |
| `test_each_constraint_has_one_slave_dof` | Every constraint has exactly one slave DOF. |
| `test_empty_variables_produces_no_constraints` | An empty `"constrained_variables"` list generates zero constraints (default). |
| `test_dual_mesh_no_hanging_constraints` | The dual mesh path produces zero hanging-node constraints (it is conforming by construction). |

#### `TestOctreeHybridColorCellsInTouch`

Unit tests for the `OctreeHybridColorCellsInTouch` coloring component.

| Test | Assertion |
|------|-----------|
| `test_cells_in_touch_produces_some_colored_cells` | After running without prior inside/outside classification, at least some cells touching the box surface are colored. |
| `test_cells_in_touch_not_all_cells_colored` | Interior cells not touching any surface triangle remain uncolored (in-touch count < total cell count). |
| `test_cells_in_touch_elements_entities` | `input_entities="elements"` is accepted and colors cells touching the element geometries. |
| `test_registry_path_exists` | The Registry path `OctreeHybridMesherColoring.All.OctreeHybridColorCellsInTouch.Prototype` exists. |

#### `TestOctreeHybridColorConnectedCellsInTouch`

Unit tests for the `OctreeHybridColorConnectedCellsInTouch` coloring component.

| Test | Assertion |
|------|-----------|
| `test_connected_flood_fill_from_surface` | After classify + flood-fill through outside cells (cell_color=0), the flooded set is non-empty. |
| `test_connected_flood_fill_fewer_than_all_outside` | Flood-filled cell count is ≤ total outside count (no over-coloring). |
| `test_connected_flood_fill_inside_cells` | Flood-fill through inside cells (cell_color=1) from the surface gives a non-empty result. |
| `test_registry_path_exists` | The Registry path `OctreeHybridMesherColoring.All.OctreeHybridColorConnectedCellsInTouch.Prototype` exists. |

#### `TestOctreeHybridColorCellsByLevel`

Unit tests for the `OctreeHybridColorCellsByLevel` coloring component.

| Test | Assertion |
|------|-----------|
| `test_color_by_target_level_finds_cells` | Coloring cells at the refinement depth (level=4 for depth=4) finds at least one cell. |
| `test_color_beyond_max_depth_finds_nothing` | Requesting level 5 on a depth-4 mesh returns zero cells. |
| `test_wide_range_covers_all_positive_levels` | min_level=1, max_level=100 in a uniform non-adaptive mesh colors all cells. |
| `test_template_hexes_captured_by_negative_level` | min_level=-1, max_level=-1 in an adaptive mesh captures only transition-template hexes (if any). |
| `test_registry_path_exists` | The Registry path `OctreeHybridMesherColoring.All.OctreeHybridColorCellsByLevel.Prototype` exists. |

#### `TestReportMeshQuality`

Unit tests for the `OctreeHybridReportMeshQuality` operation component.

| Test | Assertion |
|------|-----------|
| `test_runs_without_error` | The operation completes without raising an exception. |
| `test_zero_inverted_box` | A carved box mesh at depth 4 has zero inverted elements after the quality report runs. |
| `test_empty_model_part_does_not_crash` | Running the quality report on an empty ModelPart logs nothing and does not throw. |
| `test_registry_path_exists` | The Registry path for `OctreeHybridReportMeshQuality` exists. |

#### `TestRegistryDispatch`

Tests for the Registry-prototype dispatch mechanism inside `OctreeHybridMeshGeneratorModeler`.

| Test | Assertion |
|------|-----------|
| `test_all_base_prototypes_registered` | All four abstract base-class prototypes (`OctreeHybridRefineOperation`, `OctreeHybridMesherColoring`, `OctreeHybridMesherEntityGeneration`, `OctreeHybridMesherOperation`) are in the Registry. |
| `test_all_concrete_prototypes_registered` | All nine concrete components are in the Registry (including all five coloring stages). |
| `test_full_path_dispatch_works` | A four-segment dot-separated full Registry path in the `"type"` field is accepted and dispatched correctly. |
| `test_unknown_operation_type_raises` | An unknown operation type name triggers a Registry-not-found error. |
| `test_base_type_invocation_raises` | Invoking the abstract base `OctreeHybridMesherOperation` prototype directly raises a clear error (the base does not implement the do-work virtual). |

---

### C++ tests

The C++ tests use the GTest framework and live at:

```
kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesh_generator_modeler.cpp
```

All tests are registered in `KratosCoreFastSuite`.  Run them with:

```bash
# From the build directory
./KratosCore.Tests --gtest_filter="*OctreeHybridMesher*"
# or via the VS Code "Run C++ Test Suite Filtered" task with the pattern OctreeHybridMesher
```

The C++ suite mirrors the Python suite, covering the same functional groups plus new tet/triangle tests:

| Group | C++ test names |
|-------|---------------|
| Top-level modeler | `OctreeHybridMeshGeneratorModelerDualElementsCreated`, `…DualZeroInverted`, `…DualCarveBbox`, `…DefaultParametersValid`, `…InfoString`, `…UnknownOperationThrows` |
| `OctreeHybridClassifyCellsInsideOutside` | `OctreeHybridMesherClassifyReducesCellCount`, `…ClassifyRegistered` |
| `GenerateHybridOctreeHexahedraElementsWithCellColor` | `OctreeHybridMesherGenerateHexesRegistered`, `…GenerateHexesNodeDeduplication`, `…GenerateHexesRefinementLevelTagged`, `…GenerateHexesNoLevelWhenDisabled`, `…GenerateHexesUniqueIds` |
| `GenerateHybridOctreeQuadrilateralConditionsWithFaceColor` | `OctreeHybridMesherBoundaryConditionsRegistered`, `…BoundaryConditionsCreated`, `…BoundaryConditionsQuadNodes`, `…BoundaryConditionsFewerthanSixTimesElements`, `…BoundaryNodesSubsetOfVolume` |
| `GenerateHybridOctreeHexahedraElementsWithCellColor` (hanging-node path) | `OctreeHybridMesherGenerateHexesByCellColorHasConstraintParams`, `…PrimalMeshConstraintsGenerated`, `…PrimalConstraintsPartitionOfUnity`, `…PrimalConstraintsMasterCountsValid`, `…PrimalMultiVariableConstraints`, `…DualMeshNoHangingConstraints` |
| `OctreeHybridReportMeshQuality` | `OctreeHybridMesherReportMeshQualityRegistered`, `…ReportMeshQualityRunsWithoutError`, `…ReportMeshQualityEmptyModelPart` |
| Registry dispatch | `OctreeHybridMesherRegistryBasePrototypesPresent`, `…RegistryKratosMultiphysicsPaths`, `…RegistryFullPathDispatchWorks`, `…RegistryBaseColoringInvocationThrows`, `…RegistryBaseOperationInvocationThrows` |
| `GenerateHybridOctreeTetrahedraElementsWithCellColor` | `OctreeHybridGenerateTetrahedraRegistryEntry`, `OctreeHybridMeshGeneratorModelerTetraElementsCreated`, `…TetraCountIsHexTimes6`, `…TetraZeroInverted`, `…TetraNodeSubsetFromHex`, `…TetraTagRefinementLevel` |
| `GenerateHybridOctreeTriangularConditionsWithFaceColor` | `OctreeHybridGenerateTriangleBCsRegistryEntry`, `OctreeHybridMeshGeneratorModelerTriangleBCsCreated`, `…TriangleBCsCountIsTwiceQuad`, `…TriangleBCsNodeSubset` |

---

### Example notebook

An interactive Jupyter notebook demonstrating the full modeler pipeline (with PyVista 3-D
visualisation) is provided at:

```
kratos/python_scripts/notebooks/octree_hybrid_mesh_generator_modeler_example.ipynb
```

The notebook walks through:
- Loading a surface from a closed-box or STL geometry.
- Running the dual and primal mesh pipelines with annotated JSON settings blocks.
- Visualising the octree adaptive refinement (level scalar field) and the hex mesh quality.
- Comparing the uncarved block, the coloring-carved mesh, and the surface-projected mesh.
- Generating hanging-node constraints for the primal mesh and inspecting their partition-of-unity property.
- **Step 11 (new):** Generating a conforming tetrahedral mesh via `GenerateHybridOctreeTetrahedraElementsWithCellColor` and triangular boundary conditions via `GenerateHybridOctreeTriangularConditionsWithFaceColor`; verifying the 6× tet/hex ratio and zero inverted elements; PyVista visualisation of the tet mesh coloured by refinement level.

It requires `KratosMultiphysics`, `pyvista`, and optionally `trame`/`ipywidgets` for interactive rendering.

---

## 12. Known limitations

1. **No surface projection for the primal mesh.** `project_to_surface: true` in the
   `OctreeHybridRefineInterfaceCells` entry is silently ignored when `mesh_type: "primal"`.
   The boundary of a primal mesh follows the octree grid and is not projected onto the
   input surface.  Use `mesh_type: "dual"` with `project_to_surface: true` for a
   surface-fitted boundary.

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

7. **Coloring required before entity generation.** Even when `project_to_surface: true`
   is set in `OctreeHybridRefineInterfaceCells` (which already removes outside cells),
   `OctreeHybridClassifyCellsInsideOutside` must still appear in `coloring_settings_list`
   so that `mCellColor` is populated.
   `GenerateHybridOctreeHexahedraElementsWithCellColor` filters on `mCellColor` and will produce no elements
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
