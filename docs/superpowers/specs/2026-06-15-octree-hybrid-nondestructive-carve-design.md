# Non-destructive carve/clear/project pipeline for `OctreeHybridMeshGeneratorModeler`

## 1. Problem statement

When `RefineInterfaceCellsOctreeHybrid` is run with `"mesh_type": "dual"` and
`"project_to_surface": true`, the modeler's `ApplyRefinement` step:

1. Extracts the full dual hex mesh (`ExtractDualHexMesh` → `r_data.mCells`,
   `r_data.mNodes`, `r_data.mCellLevel`).
2. Calls `RemoveOutsideElement`, which **removes** every cell whose centroid is
   classified "outside" the input triangle soup directly from
   `r_data.mCells`/`r_data.mCellLevel`.
3. Calls `ClearBufferZone`, which **removes** additional boundary cells that
   would create non-manifold/folded geometry.
4. Calls `ProjectToIsoSurface`, which deforms the remaining ("core") cells'
   boundary onto the input surface and adds a new buffer shell of cells
   (tagged `mCellLevel == -2`).

After this, `ApplyColoring` runs against whatever is left in `mCells`. Because
steps 2–3 already discarded everything outside the projection surface, any
coloring stage that targets a region *outside* that surface (e.g. a mold
region surrounding a liquid cavity, as in the reported bug) has no candidate
cells left — the resulting sub-model-part is empty.

The user's request: until entities are generated, carving/clearing/projection
should **never delete cells from `r_data.mCells`**. Instead, each stage should
record *classification metadata* per cell. Coloring stages — driven entirely
by their own `"color"` parameter, as today — then decide which subset of cells
(by classification and/or any other existing per-cell metadata) is relevant
for entity generation.

## 2. New data field: `mCarveStatus`

`Internals::OctreeHybridMesherData` gains:

```cpp
std::vector<int> mCarveStatus;  // size == mCells.size() once populated
```

Values, by analogy with `mCellLevel` (which is also plain metadata, not a
color):

| Value | Meaning |
|-------|---------|
| `0` | "outside" — not part of the projected core, not part of the buffer shell. Geometrically untouched (raw octree dual cell). |
| `1` | "core" — inside the projection surface, boundary deformed by `ProjectToIsoSurface` (or untouched if not on the projected boundary). |
| `2` | "buffer shell" — new cell created by `ProjectToIsoSurface` to pad the projected boundary. Always has `mCellLevel == -2` (existing convention, unchanged). |

`mCarveStatus` is populated only when `mProjectToSurface && !mTriangles.empty()`
(i.e. only on the projected path). On the non-projected path it remains empty,
exactly like other projection-only fields (`mProjected` stays `false`).

`mCarveStatus.size() == mCells.size()` always holds once populated — including
after `ProjectToIsoSurface` appends buffer-shell cells, since that function is
the one populating the array.

## 3. New `ApplyRefinement` flow (dual mesh, projected path)

Replaces the current 3-call sequence (`octree_hybrid_mesh_generator_modeler.cpp:625-688`):

```cpp
if (r_data.mMeshType == "dual") {
    OctreeHybridMeshUtility::ExtractDualHexMesh(
        *r_data.mpOctree, r_data.mNodes, r_data.mCells, r_data.mCellLevel);

    if (r_data.mProjectToSurface && !r_data.mTriangles.empty()) {
        // 1. Classify every cell as inside/outside — non-destructive,
        //    writes into a temporary "core" flag vector, mCells untouched.
        std::vector<int> core_flag;  // 1 = inside (core candidate), 0 = outside
        OctreeHybridMeshUtility::ClassifyInsideOutside(
            r_data.mTriangles, r_data.mNodes, r_data.mCells, core_flag);

        // Build the index list of "core" cells from core_flag == 1.
        std::vector<IndexType> core_cell_indices = /* indices where core_flag[i]==1 */;

        // 2. Non-destructive ClearBufferZone: demotes folded/non-manifold
        //    cells out of core_cell_indices (does not touch mCells).
        OctreeHybridMeshUtility::ClearBufferZone(
            r_data.mNodes, r_data.mCells, r_data.mCellLevel,
            core_cell_indices, mEchoLevel);

        // 3. Non-destructive ProjectToIsoSurface: deforms the boundary of
        //    core_cell_indices, appends a buffer shell to mCells/mCellLevel
        //    (mCellLevel == -2 for new cells, as today), and produces
        //    mCarveStatus for ALL cells (0/1/2).
        OctreeHybridMeshUtility::ProjectToIsoSurface(
            r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellLevel,
            core_cell_indices, r_data.mCarveStatus,
            r_data.mProjectionIterations, r_data.mProjectionSmoothing, mEchoLevel);

        r_data.mProjected = true;
    }
}
```

`RemoveOutsideElement` is **removed from this pipeline** (it remains available
as a free function for `BuildCarveAndWriteVtk`/`BuildCarveProjectAndWriteVtk`
and the existing destructive-signature tests — see §8).

## 4. Non-destructive `ClearBufferZone` overload

New overload signature (existing destructive overload kept verbatim):

```cpp
// Existing (unchanged) — destructive, used by demo/VTK paths and their tests:
static void ClearBufferZone(
    std::vector<array_1d<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel,
    int EchoLevel = 0);

// New — non-destructive:
static void ClearBufferZone(
    const std::vector<array_1d<double,3>>& rNodes,
    const std::vector<std::array<int,8>>& rCells,
    const std::vector<int>& rCellLevel,
    std::vector<IndexType>& rCoreCellIndices,  // in/out: shrunk in place
    int EchoLevel = 0);
```

Behavior: identical fold/non-manifold detection logic (the existing
128-direction Fibonacci-sphere probe over `ExtractBoundaryFaces`), but instead
of erasing entries from `rCells`/`rCellLevel`, cells identified as
"folded"/non-manifold are **removed from `rCoreCellIndices`** (i.e. demoted
back to "outside" — they will get `mCarveStatus == 0`). `MaxRounds` iteration
behavior is preserved. `rNodes`/`rCells`/`rCellLevel` are `const` in this
overload — nothing is mutated.

## 5. Non-destructive `ProjectToIsoSurface` overload

New overload signature (existing destructive overload kept verbatim):

```cpp
// Existing (unchanged) — destructive, used by demo/VTK paths and their tests.

// New — non-destructive:
static void ProjectToIsoSurface(
    const std::vector<std::array<double,3>>& rTriangles,
    std::vector<array_1d<double,3>>& rNodes,
    std::vector<std::array<int,8>>& rCells,
    std::vector<int>& rCellLevel,
    const std::vector<IndexType>& rCoreCellIndices,
    std::vector<int>& rCarveStatus,   // out: sized to rCells.size() after this call
    int ProjectionIterations,
    int ProjectionSmoothing,
    int EchoLevel = 0);
```

Behavior follows the existing 5-stage algorithm with these adjustments:

1. **Normalize to 100-unit box** — unchanged, operates on `rNodes` (all
   nodes, since boundary faces of the core set may share nodes with outside
   cells — see §6).
2. **Extract boundary quad faces** — computed from `ExtractBoundaryFaces`
   restricted to `rCoreCellIndices` (faces owned by exactly one core cell, or
   shared between a core cell and a non-core cell).
3. **Buffer shell construction** — unchanged: duplicates boundary nodes,
   builds new buffer hexes, **appended to `rCells`/`rCellLevel`** (new cells
   get `rCellLevel == -2`, as today). This is the *only* place `rCells`
   grows; it never shrinks.
4. **Gradient/Laplacian-smoothing optimization** — unchanged, operates on the
   core + buffer node set as before.
5. **Undo normalization** — unchanged.

Additionally, at the end:

```cpp
rCarveStatus.assign(rCells.size(), 0);
for (auto idx : rCoreCellIndices) rCarveStatus[idx] = 1;
for (auto idx : buffer_cell_indices /* newly appended */) rCarveStatus[idx] = 2;
```

All cells not in `rCoreCellIndices` and not part of the new buffer shell keep
`rCarveStatus == 0` and are geometrically untouched by this function.

## 6. Shared nodes at the core/outside interface (conforming projection)

Per the user's choice ("Let projection drag the shared nodes — conforming
interface"): nodes on the boundary between a core cell and a directly
adjacent **non-core (outside)** cell are **not duplicated**. When stage 4
moves such a shared node to fit the projected surface, the adjacent outside
cell's geometry moves with it.

Consequence: outside cells immediately adjacent to the projected core can
become non-convex/low-quality (dragged). This is accepted — see §7 — because
duplicating those nodes would re-introduce a manifold seam between "kept" and
"discarded" regions, which contradicts "keep the whole mesh as one mesh."

Buffer-shell cells (`mCarveStatus == 2`) *do* get duplicated boundary nodes
(unchanged from today), since they are new cells with no pre-existing outside
neighbor sharing those nodes.

## 7. Diagnostics for dragged outside cells (v1: diagnostics only)

After `ProjectToIsoSurface` returns, `ApplyRefinement` computes
`ScaledJacobianMin`/`JacobianMin` (existing public static helpers, unchanged)
for every cell with `mCarveStatus == 0` that has at least one node shared with
a `mCarveStatus != 0` cell. If any such cell has a non-positive scaled
Jacobian, log a single `KRATOS_INFO` summary (count of affected cells, worst
value) at `mEchoLevel >= 1`. **No mesh changes** result from this check in v1
— it exists purely so users can detect and react to (e.g. via further
refinement) badly dragged outside cells. No optimizer/algorithm changes.

## 8. `RemoveOutsideElement` — retained, not removed

`OctreeHybridMeshUtility::RemoveOutsideElement` keeps its existing destructive
signature and implementation **unchanged**, for:

- `BuildCarveAndWriteVtk` / `BuildCarveProjectAndWriteVtk` (demo/VTK paths,
  `--carve`/`--project` CLI flags).
- Existing C++ tests in `test_octree_hybrid_mesh_utility.cpp` Section H
  (lines 838-877) that call it directly.

It is simply no longer called from `ApplyRefinement`.

## 9. New coloring stage: `OctreeHybridColorCellsByCarveStatus`

New files `kratos/modeler/coloring/octree_hybrid_color_cells_by_carve_status.h`
/ `.cpp`, mirroring `OctreeHybridColorCellsByLevel` exactly:

```cpp
const Parameters OctreeHybridColorCellsByCarveStatus::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"       : "OctreeHybridColorCellsByCarveStatus",
        "color"      : 1,
        "min_status" : 0,
        "max_status" : 2
    })");
}

void OctreeHybridColorCellsByCarveStatus::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler, Parameters ColoringParameters) const
{
    auto& r_data = rModeler.GetData();
    const std::size_t n_cells = r_data.mCells.size();
    if (r_data.mCellColor.size() != n_cells) r_data.mCellColor.assign(n_cells, 0);

    KRATOS_ERROR_IF(r_data.mCarveStatus.size() != n_cells)
        << "OctreeHybridColorCellsByCarveStatus requires mCarveStatus to be "
           "populated (project_to_surface: true)." << std::endl;

    const int color = ColoringParameters["color"].GetInt();
    const int min_status = ColoringParameters["min_status"].GetInt();
    const int max_status = ColoringParameters["max_status"].GetInt();
    for (std::size_t i = 0; i < n_cells; ++i) {
        const int status = r_data.mCarveStatus[i];
        if (status >= min_status && status <= max_status) r_data.mCellColor[i] = color;
    }
}
```

Registered in `<app>_application.cpp` / the coloring registry the same way as
`OctreeHybridColorCellsByLevel`. Added to the CMake glob (already covered by
`custom_*`/`modeler/coloring` `file(GLOB_RECURSE ...)` pattern — verify and
add explicitly if not).

## 10. `OctreeHybridClassifyCellsInsideOutside` update

Current behavior (`octree_hybrid_classify_cells_inside_outside.cpp:36-56`):
when `mProjected == true`, it short-circuits with `mCellColor.assign(n, 1)`
because under the old destructive pipeline every surviving cell was inside by
construction.

New behavior:

```cpp
void OctreeHybridClassifyCellsInsideOutside::Apply(
    OctreeHybridMeshGeneratorModeler& rModeler, Parameters) const
{
    auto& r_data = rModeler.GetData();

    if (r_data.mProjected) {
        // mCarveStatus already distinguishes inside (core/buffer, 1/2) from
        // outside (0) without re-running the ray-caster.
        const std::size_t n_cells = r_data.mCells.size();
        r_data.mCellColor.assign(n_cells, 0);
        for (std::size_t i = 0; i < n_cells; ++i) {
            if (r_data.mCarveStatus[i] != 0) r_data.mCellColor[i] = 1;
        }
        return;
    }

    OctreeHybridMeshUtility::ClassifyInsideOutside(
        r_data.mTriangles, r_data.mNodes, r_data.mCells, r_data.mCellColor);
}
```

This preserves the "1 = inside" contract of this stage and the "skip the
ray-caster on the projected path" optimization (buffer-shell centroids near
the surface are unreliable for ray-casting, which was the original reason for
the shortcut), while being correct now that `mCells` includes outside cells.

## 11. Testing plan

### 11.1 C++ unit tests — `test_octree_hybrid_mesh_utility.cpp`

New test cases for the non-destructive overloads:

- `ClearBufferZoneNonDestructiveDoesNotResizeCells` — `rCells`/`rCellLevel`
  sizes unchanged before/after; `rCoreCellIndices` shrinks by exactly the
  count of folded cells (cross-check against the destructive overload on the
  same input: `original_count - destructive_result_count ==
  core_indices_before - core_indices_after`).
- `ProjectToIsoSurfaceNonDestructiveProducesCarveStatus` — after the call,
  `mCarveStatus.size() == mCells.size()`; every value is in `{0,1,2}`; status
  `1` count equals `rCoreCellIndices.size()` (input); status `2` cells all
  have `mCellLevel == -2`; status `0` cells have coordinates identical to
  pre-call values (geometrically untouched), except for shared-node dragging
  at the core/outside interface (§6) — verify only *non-adjacent* outside
  cells are untouched.
- `ProjectToIsoSurfaceNonDestructiveTotalCellCountGrows` —
  `rCells.size()_after >= rCells.size()_before` (only grows, by the buffer
  shell count).

Existing Section H tests (destructive `RemoveOutsideElement`,
lines 838-877) are left as-is.

### 11.2 New coloring stage tests

New file `test_octree_hybrid_color_cells_by_carve_status.cpp`, mirroring the
existing `OctreeHybridColorCellsByLevel` test structure: construct a small
`OctreeHybridMesherData` with a hand-built `mCarveStatus` vector (e.g.
`{0,1,1,2}`), run `Apply` with `color=5, min_status=1, max_status=2`, assert
`mCellColor == {0,5,5,5}`. Also a test for the `KRATOS_ERROR_IF` when
`mCarveStatus` is empty/mismatched.

### 11.3 `OctreeHybridClassifyCellsInsideOutside` tests
(`test_octree_hybrid_mesher_modeler.py`)

- Rewrite `test_projected_shortcut_all_cells_inside` (line 365): with
  `"project_to_surface": true`, assert:
  - `model.GetModelPart("All").NumberOfElements()` (no coloring filter,
    `color: 0` or omitted, covering the full `mCells`) equals the *uncarved*
    block element count (same as a `project_to_surface: false` run at the
    same refinement settings would produce as its starting point — i.e.
    `mCells` was not shrunk).
  - `OctreeHybridClassifyCellsInsideOutside` then run, and
    `model.GetModelPart("Proj")` (color `1`) is strictly smaller than `"All"`
    but non-empty (core+buffer only).
  - Rename to `test_projected_carve_status_shortcut_partitions_cells` and
    update its docstring.

### 11.4 New regression test reproducing the reported bug

New Python test in `TestOctreeHybridMeshGeneratorModelerDual` (or a new test
class), modeled on the user's `mesh.py`:

- Build a small closed-box "liquid" surface fully contained inside a larger
  bounding box (representing the "mold").
- Run with `mesh_type: "dual"`, `project_to_surface: true`,
  `adaptive: false`, small `refinement_depth` for speed.
- Coloring list includes `OctreeHybridColorCellsWithInsideCenter` /
  `OctreeHybridColorCellsInTouch` targeting a region geometrically *outside*
  the liquid surface (analogous to `mold_106`).
- Assert the resulting "mold" sub-model-part is **non-empty** — this is the
  direct regression check for the originally-reported empty-model-part bug.

### 11.5 C++ modeler test update —
`OctreeHybridMeshGeneratorModelerBunnyProjectedDualMeshIsConnected`
(`test_octree_hybrid_mesher_modeler.cpp:2509+`)

Currently uses `"coloring_settings_list": []` and
`GenerateOctreeHybridHexahedraElementsWithCellColor` with `color: 1`, relying
on the old blanket `mCellColor.assign(n,1)` shortcut.

Update: add to `coloring_settings_list`:

```json
{ "type": "OctreeHybridColorCellsByCarveStatus", "color": 1, "min_status": 1, "max_status": 2 }
```

This selects exactly the core+buffer (projected) cells, preserving the
original intent of the "single connected component, no islands" check
(`CountConnectedElementGroups(out) == 1`), now scoped correctly to the
projected region rather than relying on the old assign-all shortcut.

## 12. Documentation updates

### `docs/pages/Kratos/Utilities/General/octree_hybrid_mesh_generator_modeler.md`

- §"ApplyRefinement" description (~line 223-228): replace the
  `RemoveOutsideElement` → `ClearBufferZone` → `ProjectToIsoSurface`
  (destructive) description with the new non-destructive flow from §3,
  introducing `mCarveStatus`.
- `mesh_type`/`adaptive`/`project_to_surface` results table (~245-253): add a
  row clarifying that with `project_to_surface: true`, `mCells` now contains
  the *entire* uncarved block plus the buffer shell, classified via
  `mCarveStatus`; coloring stages decide what's exported.
- §8.3 example note (~1739) and the `OctreeHybridClassifyCellsInsideOutside`
  behavior section (~620-645): update per §10 above — describe the
  `mCarveStatus`-based shortcut instead of `assign(n,1)`.
- New coloring-stage table entry (~line 280-290 area) for
  `OctreeHybridColorCellsByCarveStatus`, with its own subsection (parameter
  schema + example), following the `OctreeHybridColorCellsByLevel` template.
- Test table (~2324): update the `test_projected_shortcut_all_cells_inside`
  row per §11.3's rename, and add a row for the new regression test (§11.4).
- Limitation #7 (~2528): rewrite — coloring is *still* required before entity
  generation, but now because `mCellColor` defaults to all-`0`/empty (not
  because outside cells were removed); clarify that with
  `project_to_surface: true`, `OctreeHybridColorCellsByCarveStatus` is the
  recommended way to select the projected region, and arbitrary other
  coloring stages can now also target the previously-discarded outside cells.

### `docs/pages/Kratos/Utilities/General/octree_hybrid_mesh_utility.md`

- §13.1 (`RemoveOutsideElement`) and §13.2 (`ProjectToIsoSurface`): add a note
  that the modeler's `ApplyRefinement` no longer calls the destructive
  `RemoveOutsideElement`/`ClearBufferZone`/`ProjectToIsoSurface` signatures
  directly — it uses `ClassifyInsideOutside` plus the new non-destructive
  `ClearBufferZone`/`ProjectToIsoSurface` overloads (§4, §5). The destructive
  signatures remain available and are used unchanged by
  `BuildCarveAndWriteVtk`/`BuildCarveProjectAndWriteVtk` (`--carve`/`--project`
  demo flags).
- No change needed to the demo CLI flag descriptions themselves (§ around
  line 1054-1069) — behavior there is unchanged.

## 13. Example notebook / demo script

Check `kratos/python_scripts/notebooks/octree_hybrid_mesh_generator_modeler_example.ipynb`
and `kratos/tests/demo_octree_hybrid_mesh.py` for any cell that runs
`project_to_surface: true` with an empty or `OctreeHybridClassifyCellsInsideOutside`-only
`coloring_settings_list` and relies on the old "all cells become color 1"
behavior for its output. If found, update to either:
- add `OctreeHybridColorCellsByCarveStatus` (`min_status: 1, max_status: 2`) to
  select the projected region, or
- keep `OctreeHybridClassifyCellsInsideOutside` (still works per §10, same
  output set as before for this case).

If no such cell exists (the demo CLI's `--project` flag uses the unchanged
destructive `BuildCarveProjectAndWriteVtk` path per §8), no changes are
needed — confirm this during implementation.

## 14. Backward compatibility summary

| Item | Status |
|------|--------|
| `RemoveOutsideElement` (destructive) | Unchanged, still used by `BuildCarveAndWriteVtk`/`BuildCarveProjectAndWriteVtk` and their tests. |
| `ClearBufferZone` (destructive overload) | Unchanged, still used by `BuildCarveProjectAndWriteVtk` and its tests. |
| `ProjectToIsoSurface` (destructive overload) | Unchanged, still used by `BuildCarveProjectAndWriteVtk` and its tests. |
| `ClassifyInsideOutside` | Unchanged signature, now also used inside `ApplyRefinement`. |
| `ExtractBoundaryFaces` | Unchanged, reused by new overloads. |
| `OctreeHybridClassifyCellsInsideOutside` coloring stage | Behavior change on the `mProjected==true` path (§10); non-projected path unchanged. |
| `project_to_surface: true` configs without `OctreeHybridColorCellsByCarveStatus`/`OctreeHybridClassifyCellsInsideOutside` in `coloring_settings_list` | `mCellColor` defaults to all-`0`/unset (as on the non-projected path today) — entity generators filtering on `color != 0` produce nothing. This is the existing "coloring required" contract (limitation #7), now applying uniformly to both paths. |
| Existing `mesh.py`-style configs (the reported bug) | Coloring stages targeting regions outside the projection surface now find candidate cells, since `mCells` retains the full block. |
