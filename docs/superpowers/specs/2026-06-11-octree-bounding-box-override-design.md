# Octree bounding box override — design

## Context

`OctreeHybridMeshGeneratorModeler` already declares two unused
`GetDefaultParameters()` entries:

```json
"bounding_box_model_part" : "",
"bounding_box" : { "min_point" : [], "max_point" : [] }
```

and an unused private member `BoundingBox<Kratos::Point> mInputBoundingBox;`.
Today the octree's domain is always derived automatically from the input
surface mesh inside `OctreeHybridMeshUtility::BuildFromSurfaceMesh` /
`BuildAdaptiveFromSurfaceMesh`. This feature lets the user pin the octree's
bounding box explicitly (or derive it from another ModelPart), while
guaranteeing the chosen box still fully contains the input geometry.

## Parameter resolution

New private method `OctreeHybridMeshGeneratorModeler::ResolveOctreeBoundingBox()`,
called from `Initialize()` right after `ReadModelParts()` (so
`mpInputModelPart` is available):

1. Read `"bounding_box"` (`min_point` / `max_point`, each either empty or a
   3-vector) and `"bounding_box_model_part"` (string) from `mParameters`.
2. `KRATOS_ERROR_IF` both are provided (non-empty `min_point`/`max_point`
   *and* non-empty `bounding_box_model_part`) — ambiguous configuration.
3. If `"bounding_box"` has both `min_point` and `max_point` of size 3:
   build `mOctreeBoundingBox` directly from those two points.
4. Else if `"bounding_box_model_part"` is non-empty: look up that ModelPart
   in `*mpModel` (`KRATOS_ERROR_IF_NOT(mpModel->HasModelPart(name))`), and
   build `mOctreeBoundingBox` from its nodes via
   `BoundingBox<Point>(begin, end)`. `KRATOS_ERROR_IF` it has zero nodes.
5. Else: no override. `mOctreeBoundingBoxSet = false`; existing
   auto-computed-from-surface behavior is unchanged.

New private member `bool mOctreeBoundingBoxSet = false;` and new public
const accessor:

```cpp
/// @brief Returns whether an explicit octree bounding box override was
///        resolved from "bounding_box" or "bounding_box_model_part".
bool HasOctreeBoundingBox() const;
```

## Conflict check

When an override is resolved (step 3 or 4 above):

1. Compute `mInputBoundingBox` from `GetInputModelPart().Nodes()` via
   `BoundingBox<Point>(begin, end)`. `KRATOS_ERROR_IF` the input model part
   has zero nodes (cannot validate).
2. Tolerance `tol = 1e-6 * ||GetMaxPoint() - GetMinPoint()||` of
   `mOctreeBoundingBox` (matches the relative-tolerance convention already
   used in `octree_hybrid_mesh_utility.cpp`).
3. Require `mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMinPoint(), tol)`
   and `mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMaxPoint(), tol)`.
   `KRATOS_ERROR` otherwise, printing both boxes' min/max coordinates.

## Wiring into the octree build

`OctreeHybridMeshUtility::BuildFromSurfaceMesh`, `BuildAdaptiveFromSurfaceMesh`,
and the internal `BuildRefineSets` helper gain an optional trailing parameter:

```cpp
const BoundingBox<Point>* pOverrideBoundingBox = nullptr
```

- **Uniform path** (`BuildFromSurfaceMesh`, `Adaptive == false`): when
  `pOverrideBoundingBox` is non-null, `lo`/`hi` are taken directly from its
  min/max points — the existing 1 % auto-padding step is skipped entirely
  (the user-provided box is used verbatim).
- **Adaptive path** (`BuildRefineSets` / `BuildAdaptiveFromSurfaceMesh`): when
  non-null, `lo`/`hi` (used to derive `cube_lo` / `cube_side`) come from the
  override instead of the triangle-corner extents. The existing
  centered-cube derivation (`cube_side = max(hi-lo)`, `cube_lo` centered)
  is unchanged — it now just operates on the override box's extents.

`OctreeHybridRefineInterfaceCells::Refine` (first-call / build branch) passes:

```cpp
rModeler.HasOctreeBoundingBox() ? &rModeler.GetOctreeBoundingBox() : nullptr
```

No other call sites of `BuildFromSurfaceMesh`/`BuildAdaptiveFromSurfaceMesh`
(e.g. `BuildAndWriteVtk`, `BuildCarveAndWriteVtk`) need to change — the new
parameter defaults to `nullptr`, preserving current behavior.

## Testing

New C++ GTest cases in `kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp`
(or `test_octree_hybrid_direct_operations.cpp` for the utility-level overload):

1. `"bounding_box"` with `min_point`/`max_point` that strictly contains the
   input surface → octree domain matches the explicit box (check extracted
   mesh node coordinates stay within it, similar to
   `OctreeHybridMeshGeneratorModelerDualCarveBbox`).
2. `"bounding_box_model_part"` referencing a second ModelPart whose nodes
   define a larger box → same containment check, derived from that
   ModelPart's extents.
3. Both `"bounding_box"` and `"bounding_box_model_part"` set → `KRATOS_ERROR`.
4. Override box that does **not** fully contain the input geometry →
   `KRATOS_ERROR`.
5. Neither set (existing tests) → unchanged behavior, regression check.

## Out of scope

- No change to `mInputBoundingBox`'s visibility/usage beyond the conflict
  check (it remains a private implementation detail).
- No change to `BuildAndWriteVtk` / `BuildCarveAndWriteVtk` public signatures
  beyond the new defaulted parameter.
- No support for 2D / non-3D bounding boxes.
