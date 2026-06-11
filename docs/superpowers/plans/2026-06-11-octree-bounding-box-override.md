# Octree Bounding Box Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `OctreeHybridMeshGeneratorModeler` accept an explicit octree bounding box (via `"bounding_box"` or `"bounding_box_model_part"`), validate it fully contains the input model part's own bounding box, and use it to override the auto-computed octree domain in `OctreeHybridMeshUtility::BuildFromSurfaceMesh`/`BuildAdaptiveFromSurfaceMesh`.

**Architecture:** A new private `ResolveOctreeBoundingBox()` method (called from `Initialize()`) reads the two existing-but-unused parameters, builds `mOctreeBoundingBox`/`mInputBoundingBox`, and validates containment. `BuildFromSurfaceMesh`/`BuildAdaptiveFromSurfaceMesh`/`BuildRefineSets` gain an optional `const BoundingBox<Point>* pOverrideBoundingBox = nullptr` that, when non-null, replaces the geometry-derived `lo`/`hi` extents. `OctreeHybridRefineInterfaceCells::Refine` passes the resolved override through on the first (build) call.

**Tech Stack:** C++20, Kratos core (`BoundingBox<Point>`, `Parameters`), GTest via `KRATOS_TEST_CASE_IN_SUITE`.

**Spec:** `docs/superpowers/specs/2026-06-11-octree-bounding-box-override-design.md`

---

## Build & test commands

Build (full reconfigure+build, ~minutes — only run when source files changed):
```bash
bash build/configure.sh
```

Run a filtered subset of the core C++ test suite (fast, no rebuild needed if binary is up to date):
```bash
OMP_NUM_THREADS=1 LD_LIBRARY_PATH=bin/Release/libs ./bin/Release/test/KratosCoreTest --gtest_filter='*OctreeHybrid*'
```

---

### Task 1: Modeler header — new members and accessors

**Files:**
- Modify: `kratos/modeler/octree_hybrid_mesh_generator_modeler.h`

- [ ] **Step 1: Add the `HasOctreeBoundingBox()` accessor declaration**

In `kratos/modeler/octree_hybrid_mesh_generator_modeler.h`, immediately after the existing `GetOctreeBoundingBox()` const overload (around line 294), add:

```cpp
    /**
     * @brief Returns whether an explicit octree bounding box override was resolved
     *        from the `"bounding_box"` or `"bounding_box_model_part"` parameters.
     * @details Set by @ref ResolveOctreeBoundingBox during @ref Initialize. When `false`,
     *          @ref GetOctreeBoundingBox returns a default-constructed (degenerate) box and
     *          the octree build falls back to its auto-computed domain.
     * @return `true` if an override is in effect.
     */
    bool HasOctreeBoundingBox() const;
```

So the surrounding code reads:

```cpp
    /**
     * @brief Returns the bounding box of the octree
     * @return The bounding box of the octree
     */
     BoundingBox<Point>& GetOctreeBoundingBox();

    /**
     * @brief Returns the bounding box of the octree (const version)
     * @return The bounding box of the octree
     */
    const BoundingBox<Point>& GetOctreeBoundingBox() const;

    /**
     * @brief Returns whether an explicit octree bounding box override was resolved
     *        from the `"bounding_box"` or `"bounding_box_model_part"` parameters.
     * @details Set by @ref ResolveOctreeBoundingBox during @ref Initialize. When `false`,
     *          @ref GetOctreeBoundingBox returns a default-constructed (degenerate) box and
     *          the octree build falls back to its auto-computed domain.
     * @return `true` if an override is in effect.
     */
    bool HasOctreeBoundingBox() const;
```

- [ ] **Step 2: Add the `mOctreeBoundingBoxSet` member**

Immediately after the existing `mInputBoundingBox` member declaration (around line 446), add:

```cpp
    /// The bounding box of the input model part
    BoundingBox<Kratos::Point> mInputBoundingBox;

    /// Whether @ref mOctreeBoundingBox was set from `"bounding_box"` /
    /// `"bounding_box_model_part"` (vs. left default for the auto-computed domain).
    bool mOctreeBoundingBoxSet = false;
```

- [ ] **Step 3: Declare `ResolveOctreeBoundingBox()`**

In the `Private Operations` section, immediately before `PreparingTheInternalDataStructure` (around line 456), add:

```cpp
    /**
     * @brief Resolves the octree bounding box override from `"bounding_box"` or
     *        `"bounding_box_model_part"`, validating it against the input model part.
     * @details Called from @ref Initialize, after @ref ReadModelParts (so
     *          @ref GetInputModelPart is available).
     *
     * - `KRATOS_ERROR` if both `"bounding_box"` (non-empty `min_point`/`max_point`) and a
     *   non-empty `"bounding_box_model_part"` are provided.
     * - If `"bounding_box"` has both `min_point` and `max_point` of size 3, builds
     *   @ref mOctreeBoundingBox directly from them.
     * - Else if `"bounding_box_model_part"` is non-empty, looks it up in the `Model`
     *   (`KRATOS_ERROR` if missing or empty) and builds @ref mOctreeBoundingBox from its
     *   nodes.
     * - Else, leaves @ref mOctreeBoundingBoxSet `false` (no override; existing
     *   auto-computed-domain behaviour is unchanged).
     *
     * When an override is resolved, computes @ref mInputBoundingBox from
     * @ref GetInputModelPart's nodes (`KRATOS_ERROR` if it has none) and `KRATOS_ERROR`s if
     * @ref mOctreeBoundingBox does not fully contain it (within a relative tolerance of
     * `1e-6` of the override box's diagonal).
     */
    void ResolveOctreeBoundingBox();

    /**
     * @brief This initializes de internal cartesian mesh data structure to be used for coloring
     * @param rTheInputModelPart The input model part
     */
    void PreparingTheInternalDataStructure(ModelPart& rTheInputModelPart);
```

- [ ] **Step 4: Document the two parameters in `GetDefaultParameters()`'s Doxygen**

In the `GetDefaultParameters()` doc comment (around lines 218-246), update the `@code{.json}` block and add an explanatory paragraph. Replace:

```cpp
     * @code{.json}
     * {
     *     "refinement_settings_list"  : [],
     *     "coloring_settings_list"  : [],
     *     "entities_generator_list" : [],
     *     "model_part_operations"   : [],
     *     "mdpa_file_name"          : "",
     *     "input_model_part_name"   : "",
     *     "default_outside_color"   : 1,
     *     "output_files"            : [],
     *     "remove_orphan_nodes"     : true,
     *     "echo_level"              : 1
     * }
     * @endcode
     * The `refinement_settings_list` must start with an @ref OctreeHybridRefineInterfaceCells
```

with:

```cpp
     * @code{.json}
     * {
     *     "refinement_settings_list"  : [],
     *     "coloring_settings_list"  : [],
     *     "entities_generator_list" : [],
     *     "model_part_operations"   : [],
     *     "mdpa_file_name"          : "",
     *     "input_model_part_name"   : "",
     *     "bounding_box_model_part" : "",
     *     "bounding_box"            : { "min_point" : [], "max_point" : [] },
     *     "default_outside_color"   : 1,
     *     "output_files"            : [],
     *     "remove_orphan_nodes"     : true,
     *     "echo_level"              : 1
     * }
     * @endcode
     * `"bounding_box"` (both `min_point` and `max_point` given as 3-vectors) and
     * `"bounding_box_model_part"` (the name of another ModelPart in the `Model`) are mutually
     * exclusive ways to override the octree's domain; see @ref ResolveOctreeBoundingBox. When
     * neither is set, the domain is computed automatically from the input surface, as before.
     * The `refinement_settings_list` must start with an @ref OctreeHybridRefineInterfaceCells
```

- [ ] **Step 5: Commit**

```bash
git add kratos/modeler/octree_hybrid_mesh_generator_modeler.h
git commit -m "Declare octree bounding box override resolution API"
```

---

### Task 2: Modeler implementation — `ResolveOctreeBoundingBox()`

**Files:**
- Modify: `kratos/modeler/octree_hybrid_mesh_generator_modeler.cpp`

- [ ] **Step 1: Add `<cmath>` to the system includes**

At the top of `kratos/modeler/octree_hybrid_mesh_generator_modeler.cpp`, the file currently has:

```cpp
// System includes

// External includes
```

Change to:

```cpp
// System includes
#include <cmath>

// External includes
```

- [ ] **Step 2: Implement `HasOctreeBoundingBox()`**

Immediately after the existing `GetOctreeBoundingBox() const` definition (around line 113), add:

```cpp
/***********************************************************************************/
/***********************************************************************************/

bool OctreeHybridMeshGeneratorModeler::HasOctreeBoundingBox() const
{
    return mOctreeBoundingBoxSet;
}
```

- [ ] **Step 3: Implement `ResolveOctreeBoundingBox()`**

Add the new method right before `PreparingTheInternalDataStructure` (around line 528, just above its current definition):

```cpp
/***********************************************************************************/
/***********************************************************************************/

void OctreeHybridMeshGeneratorModeler::ResolveOctreeBoundingBox()
{
    KRATOS_TRY

    const Parameters bounding_box = mParameters["bounding_box"];
    const bool has_explicit_bounding_box = bounding_box["min_point"].size() == 3 && bounding_box["max_point"].size() == 3;
    const std::string bounding_box_model_part_name = mParameters["bounding_box_model_part"].GetString();
    const bool has_bounding_box_model_part = !bounding_box_model_part_name.empty();

    KRATOS_ERROR_IF(has_explicit_bounding_box && has_bounding_box_model_part)
        << "OctreeHybridMeshGeneratorModeler: \"bounding_box\" and \"bounding_box_model_part\" "
        << "cannot both be defined. Choose one." << std::endl;

    if (has_explicit_bounding_box) {
        const array_1d<double, 3> min_point = bounding_box["min_point"].GetVector();
        const array_1d<double, 3> max_point = bounding_box["max_point"].GetVector();
        mOctreeBoundingBox = BoundingBox<Point>(Point(min_point), Point(max_point));
        mOctreeBoundingBoxSet = true;
    } else if (has_bounding_box_model_part) {
        KRATOS_ERROR_IF_NOT(mpModel->HasModelPart(bounding_box_model_part_name))
            << "OctreeHybridMeshGeneratorModeler: \"bounding_box_model_part\" '"
            << bounding_box_model_part_name << "' was not found in the Model." << std::endl;
        ModelPart& r_bounding_box_model_part = mpModel->GetModelPart(bounding_box_model_part_name);
        KRATOS_ERROR_IF(r_bounding_box_model_part.NumberOfNodes() == 0)
            << "OctreeHybridMeshGeneratorModeler: \"bounding_box_model_part\" '"
            << bounding_box_model_part_name << "' has no nodes." << std::endl;
        mOctreeBoundingBox = BoundingBox<Point>(r_bounding_box_model_part.NodesBegin(), r_bounding_box_model_part.NodesEnd());
        mOctreeBoundingBoxSet = true;
    } else {
        mOctreeBoundingBoxSet = false;
        return;
    }

    // Validate that the resolved octree bounding box fully contains the input model part.
    ModelPart& r_input_model_part = GetInputModelPart();
    KRATOS_ERROR_IF(r_input_model_part.NumberOfNodes() == 0)
        << "OctreeHybridMeshGeneratorModeler: input model part '" << GetInputModelPartName()
        << "' has no nodes; cannot validate the octree bounding box against it." << std::endl;
    mInputBoundingBox = BoundingBox<Point>(r_input_model_part.NodesBegin(), r_input_model_part.NodesEnd());

    const auto& r_octree_min = mOctreeBoundingBox.GetMinPoint();
    const auto& r_octree_max = mOctreeBoundingBox.GetMaxPoint();
    double diagonal_squared = 0.0;
    for (unsigned int i = 0; i < 3; ++i) {
        const double extent = r_octree_max[i] - r_octree_min[i];
        diagonal_squared += extent * extent;
    }
    const double tolerance = 1e-6 * std::sqrt(diagonal_squared);

    KRATOS_ERROR_IF_NOT(
        mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMinPoint(), tolerance) &&
        mOctreeBoundingBox.IsInside(mInputBoundingBox.GetMaxPoint(), tolerance))
        << "OctreeHybridMeshGeneratorModeler: the octree bounding box "
        << "[(" << r_octree_min[0] << ", " << r_octree_min[1] << ", " << r_octree_min[2] << "), ("
        << r_octree_max[0] << ", " << r_octree_max[1] << ", " << r_octree_max[2] << ")] "
        << "does not contain the input model part '" << GetInputModelPartName() << "' bounding box "
        << "[(" << mInputBoundingBox.GetMinPoint()[0] << ", " << mInputBoundingBox.GetMinPoint()[1] << ", " << mInputBoundingBox.GetMinPoint()[2] << "), ("
        << mInputBoundingBox.GetMaxPoint()[0] << ", " << mInputBoundingBox.GetMaxPoint()[1] << ", " << mInputBoundingBox.GetMaxPoint()[2] << ")]."
        << std::endl;

    KRATOS_CATCH("")
}
```

- [ ] **Step 4: Call it from `Initialize()`**

`Initialize()` currently reads:

```cpp
void OctreeHybridMeshGeneratorModeler::Initialize()
{
    // Get the echo level
    mEchoLevel = mParameters["echo_level"].GetInt();

    // Read the model parts
    ReadModelParts();

    // Prepare the internal data structure
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Preparing Internal Data Structure" << std::endl;
    PreparingTheInternalDataStructure(GetInputModelPart());
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Internal Data Structure prepared" << std::endl;
}
```

Change to:

```cpp
void OctreeHybridMeshGeneratorModeler::Initialize()
{
    // Get the echo level
    mEchoLevel = mParameters["echo_level"].GetInt();

    // Read the model parts
    ReadModelParts();

    // Resolve and validate the octree bounding box override, if any
    ResolveOctreeBoundingBox();

    // Prepare the internal data structure
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Preparing Internal Data Structure" << std::endl;
    PreparingTheInternalDataStructure(GetInputModelPart());
    KRATOS_INFO_IF(GetLabel(), mEchoLevel > 0) << "Internal Data Structure prepared" << std::endl;
}
```

- [ ] **Step 5: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds (this task adds dead code only — `ResolveOctreeBoundingBox` always takes the "no override" branch with default parameters, so existing tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add kratos/modeler/octree_hybrid_mesh_generator_modeler.cpp
git commit -m "Implement octree bounding box override resolution and validation"
```

---

### Task 3: Modeler-level tests for `ResolveOctreeBoundingBox()`

**Files:**
- Modify: `kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp`

These tests call `Initialize()` directly (not `SetupModelPart()`/`RunModeler()`), since bounding-box resolution happens before any octree is built and only needs `Nodes()` on the input ModelPart — no surface triangles required.

- [ ] **Step 1: Write the five failing tests**

Add the following near the end of the `OctreeHybridMeshGeneratorModeler — top-level modeler tests` section (i.e. after `OctreeHybridMeshGeneratorModelerDualCarveBbox`, before `OctreeHybridMeshGeneratorModelerDefaultParametersValid`):

```cpp
// ===========================================================================
// OctreeHybridMeshGeneratorModeler — bounding box override resolution
// ===========================================================================

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerNoBoundingBoxOverride, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin"
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_FALSE(modeler.HasOctreeBoundingBox());
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerExplicitBoundingBoxResolved, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "bounding_box" : {
            "min_point" : [0.0, 0.0, 0.0],
            "max_point" : [1.0, 1.0, 1.0]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_TRUE(modeler.HasOctreeBoundingBox());
    const auto& r_min = modeler.GetOctreeBoundingBox().GetMinPoint();
    const auto& r_max = modeler.GetOctreeBoundingBox().GetMaxPoint();
    for (unsigned int i = 0; i < 3; ++i) {
        KRATOS_EXPECT_NEAR(r_min[i], 0.0, 1e-12);
        KRATOS_EXPECT_NEAR(r_max[i], 1.0, 1e-12);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBoundingBoxModelPartResolved, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    ModelPart& r_bbox = model.CreateModelPart("BBox");
    r_bbox.CreateNewNode(1, 0.0, 0.0, 0.0);
    r_bbox.CreateNewNode(2, 1.0, 1.0, 1.0);

    Parameters settings(R"({
        "input_model_part_name"   : "Skin",
        "bounding_box_model_part" : "BBox"
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);
    modeler.Initialize();

    KRATOS_EXPECT_TRUE(modeler.HasOctreeBoundingBox());
    const auto& r_min = modeler.GetOctreeBoundingBox().GetMinPoint();
    const auto& r_max = modeler.GetOctreeBoundingBox().GetMaxPoint();
    for (unsigned int i = 0; i < 3; ++i) {
        KRATOS_EXPECT_NEAR(r_min[i], 0.0, 1e-12);
        KRATOS_EXPECT_NEAR(r_max[i], 1.0, 1e-12);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBothBoundingBoxSourcesThrows, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    ModelPart& r_bbox = model.CreateModelPart("BBox");
    r_bbox.CreateNewNode(1, 0.0, 0.0, 0.0);
    r_bbox.CreateNewNode(2, 1.0, 1.0, 1.0);

    Parameters settings(R"({
        "input_model_part_name"   : "Skin",
        "bounding_box_model_part" : "BBox",
        "bounding_box" : {
            "min_point" : [0.0, 0.0, 0.0],
            "max_point" : [1.0, 1.0, 1.0]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.Initialize(), "");
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerBoundingBoxConflictThrows, KratosCoreFastSuite)
{
    Model model;
    ModelPart& r_skin = model.CreateModelPart("Skin");
    r_skin.CreateNewNode(1, 0.2, 0.2, 0.2);
    r_skin.CreateNewNode(2, 0.8, 0.8, 0.8);

    Parameters settings(R"({
        "input_model_part_name" : "Skin",
        "bounding_box" : {
            "min_point" : [0.3, 0.3, 0.3],
            "max_point" : [0.7, 0.7, 0.7]
        }
    })");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.Initialize(), "");
}
```

- [ ] **Step 2: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds.

- [ ] **Step 3: Run the new tests**

```bash
OMP_NUM_THREADS=1 LD_LIBRARY_PATH=bin/Release/libs ./bin/Release/test/KratosCoreTest --gtest_filter='*OctreeHybridMeshGeneratorModeler*BoundingBox*'
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp
git commit -m "Add tests for octree bounding box override resolution and validation"
```

---

### Task 4: Utility header — override parameter on the build functions

**Files:**
- Modify: `kratos/modeler/utilities/octree_hybrid_mesh_utility.h`

- [ ] **Step 1: Add the bounding-box includes**

The file currently starts with:

```cpp
// Project includes
#include "spatial_containers/octree_hybrid.h"
#include "spatial_containers/octree_hybrid_cell.h"
#include "spatial_containers/octree_hybrid_configure.h"
```

Change to:

```cpp
// Project includes
#include "geometries/bounding_box.h"
#include "geometries/point.h"
#include "spatial_containers/octree_hybrid.h"
#include "spatial_containers/octree_hybrid_cell.h"
#include "spatial_containers/octree_hybrid_configure.h"
```

- [ ] **Step 2: Extend `BuildFromSurfaceMesh`'s signature and docs**

Replace:

```cpp
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        bool Adaptive = true);
```

with:

```cpp
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        bool Adaptive = true,
        const BoundingBox<Point>* pOverrideBoundingBox = nullptr);
```

And, in the `@details` of its Doxygen comment immediately above, append a final sentence:

```
     *          When @p pOverrideBoundingBox is non-null, its min/max points are used as the
     *          octree's world-space domain instead of the geometry-derived extents (no
     *          1% auto-padding is applied in the non-adaptive path; in the adaptive path
     *          the centred reference cube is derived from this box instead of the
     *          triangle-corner extents).
```

so the full comment block reads:

```cpp
    /**
     * @brief Builds an OctreeHybrid from a surface mesh, with optional adaptive refinement.
     *
     * @details Sets the bounding box from the surface geometry, inserts all surface
     *          triangle vertices into the octree, and refines the octree to
     *          @p RefinementDepth.  When @p Adaptive is `true`, the reference
     *          HybridOctree_Hex curvature + feature-thickness criterion is used
     *          (see @ref BuildAdaptiveFromSurfaceMesh); when `false`, every cell
     *          containing a surface vertex is uniformly subdivided to
     *          @p RefinementDepth via @ref RefineInterfaceCells.
     *          When @p pOverrideBoundingBox is non-null, its min/max points are used as the
     *          octree's world-space domain instead of the geometry-derived extents (no
     *          1% auto-padding is applied in the non-adaptive path; in the adaptive path
     *          the centred reference cube is derived from this box instead of the
     *          triangle-corner extents).
     *
     * @param rSurfaceMesh     ModelPart whose Geometries() container holds the
     *                         surface triangles (populated by StlIO::ReadModelPart).
     * @param RefinementDepth  Maximum refinement depth near the surface.
     *                         Must be in [1, OctreeHybridKratosConfiguration::MAX_DEPTH].
     * @param Adaptive         When `true` (default) use the curvature + thickness
     *                         criterion from the reference code so that leaf counts
     *                         match the reference for real geometries.  When `false`
     *                         use simple interface-cell refinement (faster but less
     *                         resolution near high-curvature features).
     * @param pOverrideBoundingBox  Optional world-space domain override (see @details).
     * @return Unique pointer to the built (but not yet 2:1-balanced) octree.
     */
    static std::unique_ptr<OctreeType> BuildFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        bool Adaptive = true,
        const BoundingBox<Point>* pOverrideBoundingBox = nullptr);
```

- [ ] **Step 3: Extend `BuildAdaptiveFromSurfaceMesh`'s signature and docs**

Replace:

```cpp
    static std::unique_ptr<OctreeType> BuildAdaptiveFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth);
```

with:

```cpp
    static std::unique_ptr<OctreeType> BuildAdaptiveFromSurfaceMesh(
        ModelPart& rSurfaceMesh,
        std::size_t RefinementDepth,
        const BoundingBox<Point>* pOverrideBoundingBox = nullptr);
```

And add a final `@param` line to its Doxygen comment immediately above (right before `@return`):

```
     * @param pOverrideBoundingBox  Optional world-space domain override: when non-null, its
     *                              min/max points replace the triangle-corner extents used to
     *                              derive the centred reference cube (`cube_lo`/`cube_side`).
```

- [ ] **Step 4: Build**

```bash
bash build/configure.sh
```

Expected: build **fails** — `BuildFromSurfaceMesh`/`BuildAdaptiveFromSurfaceMesh` definitions in `octree_hybrid_mesh_utility.cpp` no longer match their declarations (extra parameter declared but not defined yet). This confirms the header change is wired to the (not-yet-updated) implementation.

- [ ] **Step 5: Commit**

```bash
git add kratos/modeler/utilities/octree_hybrid_mesh_utility.h
git commit -m "Declare optional bounding box override on octree build functions"
```

---

### Task 5: Utility implementation — apply the override in `BuildRefineSets`/`BuildAdaptiveFromSurfaceMesh`/`BuildFromSurfaceMesh`

**Files:**
- Modify: `kratos/modeler/utilities/octree_hybrid_mesh_utility.cpp`

- [ ] **Step 1: Add the override parameter to `BuildRefineSets` and apply it**

`BuildRefineSets` (anonymous-namespace helper, around line 59) currently reads:

```cpp
AdaptiveRefineData BuildRefineSets(ModelPart& rSurfaceMesh)
{
    constexpr double PI = 3.1415926535897932384626433;
    AdaptiveRefineData data;

    // --- Gather triangle corners (world coords) and the bounding box ------
    std::vector<std::array<double,3>> corners;       // 3 per triangle
    corners.reserve(rSurfaceMesh.NumberOfGeometries() * 3);
    double lo[3] = { std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max() };
    double hi[3] = { std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest() };
    for (auto& g : rSurfaceMesh.Geometries()) {
        if (g.PointsNumber() < 3) continue;
        for (int k = 0; k < 3; ++k) {
            const double x = g[k].X(), y = g[k].Y(), z = g[k].Z();
            corners.push_back({x,y,z});
            lo[0]=std::min(lo[0],x); hi[0]=std::max(hi[0],x);
            lo[1]=std::min(lo[1],y); hi[1]=std::max(hi[1],y);
            lo[2]=std::min(lo[2],z); hi[2]=std::max(hi[2],z);
        }
        data.tri_geom.push_back(&g);
    }
    const int nTri = static_cast<int>(data.tri_geom.size());
    if (nTri == 0) return data;

    // --- Reference cube: centred, side = largest extent (START_POINT/BOX_LENGTH)
    double L = hi[0]-lo[0];
```

Change the signature and insert the override application between the `if (nTri == 0) return data;` line and the `// --- Reference cube ...` comment:

```cpp
AdaptiveRefineData BuildRefineSets(ModelPart& rSurfaceMesh, const BoundingBox<Point>* pOverrideBoundingBox = nullptr)
{
    constexpr double PI = 3.1415926535897932384626433;
    AdaptiveRefineData data;

    // --- Gather triangle corners (world coords) and the bounding box ------
    std::vector<std::array<double,3>> corners;       // 3 per triangle
    corners.reserve(rSurfaceMesh.NumberOfGeometries() * 3);
    double lo[3] = { std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max() };
    double hi[3] = { std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest() };
    for (auto& g : rSurfaceMesh.Geometries()) {
        if (g.PointsNumber() < 3) continue;
        for (int k = 0; k < 3; ++k) {
            const double x = g[k].X(), y = g[k].Y(), z = g[k].Z();
            corners.push_back({x,y,z});
            lo[0]=std::min(lo[0],x); hi[0]=std::max(hi[0],x);
            lo[1]=std::min(lo[1],y); hi[1]=std::max(hi[1],y);
            lo[2]=std::min(lo[2],z); hi[2]=std::max(hi[2],z);
        }
        data.tri_geom.push_back(&g);
    }
    const int nTri = static_cast<int>(data.tri_geom.size());
    if (nTri == 0) return data;

    // An explicit domain override replaces the geometry-derived extents used to
    // derive the centred reference cube below.
    if (pOverrideBoundingBox) {
        for (int d = 0; d < 3; ++d) {
            lo[d] = (*pOverrideBoundingBox).GetMinPoint()[d];
            hi[d] = (*pOverrideBoundingBox).GetMaxPoint()[d];
        }
    }

    // --- Reference cube: centred, side = largest extent (START_POINT/BOX_LENGTH)
    double L = hi[0]-lo[0];
```

- [ ] **Step 2: Thread the override through `BuildAdaptiveFromSurfaceMesh`**

Replace:

```cpp
auto OctreeHybridMeshUtility::BuildAdaptiveFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth) -> std::unique_ptr<OctreeType>
{
    const AdaptiveRefineData data = BuildRefineSets(rSurfaceMesh);
```

with:

```cpp
auto OctreeHybridMeshUtility::BuildAdaptiveFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth,
    const BoundingBox<Point>* pOverrideBoundingBox) -> std::unique_ptr<OctreeType>
{
    const AdaptiveRefineData data = BuildRefineSets(rSurfaceMesh, pOverrideBoundingBox);
```

- [ ] **Step 3: Thread the override through `BuildFromSurfaceMesh`**

Replace:

```cpp
auto OctreeHybridMeshUtility::BuildFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth,
    bool Adaptive) -> std::unique_ptr<OctreeType>
{
    KRATOS_ERROR_IF(RefinementDepth < 1 || RefinementDepth > ConfigurationType::MAX_DEPTH)
        << "OctreeHybridMeshUtility: RefinementDepth must be in [1, "
        << ConfigurationType::MAX_DEPTH << "], got " << RefinementDepth << std::endl;

    if (Adaptive)
        return BuildAdaptiveFromSurfaceMesh(rSurfaceMesh, RefinementDepth);

    // ----------------------------------------------------------------- //
    //  Uniform refinement (legacy path): every leaf whose box intersects
    //  any triangle is split to RefinementDepth.  Domain is the 1 %-padded
    //  axis-aligned bounding box.  Kept for the transition-template unit
    //  tests, whose synthetic flat patches carry no curvature and so would
    //  not refine under the adaptive criterion.
    // ----------------------------------------------------------------- //
    double lo[3] = { std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max() };
    double hi[3] = { std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest() };

    for (const auto& r_node : rSurfaceMesh.Nodes()) {
        lo[0] = std::min(lo[0], r_node.X()); hi[0] = std::max(hi[0], r_node.X());
        lo[1] = std::min(lo[1], r_node.Y()); hi[1] = std::max(hi[1], r_node.Y());
        lo[2] = std::min(lo[2], r_node.Z()); hi[2] = std::max(hi[2], r_node.Z());
    }
    // Inflate the bounding box by 1 % per side so surface triangles that
    // touch the exact domain boundary are still enclosed rather than clipped
    // by the octree's root cell.
    for (std::size_t d = 0; d < 3; ++d) {
        const double span = hi[d] - lo[d];
        lo[d] -= 0.01 * span;
        hi[d] += 0.01 * span;
    }

    auto p_octree = std::make_unique<OctreeType>(RefinementDepth);
    p_octree->SetBoundingBox(lo, hi);
```

with:

```cpp
auto OctreeHybridMeshUtility::BuildFromSurfaceMesh(
    ModelPart& rSurfaceMesh,
    std::size_t RefinementDepth,
    bool Adaptive,
    const BoundingBox<Point>* pOverrideBoundingBox) -> std::unique_ptr<OctreeType>
{
    KRATOS_ERROR_IF(RefinementDepth < 1 || RefinementDepth > ConfigurationType::MAX_DEPTH)
        << "OctreeHybridMeshUtility: RefinementDepth must be in [1, "
        << ConfigurationType::MAX_DEPTH << "], got " << RefinementDepth << std::endl;

    if (Adaptive)
        return BuildAdaptiveFromSurfaceMesh(rSurfaceMesh, RefinementDepth, pOverrideBoundingBox);

    // ----------------------------------------------------------------- //
    //  Uniform refinement (legacy path): every leaf whose box intersects
    //  any triangle is split to RefinementDepth.  Domain is the 1 %-padded
    //  axis-aligned bounding box (or pOverrideBoundingBox verbatim, if given).
    //  Kept for the transition-template unit tests, whose synthetic flat
    //  patches carry no curvature and so would not refine under the adaptive
    //  criterion.
    // ----------------------------------------------------------------- //
    double lo[3] = { std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max() };
    double hi[3] = { std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest(),
                     std::numeric_limits<double>::lowest() };

    if (pOverrideBoundingBox) {
        for (int d = 0; d < 3; ++d) {
            lo[d] = (*pOverrideBoundingBox).GetMinPoint()[d];
            hi[d] = (*pOverrideBoundingBox).GetMaxPoint()[d];
        }
    } else {
        for (const auto& r_node : rSurfaceMesh.Nodes()) {
            lo[0] = std::min(lo[0], r_node.X()); hi[0] = std::max(hi[0], r_node.X());
            lo[1] = std::min(lo[1], r_node.Y()); hi[1] = std::max(hi[1], r_node.Y());
            lo[2] = std::min(lo[2], r_node.Z()); hi[2] = std::max(hi[2], r_node.Z());
        }
        // Inflate the bounding box by 1 % per side so surface triangles that
        // touch the exact domain boundary are still enclosed rather than clipped
        // by the octree's root cell.
        for (std::size_t d = 0; d < 3; ++d) {
            const double span = hi[d] - lo[d];
            lo[d] -= 0.01 * span;
            hi[d] += 0.01 * span;
        }
    }

    auto p_octree = std::make_unique<OctreeType>(RefinementDepth);
    p_octree->SetBoundingBox(lo, hi);
```

The rest of `BuildFromSurfaceMesh` (triangle collection and refinement loop) is unchanged.

- [ ] **Step 4: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add kratos/modeler/utilities/octree_hybrid_mesh_utility.cpp
git commit -m "Apply optional bounding box override in octree build functions"
```

---

### Task 6: Utility-level tests for the override parameter

**Files:**
- Modify: `kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesh_utility.cpp`

- [ ] **Step 1: Write the two failing tests**

Add the following near the end of the `// Section G — Octree building and direct refinement` section (after `OctreeHybridMeshUtilityBuildFromSurfaceMeshLeafCountGrowsWithDepth`):

```cpp
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildFromSurfaceMeshUniformOverrideBoundingBox, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    const BoundingBox<Point> override_bbox(Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0));
    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, false, &override_bbox);

    double n_lo[3] = {0.0, 0.0, 0.0}, n_hi[3] = {1.0, 1.0, 1.0}, w_lo[3], w_hi[3];
    octree->ScaleBackToOriginalCoordinate(n_lo, w_lo);
    octree->ScaleBackToOriginalCoordinate(n_hi, w_hi);

    // The override is used verbatim: no 1% auto-padding is applied.
    for (int d = 0; d < 3; ++d) {
        KRATOS_EXPECT_NEAR(w_lo[d], 0.0, 1e-12);
        KRATOS_EXPECT_NEAR(w_hi[d], 1.0, 1e-12);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshUtilityBuildAdaptiveFromSurfaceMeshOverrideBoundingBox, KratosCoreFastSuite)
{
    Model model;
    BuildBoxSurface(model.CreateModelPart("Skin"), 0.3, 0.7);

    // Already a centred cube of side 2: cube_lo == min_point, cube_side == 2.
    const BoundingBox<Point> override_bbox(Point(-1.0, -1.0, -1.0), Point(1.0, 1.0, 1.0));
    auto octree = Util::BuildFromSurfaceMesh(model.GetModelPart("Skin"), 3, true, &override_bbox);

    double n_lo[3] = {0.0, 0.0, 0.0}, n_hi[3] = {1.0, 1.0, 1.0}, w_lo[3], w_hi[3];
    octree->ScaleBackToOriginalCoordinate(n_lo, w_lo);
    octree->ScaleBackToOriginalCoordinate(n_hi, w_hi);

    for (int d = 0; d < 3; ++d) {
        KRATOS_EXPECT_NEAR(w_lo[d], -1.0, 1e-12);
        KRATOS_EXPECT_NEAR(w_hi[d], 1.0, 1e-12);
    }
}
```

- [ ] **Step 2: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds.

- [ ] **Step 3: Run the new tests**

```bash
OMP_NUM_THREADS=1 LD_LIBRARY_PATH=bin/Release/libs ./bin/Release/test/KratosCoreTest --gtest_filter='*OctreeHybridMeshUtilityBuild*OverrideBoundingBox*'
```

Expected: both tests `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesh_utility.cpp
git commit -m "Add tests for octree build bounding box override"
```

---

### Task 7: Wire the override into `OctreeHybridRefineInterfaceCells::Refine`

**Files:**
- Modify: `kratos/modeler/refine_operations/refine_interface_cells_hybrid_octree.cpp`

- [ ] **Step 1: Pass the resolved override on the first (build) call**

Replace:

```cpp
        ModelPart& r_surface = rModeler.GetModel().GetModelPart(surface_name);
        // Cache the triangle soup in r_data so downstream stages
        // (RemoveOutsideElement, ClassifyInsideOutside, ProjectToIsoSurface) can
        // reuse it without re-parsing the ModelPart.
        r_data.mTriangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        r_data.mpOctree   = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
            r_surface,
            RefineParameters["refinement_depth"].GetInt(),
            RefineParameters["adaptive"].GetBool());
```

with:

```cpp
        ModelPart& r_surface = rModeler.GetModel().GetModelPart(surface_name);
        // Cache the triangle soup in r_data so downstream stages
        // (RemoveOutsideElement, ClassifyInsideOutside, ProjectToIsoSurface) can
        // reuse it without re-parsing the ModelPart.
        r_data.mTriangles = OctreeHybridMeshUtility::ExtractTriangleSoup(r_surface);
        const BoundingBox<Point>* p_bounding_box_override =
            rModeler.HasOctreeBoundingBox() ? &rModeler.GetOctreeBoundingBox() : nullptr;
        r_data.mpOctree   = OctreeHybridMeshUtility::BuildFromSurfaceMesh(
            r_surface,
            RefineParameters["refinement_depth"].GetInt(),
            RefineParameters["adaptive"].GetBool(),
            p_bounding_box_override);
```

- [ ] **Step 2: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add kratos/modeler/refine_operations/refine_interface_cells_hybrid_octree.cpp
git commit -m "Wire resolved octree bounding box override into the initial octree build"
```

---

### Task 8: End-to-end test through `SetupModelPart`

**Files:**
- Modify: `kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp`

This exercises the full pipeline: `"bounding_box"` set at the top level → `Initialize()` resolves and validates it → `OctreeHybridRefineInterfaceCells::Refine` builds the octree on that domain → the extracted dual mesh's nodes lie within it.

- [ ] **Step 1: Write the failing test**

Add the following after `OctreeHybridMeshGeneratorModelerDualCarveBbox` (and before the bounding-box-resolution tests added in Task 3):

```cpp
KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerDualExplicitBoundingBoxOverride, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    // Skin's own node bounding box is [0,0,0]-[1,1,1] (the bbox-pin nodes), which
    // the override below fully contains.
    constexpr double override_lo = -0.5, override_hi = 1.5;
    ModelPart& out = RunModeler(model, R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "bounding_box" : {
            "min_point" : [-0.5, -0.5, -0.5],
            "max_point" : [1.5, 1.5, 1.5]
        },
        "refinement_settings_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");

    KRATOS_EXPECT_GT(out.NumberOfElements(), 0u);

    // All output nodes must lie within the explicit override domain.
    for (const auto& r_node : out.Nodes()) {
        KRATOS_EXPECT_GE(r_node.X(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.X(), override_hi + 1e-9);
        KRATOS_EXPECT_GE(r_node.Y(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.Y(), override_hi + 1e-9);
        KRATOS_EXPECT_GE(r_node.Z(), override_lo - 1e-9);
        KRATOS_EXPECT_LE(r_node.Z(), override_hi + 1e-9);
    }
}

KRATOS_TEST_CASE_IN_SUITE(OctreeHybridMeshGeneratorModelerSetupModelPartBoundingBoxConflictThrows, KratosCoreFastSuite)
{
    constexpr double lo = 0.3, hi = 0.7;
    Model model;
    BuildClosedBoxSurface(model.CreateModelPart("Skin"), lo, hi);

    // Skin's own node bounding box is [0,0,0]-[1,1,1] (the bbox-pin nodes); this
    // override does not contain it.
    Parameters settings(R"({
        "input_model_part_name"  : "Skin",
        "output_model_part_name" : "Output",
        "bounding_box" : {
            "min_point" : [0.3, 0.3, 0.3],
            "max_point" : [0.7, 0.7, 0.7]
        },
        "refinement_settings_list" : [{ "type": "OctreeHybridRefineInterfaceCells",
                                      "refinement_depth": 3, "adaptive": false }],
        "coloring_settings_list" : [{ "type": "OctreeHybridClassifyCellsInsideOutside" }],
        "entities_generator_list": [{ "type": "GenerateHybridOctreeHexahedraElementsWithCellColor",
                                      "model_part_name": "Output", "color": 1 }],
        "model_part_operations"  : []
    })");
    settings.RemoveValue("output_model_part_name");
    OctreeHybridMeshGeneratorModeler modeler(model, settings);

    KRATOS_EXPECT_EXCEPTION_IS_THROWN(modeler.SetupModelPart(), "");
}
```

- [ ] **Step 2: Build**

```bash
bash build/configure.sh
```

Expected: build succeeds.

- [ ] **Step 3: Run the new tests**

```bash
OMP_NUM_THREADS=1 LD_LIBRARY_PATH=bin/Release/libs ./bin/Release/test/KratosCoreTest --gtest_filter='*OctreeHybridMeshGeneratorModeler*BoundingBox*:*OctreeHybridMeshUtilityBuild*OverrideBoundingBox*'
```

Expected: all tests `PASSED`.

- [ ] **Step 4: Run the full octree hybrid mesher test group as a regression check**

```bash
OMP_NUM_THREADS=1 LD_LIBRARY_PATH=bin/Release/libs ./bin/Release/test/KratosCoreTest --gtest_filter='*OctreeHybrid*'
```

Expected: all tests `PASSED` (no regressions in the existing 1%-padding / no-override paths).

- [ ] **Step 5: Commit**

```bash
git add kratos/tests/cpp_tests/modeler/test_octree_hybrid_mesher_modeler.cpp
git commit -m "Add end-to-end tests for octree bounding box override via SetupModelPart"
```

---

## Self-review notes

- **Spec coverage:** parameter resolution (Task 2), mutual-exclusion error (Task 2/3), conflict validation (Task 2/3), uniform-path override (Task 5/6), adaptive-path override (Task 5/6), wiring into `OctreeHybridRefineInterfaceCells` (Task 7), regression check (Task 8 step 4). All spec sections covered.
- **Type/signature consistency:** `HasOctreeBoundingBox() const`, `GetOctreeBoundingBox()` (existing), `mOctreeBoundingBox`/`mInputBoundingBox`/`mOctreeBoundingBoxSet` member names, and `BuildFromSurfaceMesh(ModelPart&, std::size_t, bool, const BoundingBox<Point>*)` / `BuildAdaptiveFromSurfaceMesh(ModelPart&, std::size_t, const BoundingBox<Point>*)` / `BuildRefineSets(ModelPart&, const BoundingBox<Point>*)` signatures are used identically across Tasks 1–8.
- **Out of scope honored:** `BuildAndWriteVtk`/`BuildCarveAndWriteVtk` call sites (octree_hybrid_mesh_utility.cpp lines ~1711-1765) are untouched — the new parameter defaults to `nullptr`.
