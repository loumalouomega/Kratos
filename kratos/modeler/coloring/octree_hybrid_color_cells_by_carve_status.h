//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#pragma once

// System includes

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridColorCellsByCarveStatus
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex cells by their non-destructive carve
 *        classification.
 * @details Iterates `OctreeHybridMesherData::mCarveStatus` and sets
 *          `mCellColor[i] = color` for every cell whose status satisfies
 *          `min_status <= mCarveStatus[i] <= max_status`.
 *
 *          `mCarveStatus` is populated by the non-destructive
 *          `OctreeHybridMeshUtility::ProjectToIsoSurface` overload during
 *          `ApplyRefinement` when `"project_to_surface": true`. Running this stage
 *          before that projection step (i.e. when `mCarveStatus` is empty or has
 *          the wrong size) is an error.
 *
 *          ### Status convention
 *          | Value | Meaning |
 *          |-------|---------|
 *          |   0   | Outside the surface (carved-but-retained) |
 *          |   1   | Core (inside the surface, projected) |
 *          |   2   | Buffer shell (new cells appended during projection) |
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised (or its
 *          size differs from `mCells.size()`), it is resized and filled with 0 before
 *          any colour is written.
 *
 *          @warning In normal usage `mCellColor` is already sized to `mCells.size()`
 *          by `OctreeHybridMeshGeneratorModeler::ApplyColoring`, which pre-fills it
 *          with the top-level `"default_outside_color"` parameter (default 1) before
 *          any coloring stage runs. The "resize and fill with 0" behaviour above is
 *          therefore effectively dead in single-stage pipelines. To leave
 *          `mCarveStatus == 0` cells at color 0 (e.g. to exclude them from volume
 *          entity generation while keeping them selectable by a later stage), set
 *          `"default_outside_color": 0` at the top level of the modeler parameters.
 *
 * ### Usage
 * @code{.json}
 * {
 *     "type"       : "OctreeHybridColorCellsByCarveStatus",
 *     "color"      : 1,
 *     "min_status" : 1,
 *     "max_status" : 2
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridClassifyCellsInsideOutside
 * @see OctreeHybridMesherData::mCarveStatus
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsByCarveStatus : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsByCarveStatus() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsByCarveStatus(OctreeHybridColorCellsByCarveStatus const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours cells whose carve status is within `[min_status, max_status]`.
     * @param rModeler            The owning modeler; `mCellColor` in its shared data is
     *                            updated in-place.
     * @param ColoringParameters  Validated JSON parameters for this step (see
     *                            @ref GetDefaultParameters for the schema).
     */
    void Apply(OctreeHybridMeshGeneratorModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema.
     * @return Parameters object:
     * @code{.json}
     * {
     *     "type"       : "OctreeHybridColorCellsByCarveStatus",
     *     "color"      : 1,
     *     "min_status" : 0,
     *     "max_status" : 2
     * }
     * @endcode
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsByCarveStatus.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsByCarveStatus)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsByCarveStatus.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsByCarveStatus)

    ///@}
};

///@}

} // namespace Kratos
