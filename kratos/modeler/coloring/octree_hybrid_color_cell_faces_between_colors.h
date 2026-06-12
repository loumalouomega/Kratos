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
 * @class OctreeHybridColorCellFacesBetweenColors
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex-cell faces lying on the interface between
 *        two cell-colour regions.
 * @details For every cell `c` whose entry in `OctreeHybridMesherData::mCellColor`
 *          equals `cell_color`, and for every one of its 6 local faces `f`
 *          (`FACE_FIDC`/Hexahedra3D8 ordering, see
 *          `OctreeHybridMeshUtility::ExtractBoundaryFaces`), the face-neighbour
 *          cell across `f` is looked up via
 *          `OctreeHybridMeshUtility::ComputeCellFaceNeighbors`. The neighbour's
 *          colour is:
 *          - `mCellColor[neighbour]`, if `f` is an interior face (shared with
 *            another cell), or
 *          - `default_outside_color`, if `f` lies on the outer boundary of the
 *            extracted mesh (no neighbour cell).
 *
 *          When the neighbour colour equals `outside_color`, `f` is on the
 *          requested colour transition and `OctreeHybridMesherData::mCellFaceColor[c][f]`
 *          is set to `color`. This is the octree-hybrid analogue of the voxel
 *          mesher's `ColorCellFacesBetweenColors`
 *          (`CartesianMeshColors::CalculateElementalFaceColorsBetweenColors`).
 *
 *          As a guard against a no-op configuration, if `color == outside_color`
 *          the method returns immediately without touching `mCellFaceColor`
 *          (mirrors the voxel-mesher behaviour).
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised yet (or
 *          its size differs from `mCells.size()`), it is resized and filled with 0
 *          before the colour comparisons. Likewise, if
 *          `OctreeHybridMesherData::mCellFaceColor` has not been initialised yet (or
 *          its size differs from `mCells.size()`), it is resized and filled with
 *          `{0,0,0,0,0,0}` per cell before any face colour is written. Faces that
 *          were already coloured by a previous instance of this stage are
 *          preserved unless overwritten by the current pass.
 *
 * ### Typical pipeline position
 * Run after the coloring stages that produce `mCellColor` (e.g.
 * `OctreeHybridClassifyCellsInsideOutside`, `OctreeHybridColorCellsByLevel`,
 * `OctreeHybridColorCellsWithInsideCenter`), so that the colour regions to be
 * compared already exist.
 *
 * ### Usage
 * Add the following block to the `"coloring_settings_list"` in the modeler JSON,
 * after the stages that establish `mCellColor`:
 * @code{.json}
 * {
 *     "type"                  : "OctreeHybridColorCellFacesBetweenColors",
 *     "color"                 : 2,
 *     "cell_color"            : 1,
 *     "outside_color"         : 0,
 *     "default_outside_color" : 0
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridMeshUtility::ComputeCellFaceNeighbors
 * @see OctreeHybridMesherData::mCellFaceColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellFacesBetweenColors : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellFacesBetweenColors() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellFacesBetweenColors(OctreeHybridColorCellFacesBetweenColors const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours, for every cell with colour `cell_color`, the local faces whose
     *        neighbour (or the implicit outer-boundary neighbour) carries `outside_color`.
     * @param rModeler            The owning modeler; `mCellFaceColor` in its shared data
     *                            is updated in-place.
     * @param ColoringParameters  Validated JSON parameters for this step (see
     *                            @ref GetDefaultParameters for the schema).
     */
    void Apply(OctreeHybridMeshGeneratorModeler& rModeler, Parameters ColoringParameters) const override;

    /**
     * @brief Returns the default parameter schema.
     * @return Parameters object:
     * @code{.json}
     * {
     *     "type"                  : "OctreeHybridColorCellFacesBetweenColors",
     *     "model_part_name"       : "Undefined",
     *     "color"                 : -1,
     *     "cell_color"            : -1,
     *     "outside_color"         : 0,
     *     "default_outside_color" : 0
     * }
     * @endcode
     * `"model_part_name"` is unused by this stage; it is kept for schema
     * consistency with other coloring components and is auto-filled by the
     * modeler dispatch loop when absent. `"default_outside_color"` is normally
     * overridden by the modeler's top-level `"default_outside_color"` setting.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellFacesBetweenColors.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellFacesBetweenColors)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellFacesBetweenColors.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellFacesBetweenColors)

    ///@}
};

///@}

} // namespace Kratos
