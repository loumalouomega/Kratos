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
 * @class OctreeHybridColorOuterCellFaces
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex-cell faces located on the outer boundary
 *        of the extracted mesh.
 * @details For every cell `c` whose entry in `OctreeHybridMesherData::mCellColor`
 *          equals `cell_color`, and for every one of its 6 local faces `f`
 *          (`FACE_FIDC`/Hexahedra3D8 ordering, see
 *          `OctreeHybridMeshUtility::ExtractBoundaryFaces`), the face-neighbour cell
 *          across `f` is looked up via `OctreeHybridMeshUtility::ComputeCellFaceNeighbors`.
 *          When `f` has **no** neighbour (it lies on the outer boundary of the
 *          extracted hex mesh, i.e.\ it is not shared with any other cell),
 *          `OctreeHybridMesherData::mCellFaceColor[c][f]` is set to `color`.
 *
 *          Unlike `OctreeHybridColorCellFacesBetweenColors`, this stage does not
 *          compare the neighbour's colour: a face is "outer" purely because it has
 *          no neighbouring hex at all. This is the octree-hybrid analogue of the
 *          outer-face detection performed by the voxel mesher's
 *          `ColorOuterFacesOfCellsWithColors` when restricted to the global domain
 *          boundary.
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised yet (or
 *          its size differs from `mCells.size()`), it is resized and filled with 0
 *          before the colour comparisons. Likewise, if
 *          `OctreeHybridMesherData::mCellFaceColor` has not been initialised yet (or
 *          its size differs from `mCells.size()`), it is resized and filled with
 *          `{0,0,0,0,0,0}` per cell before any face colour is written.
 *
 * ### Typical pipeline position
 * Run after the coloring stages that produce `mCellColor` (e.g.
 * `OctreeHybridClassifyCellsInsideOutside`), so that `cell_color` already identifies
 * the region whose outer faces should be tagged — typically for boundary-condition
 * generation via `GenerateOctreeHybridQuadrilateralConditionsWithFaceColor`.
 *
 * ### Usage
 * @code{.json}
 * {
 *     "type"       : "OctreeHybridColorOuterCellFaces",
 *     "color"      : 3,
 *     "cell_color" : 1
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridColorCellFacesBetweenColors
 * @see OctreeHybridMeshUtility::ComputeCellFaceNeighbors
 * @see OctreeHybridMesherData::mCellFaceColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorOuterCellFaces : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorOuterCellFaces() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorOuterCellFaces(OctreeHybridColorOuterCellFaces const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours, for every cell with colour `cell_color`, the local faces that
     *        have no neighbouring cell (outer boundary of the extracted mesh).
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
     *     "type"            : "OctreeHybridColorOuterCellFaces",
     *     "model_part_name" : "Undefined",
     *     "color"           : -1,
     *     "cell_color"      : -1
     * }
     * @endcode
     * `"model_part_name"` is unused by this stage; it is kept for schema
     * consistency with other coloring components and is auto-filled by the
     * modeler dispatch loop when absent.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorOuterCellFaces.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorOuterCellFaces)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorOuterCellFaces.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorOuterCellFaces)

    ///@}
};

///@}

} // namespace Kratos
