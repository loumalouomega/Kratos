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
#include <array>
#include <vector>

// External includes

// Project includes
#include "modeler/coloring/octree_hybrid_mesher_coloring.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridColorConnectedCellsInTouch
 * @ingroup KratosCore
 * @brief Colouring stage that flood-fills all hex cells connected to the input-geometry
 *        surface and carrying a specified seed colour.
 * @details The operation works in two phases:
 *
 *          **Phase 1 — seed detection.**  For each geometry in the input ModelPart
 *          (selected via `input_entities`), cells whose AABB-vs-geometry intersection
 *          test passes and whose current colour equals `cell_color` are added as
 *          flood-fill seeds.
 *
 *          **Phase 2 — BFS flood-fill.**  A stack-based depth-first search traverses
 *          face-adjacent neighbours: two cells are adjacent when they share the same
 *          4-node face (identified by a sorted node-index 4-tuple).  Every cell popped
 *          from the stack whose colour still equals `cell_color` is coloured with
 *          `color` and its neighbours are pushed.
 *
 *          ### Cell-adjacency convention
 *          The six faces of a hex stored in `mCells[i]` are identified by the following
 *          local-node-index sets (0-based within the 8-entry connectivity record):
 *          `{0,1,2,3}`, `{4,5,6,7}`, `{0,1,5,4}`, `{1,2,6,5}`, `{2,3,7,6}`,
 *          `{3,0,4,7}`.  Two cells sharing the same sorted 4-tuple of global node
 *          indices are face-adjacent.
 *
 *          If `OctreeHybridMesherData::mCellColor` has not been initialised (or its
 *          size differs from `mCells.size()`), it is resized and filled with 0 before
 *          any colour is written.
 *
 * ### Typical use-case
 * After `OctreeHybridClassifyCellsInsideOutside` assigns 1 (inside) / 0 (outside),
 * run this operation with `cell_color=0` to label the entire connected exterior region
 * that touches the surface with a distinct colour (e.g. 2), separating it from any
 * isolated voids or inner pockets that also received colour 0.
 *
 * ### Usage
 * @code{.json}
 * {
 *     "type"            : "OctreeHybridColorConnectedCellsInTouch",
 *     "model_part_name" : "MySurface",
 *     "color"           : 2,
 *     "cell_color"      : 0,
 *     "input_entities"  : "geometries"
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridColorCellsInTouch
 * @see OctreeHybridClassifyCellsInsideOutside
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorConnectedCellsInTouch
    : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorConnectedCellsInTouch() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorConnectedCellsInTouch(OctreeHybridColorConnectedCellsInTouch const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Flood-fills all cells connected (face-adjacent) to the input geometry
     *        surface through cells with colour `cell_color`.
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
     *     "type"            : "OctreeHybridColorConnectedCellsInTouch",
     *     "model_part_name" : "",
     *     "color"           : 1,
     *     "cell_color"      : 0,
     *     "input_entities"  : "geometries"
     * }
     * @endcode
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorConnectedCellsInTouch.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorConnectedCellsInTouch)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorConnectedCellsInTouch.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorConnectedCellsInTouch)

    ///@}
};

///@}

} // namespace Kratos
