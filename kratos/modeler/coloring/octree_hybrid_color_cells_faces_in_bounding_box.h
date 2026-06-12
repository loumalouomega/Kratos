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
 * @class OctreeHybridColorCellsFacesInBoundingBox
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex-cell faces whose centroid lies inside (or
 *        outside) an axis-aligned bounding box.
 * @details For every cell `c` and every one of its 6 local faces `f` (`FACE_FIDC`/
 *          Hexahedra3D8 ordering, see `OctreeHybridMeshUtility::ExtractBoundaryFaces`),
 *          the centroid of the face's 4 corner nodes is tested against an
 *          axis-aligned bounding box. The box is either given explicitly via
 *          `min_point`/`max_point`, or — when either array does not have exactly 3
 *          components — computed as the AABB of the nodes of the ModelPart
 *          `model_part_name`.
 *
 *          A face centroid is "inside" the box when, for every axis `d`,
 *          `min_point[d] <= centroid[d] <= max_point[d]`. When `inside_bounding_box`
 *          is `true`, faces whose centroid is inside the box receive `color`; when
 *          `false`, faces whose centroid is **outside** the box receive `color`.
 *
 *          If `OctreeHybridMesherData::mCellFaceColor` has not been initialised yet
 *          (or its size differs from `mCells.size()`), it is resized and filled with
 *          `{0,0,0,0,0,0}` per cell before any face colour is written.
 *
 * ### Usage
 * Add the following block to the `"coloring_settings_list"` in the modeler JSON:
 * @code{.json}
 * {
 *     "type"                : "OctreeHybridColorCellsFacesInBoundingBox",
 *     "color"               : 3,
 *     "min_point"           : [0.0, 0.0, 0.45],
 *     "max_point"           : [1.0, 1.0, 0.55],
 *     "inside_bounding_box" : true
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridColorCellsInBoundingBox
 * @see OctreeHybridColorOuterCellFaces
 * @see OctreeHybridMesherData::mCellFaceColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellsFacesInBoundingBox : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellsFacesInBoundingBox() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellsFacesInBoundingBox(OctreeHybridColorCellsFacesInBoundingBox const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours every hex-cell face whose centroid is inside (or outside) the
     *        requested bounding box.
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
     *     "type"                : "OctreeHybridColorCellsFacesInBoundingBox",
     *     "color"               : -1,
     *     "model_part_name"     : "Undefined",
     *     "min_point"           : [],
     *     "max_point"           : [],
     *     "inside_bounding_box" : false
     * }
     * @endcode
     * `"min_point"`/`"max_point"` take precedence when both have exactly 3
     * components; otherwise the bounding box is computed from the nodes of
     * `"model_part_name"`.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellsFacesInBoundingBox.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellsFacesInBoundingBox)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellsFacesInBoundingBox.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellsFacesInBoundingBox)

    ///@}
};

///@}

} // namespace Kratos
