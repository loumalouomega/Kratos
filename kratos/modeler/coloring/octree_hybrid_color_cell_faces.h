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
 * @class OctreeHybridColorCellFaces
 * @ingroup KratosCore
 * @brief Colouring stage that labels hex-cell faces based on their distance to an
 *        intersecting surface, or unconditionally as a manual override.
 * @details For every cell `c` and every one of its 6 local faces `f` (`FACE_FIDC`/
 *          Hexahedra3D8 ordering, see `OctreeHybridMeshUtility::ExtractBoundaryFaces`),
 *          the centroid of the face's 4 corner nodes is computed.
 *
 *          ### Optional bounding-box pre-filter
 *          When `bounding_box.min_point` and `bounding_box.max_point` are both
 *          3-vectors, only faces whose centroid lies within this axis-aligned box are
 *          candidates; faces outside the box are left unchanged. When either array is
 *          empty (the default), every face is a candidate.
 *
 *          ### Automatic strategy (`input_entities` non-empty)
 *          A triangle soup is built from `model_part_name`'s entities (selected by
 *          `input_entities` as `"geometries"`, `"elements"`, or `"conditions"`), and
 *          the signed distance of every candidate face centroid to that soup is
 *          computed via `OctreeHybridMeshUtility::ComputeNodeSignedDistance`. A
 *          candidate face is coloured with `color` when the absolute value of its
 *          signed distance is at most `tolerance`, i.e.\ the face lies on (or
 *          numerically on) the surface.
 *
 *          ### Manual strategy (`input_entities` empty, the default)
 *          No geometry test is performed: every candidate face (after the optional
 *          bounding-box pre-filter) is coloured with `color` directly.
 *
 *          If `OctreeHybridMesherData::mCellFaceColor` has not been initialised yet
 *          (or its size differs from `mCells.size()`), it is resized and filled with
 *          `{0,0,0,0,0,0}` per cell before any face colour is written.
 *
 * ### Usage
 * Automatic, distance-based:
 * @code{.json}
 * {
 *     "type"            : "OctreeHybridColorCellFaces",
 *     "model_part_name" : "Skin",
 *     "color"           : 5,
 *     "input_entities"  : "geometries",
 *     "tolerance"       : 1.0e-6
 * }
 * @endcode
 * Manual, restricted to a region:
 * @code{.json}
 * {
 *     "type"            : "OctreeHybridColorCellFaces",
 *     "color"           : 5,
 *     "input_entities"  : "",
 *     "bounding_box"  : {
 *         "min_point" : [0.0, 0.0, 0.95],
 *         "max_point" : [1.0, 1.0, 1.0]
 *     }
 * }
 * @endcode
 *
 * @see OctreeHybridMesherColoring
 * @see OctreeHybridColorOuterCellFaces
 * @see OctreeHybridColorCellsFacesInBoundingBox
 * @see OctreeHybridMeshUtility::ComputeNodeSignedDistance
 * @see OctreeHybridMesherData::mCellFaceColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridColorCellFaces : public OctreeHybridMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridColorCellFaces() = default;

    /**
     * @brief Copy constructor (no-op — this class is stateless).
     * @param rOther Source instance; no data is copied.
     */
    OctreeHybridColorCellFaces(OctreeHybridColorCellFaces const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Colours hex-cell faces by distance to an intersecting surface
     *        (automatic), or unconditionally (manual override).
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
     *     "type"            : "OctreeHybridColorCellFaces",
     *     "model_part_name" : "Undefined",
     *     "color"           : -1,
     *     "input_entities"  : "",
     *     "tolerance"       : 1.0e-12,
     *     "bounding_box"  : {
     *         "min_point" : [],
     *         "max_point" : []
     *     }
     * }
     * @endcode
     * `"input_entities" == ""` selects the manual strategy (no geometry test);
     * `"geometries"`, `"elements"`, or `"conditions"` select the automatic,
     * distance-based strategy. Any other non-empty value throws.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this prototype at "OctreeHybridMesherColoring.KratosMultiphysics.OctreeHybridColorCellFaces.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.KratosMultiphysics", OctreeHybridMesherColoring, OctreeHybridColorCellFaces)
    /// Registers this prototype at "OctreeHybridMesherColoring.All.OctreeHybridColorCellFaces.Prototype".
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherColoring.All", OctreeHybridMesherColoring, OctreeHybridColorCellFaces)

    ///@}
};

///@}

} // namespace Kratos
