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

// Project includes
#include "modeler/entity_generation/octree_hybrid_mesher_entity_generation.h"

namespace Kratos
{

///@name Kratos Classes
///@{

/**
 * @class GenerateHybridOctreeTetrahedraElementsWithCellColor
 * @ingroup KratosCore
 * @brief Entity-generation stage that emits tetrahedral elements from the dual/primal hex mesh
 *        by decomposing each selected hex cell into 6 Freudenthal tetrahedra.
 * @details Each hex cell whose `mCellColor` entry matches `color` is decomposed into 6
 * tetrahedra using the Freudenthal–Kuhn decomposition along the main diagonal (local nodes 0→6).
 * This implements the BCC-lattice based tetrahedral pattern described in TACS1de1.pdf §3.3.3
 * for the dual-hex representation: dual-hex cell centres form BCC body positions, and the
 * diagonal decomposition recovers the body-centred tetrahedral tiling with a minimum dihedral
 * angle of 45 °.
 *
 * Because `OctreeHybridMeshUtility::ExtractDualHexMesh` assigns local node indices via the
 * fixed `idTransform` map, adjacent hexes always place the same physical node at the same
 * logical corner, so the shared-face diagonal is identical in both hexes — the tet mesh is
 * conforming with no hanging nodes.
 *
 * Node de-duplication and batch insertion follow the same pattern as
 * @ref GenerateHybridOctreeHexahedraElementsWithCellColor.
 *
 * ### Parameters schema
 * | Key                    | Type         | Default                          | Description                                          |
 * |------------------------|--------------|----------------------------------|------------------------------------------------------|
 * | `type`                 | string       | `"GenerateHybridOctreeTetrahedraElementsWithCellColor"` | Registry lookup key.              |
 * | `model_part_name`      | string       | `"Undefined"`                    | ModelPart to create elements in.                     |
 * | `color`                | int          | `1`                              | Cell color to select (1 = inside, 0 = outside).      |
 * | `properties_id`        | int          | `1`                              | Properties ID assigned to each element.              |
 * | `generated_entity`     | string       | `"Element3D4N"`                  | Registered tetrahedral element type to instantiate.  |
 * | `tag_refinement_level` | bool         | `true`                           | Tag each tet element with `REFINEMENT_LEVEL` of its parent hex. |
 * | `initial_node_id`      | int          | `0`                              | Explicit first node ID; `0` = auto.                  |
 * | `initial_element_id`   | int          | `0`                              | Explicit first element ID; `0` = auto.               |
 * | `echo_level`           | int          | `0`                              | Verbosity of `KRATOS_INFO` logging (`0` = silent).   |
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see GenerateHybridOctreeHexahedraElementsWithCellColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) GenerateHybridOctreeTetrahedraElementsWithCellColor
    : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    GenerateHybridOctreeTetrahedraElementsWithCellColor() = default;
    GenerateHybridOctreeTetrahedraElementsWithCellColor(GenerateHybridOctreeTetrahedraElementsWithCellColor const&) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates tetrahedral elements by decomposing each colour-matched hex into 6 tets.
     * @param rModeler             Owning modeler; provides data and entity-ID counters.
     * @param GenerationParameters Validated JSON parameters (see schema above).
     */
    void Generate(OctreeHybridMeshGeneratorModeler& rModeler, Parameters GenerationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics",
                                  OctreeHybridMesherEntityGeneration,
                                  GenerateHybridOctreeTetrahedraElementsWithCellColor)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All",
                                  OctreeHybridMesherEntityGeneration,
                                  GenerateHybridOctreeTetrahedraElementsWithCellColor)

    ///@}
};

///@}

} // namespace Kratos
