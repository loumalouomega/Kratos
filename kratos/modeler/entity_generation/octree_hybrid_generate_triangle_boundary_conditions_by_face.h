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
 * @class OctreeHybridGenerateTriangleBoundaryConditionsByFace
 * @ingroup KratosCore
 * @brief Entity-generation stage that creates triangular boundary conditions on the exterior
 *        faces of the coloured hex mesh.
 * @details This generator uses `OctreeHybridMeshUtility::ExtractBoundaryFaces` to obtain the
 * outer quad faces of the colour-filtered cell set, then splits each quad `{n0, n1, n2, n3}`
 * into two triangles along the `(n0, n2)` diagonal:
 *   - triangle 1: `{n0, n1, n2}`
 *   - triangle 2: `{n0, n2, n3}`
 *
 * This diagonal is consistent with the `(0, 6)` main-diagonal Freudenthal decomposition used
 * by `OctreeHybridGenerateTetrahedraByCellColor`, so the boundary triangles are faces of the
 * interior tetrahedra — the combined tet + triangle mesh is conforming.
 *
 * Node de-duplication and batch insertion follow the same pattern as
 * @ref OctreeHybridGenerateBoundaryConditionsByFace.
 *
 * ### Prerequisite
 * The hex-extraction step must have run first so that `OctreeHybridMesherData::mCells` and
 * `mCellColor` are populated.
 *
 * ### Parameters schema
 * | Key                | Type         | Default                               | Description                               |
 * |--------------------|--------------|---------------------------------------|-------------------------------------------|
 * | `type`             | string       | `"OctreeHybridGenerateTriangleBoundaryConditionsByFace"` | Registry lookup key.   |
 * | `model_part_name`  | string       | `"Undefined"`                         | ModelPart to create conditions in.        |
 * | `color`            | int          | `1`                                   | Cell color (inside label).                |
 * | `properties_id`    | int          | `1`                                   | Properties ID for each condition.         |
 * | `generated_entity` | string       | `"SurfaceCondition3D3N"`              | Registered triangular condition type.     |
 *
 * @see OctreeHybridMesherEntityGeneration
 * @see OctreeHybridGenerateBoundaryConditionsByFace
 * @see OctreeHybridGenerateTetrahedraByCellColor
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridGenerateTriangleBoundaryConditionsByFace
    : public OctreeHybridMesherEntityGeneration
{
public:
    ///@name Life Cycle
    ///@{

    OctreeHybridGenerateTriangleBoundaryConditionsByFace() = default;
    OctreeHybridGenerateTriangleBoundaryConditionsByFace(
        OctreeHybridGenerateTriangleBoundaryConditionsByFace const&) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Generates two triangular conditions per boundary quad of the coloured hex mesh.
     * @param rModeler             Owning modeler; provides data and entity-ID counters.
     * @param GenerationParameters Validated JSON parameters (see schema above).
     */
    void Generate(OctreeHybridMesherModeler& rModeler, Parameters GenerationParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.KratosMultiphysics",
                                  OctreeHybridMesherEntityGeneration,
                                  OctreeHybridGenerateTriangleBoundaryConditionsByFace)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherEntityGeneration.All",
                                  OctreeHybridMesherEntityGeneration,
                                  OctreeHybridGenerateTriangleBoundaryConditionsByFace)

    ///@}
};

///@}

} // namespace Kratos
