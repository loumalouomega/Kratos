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
#include "modeler/coloring/octree_mesher_coloring.h"

namespace Kratos {

///@name Kratos Classes
///@{

/**
 * @class ClassifyCellsInsideOutside
 * @ingroup KratosCore
 * @brief Colours the extracted hex cells inside(1)/outside(0) the input surface.
 * @details Uses @ref OctreeHybridMeshUtility::ClassifyInsideOutside (the carve decision
 * without removal); a later hex generator emits only inside cells.  When the mesh was
 * already projected (carved) during generation, every remaining cell is inside.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) ClassifyCellsInsideOutside : public OctreeMesherColoring
{
public:
    ///@name Life Cycle
    ///@{

    ClassifyCellsInsideOutside() = default;

    ClassifyCellsInsideOutside(ClassifyCellsInsideOutside const&) {}

    ///@}
    ///@name Operations
    ///@{

    void Apply(OctreeMesherModeler& rModeler, Parameters ColoringParameters) const override;

    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.KratosMultiphysics", OctreeMesherColoring, ClassifyCellsInsideOutside)
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeMesherColoring.All", OctreeMesherColoring, ClassifyCellsInsideOutside)

    ///@}
};

///@}

} // namespace Kratos
