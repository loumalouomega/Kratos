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

// System includes
#include <cmath>

// External includes

// Project includes
#include "includes/element.h"
#include "modeler/operation/report_mesh_quality.h"
#include "modeler/octree_mesher_modeler.h"
#include "utilities/octree_hybrid_mesh_utility.h"

namespace Kratos {

/**
 * @brief Returns the default parameter schema for @ref ReportMeshQuality.
 * @details The schema contains:
 * - `"type"` — the Registry lookup key, fixed to `"ReportMeshQuality"`.
 * - `"model_part_name"` — name of the ModelPart to evaluate; must be set by the caller.
 * @return Parameters with both keys and their default values.
 */
const Parameters ReportMeshQuality::GetDefaultParameters() const
{
    return Parameters(R"({
        "type"             : "ReportMeshQuality",
        "model_part_name"  : "Undefined"
    })");
}

/**
 * @brief Computes per-element minimum scaled Jacobian and logs aggregate statistics.
 * @details Implementation notes:
 * 1. The target ModelPart is retrieved from the modeler's Model using the
 *    `"model_part_name"` parameter.
 * 2. If the part has no elements a short info message is emitted and the method returns.
 * 3. For every element whose geometry has exactly 8 nodes the 8 corner coordinates are
 *    copied into a plain `double[8][3]` array and passed to
 *    `OctreeHybridMeshUtility::ScaledJacobianMin`, which evaluates the scaled Jacobian
 *    at all 8 corners and the body centre using the `SJ_ADJ` edge-triple table, then
 *    returns the minimum value.
 * 4. Global minimum (`sj_min`), running sum (`sj_sum`), and inverted-element count
 *    (`n_inverted`, i.e. elements with scaled Jacobian <= 0) are accumulated.
 * 5. After the loop the statistics — minimum, mean, and inverted count with percentage
 *    — are logged via `KRATOS_INFO("ReportMeshQuality")`.
 *
 * Elements with a number of nodes other than 8 are silently skipped so that mixed
 * ModelParts (e.g. containing surfaces or lower-order elements) do not cause errors.
 *
 * @param rModeler            Provides access to the Model that contains the target
 *                            ModelPart.  Not modified.
 * @param OperationParameters Validated JSON parameters; must contain `"model_part_name"`
 *                            with the name of an existing ModelPart.
 */
void ReportMeshQuality::Execute(OctreeMesherModeler& rModeler, Parameters OperationParameters) const
{
    ModelPart& r_mp = rModeler.GetModel().GetModelPart(
        OperationParameters["model_part_name"].GetString());

    const int n = static_cast<int>(r_mp.NumberOfElements());
    if (n == 0) {
        KRATOS_INFO("ReportMeshQuality")
            << r_mp.FullName() << ": no elements." << std::endl;
        return;
    }

    // Compute min scaled Jacobian for each hex using the engine's SJ_ADJ convention.
    // The SJ_ADJ table uses the same corner ordering as Hexahedra3D8.
    double sj_min = 1.0e30, sj_sum = 0.0;
    int n_inverted = 0;

    for (const auto& r_el : r_mp.Elements()) {
        const auto& r_geom = r_el.GetGeometry();
        if (r_geom.size() != 8) continue;

        double p[8][3];
        for (int k = 0; k < 8; ++k) {
            p[k][0] = r_geom[k].X();
            p[k][1] = r_geom[k].Y();
            p[k][2] = r_geom[k].Z();
        }
        const double sj = OctreeHybridMeshUtility::ScaledJacobianMin(p);
        sj_sum += sj;
        if (sj < sj_min) sj_min = sj;
        if (sj <= 0.0) ++n_inverted;
    }

    const double sj_mean = sj_sum / n;

    KRATOS_INFO("ReportMeshQuality")
        << r_mp.FullName() << " (" << n << " hexes): "
        << "minSJ=" << sj_min
        << "  meanSJ=" << sj_mean
        << "  inverted=" << n_inverted
        << " (" << 100.0 * n_inverted / n << "%)" << std::endl;
}

} // namespace Kratos
