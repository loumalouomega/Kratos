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
#include <algorithm>
#include <cmath>
#include <numeric>

// External includes

// Project includes
#include "integration/line_gauss_legendre_integration_points.h"
#include "integration/triangle_gauss_legendre_integration_points.h"
#include "utilities/atomic_utilities.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"

// Application includes
#include "custom_utilities/fluid_auxiliary_utilities.h"
#include "pde_constrained_projection_utility.h"

namespace Kratos
{

namespace
{

/// Face quadrature (barycentric coordinates of the face nodes and weights normalized to 1)
template<std::size_t TDim>
using FaceQuadratureType = std::vector<std::pair<std::array<double, TDim>, double>>;

template<std::size_t TDim>
const FaceQuadratureType<TDim>& GetFaceQuadrature();

template<>
const FaceQuadratureType<2>& GetFaceQuadrature<2>()
{
    static const FaceQuadratureType<2> quadrature = []() {
        FaceQuadratureType<2> points;
        for (const auto& r_point : LineGaussLegendreIntegrationPoints3::IntegrationPoints()) {
            const double xi = r_point.X();
            points.push_back({{0.5 * (1.0 - xi), 0.5 * (1.0 + xi)}, 0.5 * r_point.Weight()});
        }
        return points;
    }();
    return quadrature;
}

template<>
const FaceQuadratureType<3>& GetFaceQuadrature<3>()
{
    static const FaceQuadratureType<3> quadrature = []() {
        FaceQuadratureType<3> points;
        for (const auto& r_point : TriangleGaussLegendreIntegrationPoints4::IntegrationPoints()) {
            const double xi = r_point.X();
            const double eta = r_point.Y();
            points.push_back({{1.0 - xi - eta, xi, eta}, 2.0 * r_point.Weight()});
        }
        return points;
    }();
    return quadrature;
}

} // namespace

template<std::size_t TDim>
PdeConstrainedProjectionUtility<TDim>::PdeConstrainedProjectionUtility(const FacetTableType& rFacetTable)
    : mrFacetTable(rFacetTable)
{
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename PdeConstrainedProjectionUtility<TDim>::ProjectionInfo PdeConstrainedProjectionUtility<TDim>::Project(
    const ParticleContainerType& rParticles,
    const NodalVectorValuesType& rFluxVelocity,
    const std::vector<FaceTypesType>& rFaceTypes,
    const std::vector<double>& rNodalDistance,
    const ProjectionSettings& rSettings,
    FacetLinearSystem& rFacetSystem,
    std::vector<ElementalValuesType>& rIndicator
    )
{
    KRATOS_TRY

    const std::size_t n_elements = mrFacetTable.NumberOfElements();
    const std::size_t n_dofs = mrFacetTable.NumberOfFacetDofs();
    const auto& r_data_communicator = mrFacetTable.GetDataCommunicator();
    KRATOS_ERROR_IF(rIndicator.size() != n_elements) << "Wrong elemental indicator size." << std::endl;
    KRATOS_ERROR_IF(rFaceTypes.size() != n_elements) << "Wrong face types size." << std::endl;
    KRATOS_ERROR_IF(rSettings.DeltaTime <= 0.0) << "Time step must be positive." << std::endl;
    KRATOS_ERROR_IF(rSettings.Beta <= 0.0) << "Facet penalty beta must be positive." << std::endl;
    KRATOS_ERROR_IF(rSettings.Zeta < 0.0) << "Gradient penalty zeta must be non-negative." << std::endl;

    const std::vector<ElementalValuesType> previous_indicator(rIndicator);
    double facet_system_residual = 0.0;

    struct TLS
    {
        LocalSystem Local;
        CondensedSystem Condensed;
        array_1d<double, MaxFacetDofs> LocalFacetValues;
    };

    // Assemble and solve the condensed facet system (global numbering of the facet unknowns)
    if (n_dofs > 0) {
        if (!rFacetSystem.IsInitialized()) {
            std::vector<std::vector<std::size_t>> element_dofs;
            mrFacetTable.ComputeElementFacetDofs(element_dofs);
            rFacetSystem.Initialize(mrFacetTable.NumberOfOwnedFacetDofs(), mrFacetTable.FirstOwnedFacetDof(), n_dofs, element_dofs);
        }
        rFacetSystem.BeginAssemble();
        IndexPartition<std::size_t>(n_elements).for_each(TLS(), [&](const std::size_t iElement, TLS& rTLS) {
            BuildLocalSystem(mrFacetTable, rParticles, iElement, previous_indicator[iElement], rFluxVelocity, rFaceTypes[iElement], rNodalDistance, rSettings, rTLS.Local);
            CondenseLocalSystem(rTLS.Local, rTLS.Condensed);
            if (rTLS.Local.NumberOfFacetDofs > 0) {
                rFacetSystem.Assemble(rTLS.Local.NumberOfFacetDofs, rTLS.Local.FacetDofs.data(), &rTLS.Condensed.S(0, 0), MaxFacetDofs, &rTLS.Condensed.RHS[0]);
            }
        });
        facet_system_residual = rFacetSystem.Solve();
    }

    // Recover the element unknowns and compute the diagnostics
    using ReductionType = CombinedReduction<SumReduction<double>, SumReduction<double>, SumReduction<double>, MaxReduction<double>, MinReduction<double>, MaxReduction<double>>;
    const auto [previous_integral, integral, outflow, max_residual, min_value, max_value] = IndexPartition<std::size_t>(n_elements).for_each<ReductionType>(TLS(), [&](const std::size_t iElement, TLS& rTLS) {
        BuildLocalSystem(mrFacetTable, rParticles, iElement, previous_indicator[iElement], rFluxVelocity, rFaceTypes[iElement], rNodalDistance, rSettings, rTLS.Local);
        CondenseLocalSystem(rTLS.Local, rTLS.Condensed);
        GatherLocalFacetValues(rFacetSystem, rTLS.Local, rTLS.LocalFacetValues);
        double lagrange_multiplier;
        auto& r_indicator = rIndicator[iElement];
        RecoverLocalSolution(rTLS.Local, rTLS.Condensed, rTLS.LocalFacetValues, r_indicator, lagrange_multiplier);

        const double volume = mrFacetTable.GetElementData(iElement).Volume;
        const double element_previous_integral = volume * std::accumulate(previous_indicator[iElement].begin(), previous_indicator[iElement].end(), 0.0) / static_cast<double>(NumNodes);
        const double element_integral = volume * std::accumulate(r_indicator.begin(), r_indicator.end(), 0.0) / static_cast<double>(NumNodes);
        const double element_outflow = inner_prod(rTLS.Local.OutflowCoefficients, r_indicator) + rTLS.Local.PrescribedBoundaryFlux;
        const double residual = std::abs(ComputeCellResidual(rTLS.Local, r_indicator, rTLS.LocalFacetValues)) / volume;
        const double element_min = *std::min_element(r_indicator.begin(), r_indicator.end());
        const double element_max = *std::max_element(r_indicator.begin(), r_indicator.end());
        return std::make_tuple(element_previous_integral, element_integral, element_outflow, residual, element_min, element_max);
    });

    ProjectionInfo info;
    info.PreviousIntegral = r_data_communicator.SumAll(previous_integral);
    info.Integral = r_data_communicator.SumAll(integral);
    info.NetBoundaryOutflow = r_data_communicator.SumAll(outflow);
    info.MaxCellResidual = r_data_communicator.MaxAll(max_residual);
    info.MinValue = r_data_communicator.MinAll(min_value);
    info.MaxValue = r_data_communicator.MaxAll(max_value);
    info.FacetSystemResidual = facet_system_residual;
    return info;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::BuildLocalSystem(
    const FacetTableType& rFacetTable,
    const ParticleContainerType& rParticles,
    const std::size_t ElementIndex,
    const ElementalValuesType& rPreviousIndicator,
    const NodalVectorValuesType& rFluxVelocity,
    const FaceTypesType& rFaceTypes,
    const std::vector<double>& rNodalDistance,
    const ProjectionSettings& rSettings,
    LocalSystem& rLocalSystem
    )
{
    const auto& r_data = rFacetTable.GetElementData(ElementIndex);
    const double volume = r_data.Volume;
    const double dt = rSettings.DeltaTime;
    const double beta = rSettings.Beta * std::pow(r_data.Size, 1.0 - static_cast<double>(TDim));
    const double zeta = rSettings.Zeta * std::pow(r_data.Size, 2.0 - static_cast<double>(TDim));

    auto& r_K = rLocalSystem.K;
    auto& r_C = rLocalSystem.C;
    auto& r_B = rLocalSystem.B;
    auto& r_R = rLocalSystem.R;
    auto& r_outflow = rLocalSystem.OutflowCoefficients;
    noalias(r_K) = ZeroMatrix(LocalSize, LocalSize);
    noalias(r_C) = ZeroMatrix(LocalSize, MaxFacetDofs);
    noalias(r_B) = ZeroMatrix(MaxFacetDofs, MaxFacetDofs);
    noalias(r_R) = ZeroVector(LocalSize);
    noalias(r_outflow) = ZeroVector(NumNodes);
    rLocalSystem.PrescribedBoundaryFlux = 0.0;
    rLocalSystem.NumberOfFacetDofs = 0;

    // Particle least-squares terms
    typename FacetTableType::ShapeFunctionsType N;
    for (auto it_particle = rParticles.ParticlesBegin(ElementIndex); it_particle != rParticles.ParticlesEnd(ElementIndex); ++it_particle) {
        rFacetTable.ShapeFunctions(ElementIndex, it_particle->Coordinates, N);
        for (std::size_t a = 0; a < NumNodes; ++a) {
            r_R[a] += N[a] * it_particle->Indicator;
            for (std::size_t b = 0; b < NumNodes; ++b) {
                r_K(a, b) += N[a] * N[b];
            }
        }
    }

    // Gradient penalty
    for (std::size_t a = 0; a < NumNodes; ++a) {
        for (std::size_t b = 0; b < NumNodes; ++b) {
            double grad_product = 0.0;
            for (std::size_t d = 0; d < TDim; ++d) {
                grad_product += r_data.DN_DX(a, d) * r_data.DN_DX(b, d);
            }
            r_K(a, b) += zeta * volume * grad_product;
        }
    }

    // Constraint: int_K c + dt int_{dK} (u.n) ctilde = int_K c*
    double previous_integral = 0.0;
    for (std::size_t a = 0; a < NumNodes; ++a) {
        previous_integral += rPreviousIndicator[a];
    }
    r_R[NumNodes] = volume * previous_integral / static_cast<double>(NumNodes);

    const auto& r_quadrature = GetFaceQuadrature<TDim>();
    std::array<std::size_t, TDim> face_nodes;
    for (std::size_t i_face = 0; i_face < NumFaces; ++i_face) {
        std::size_t aux = 0;
        for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
            if (i_node != i_face) {
                face_nodes[aux++] = i_node;
            }
        }
        const auto& r_area_normal = r_data.FaceAreaNormals[i_face];
        const double area = norm_2(r_area_normal);

        if (r_data.Facets[i_face] != FacetTableType::Invalid) {
            // Interior facet: flux through the facet unknown and facet penalty
            std::array<std::size_t, TDim> slots;
            for (std::size_t j = 0; j < TDim; ++j) {
                slots[j] = rLocalSystem.NumberOfFacetDofs++;
                rLocalSystem.FacetDofs[slots[j]] = rFacetTable.FacetDof(ElementIndex, i_face, face_nodes[j]);
            }
            for (std::size_t i = 0; i < TDim; ++i) {
                const std::size_t a = face_nodes[i];
                for (std::size_t j = 0; j < TDim; ++j) {
                    const std::size_t b = face_nodes[j];
                    const double face_mass = FacetTableType::FaceMassCoefficient(area, i == j);
                    r_K(a, b) += beta * face_mass;
                    r_C(a, slots[j]) = -beta * face_mass;
                    r_B(slots[i], slots[j]) = beta * face_mass;
                }
            }
            for (std::size_t j = 0; j < TDim; ++j) {
                double flux = 0.0;
                for (std::size_t i = 0; i < TDim; ++i) {
                    const std::size_t node = r_data.NodeIndices[face_nodes[i]];
                    flux += inner_prod(rFluxVelocity[node], r_area_normal) * FacetTableType::FaceMassCoefficient(1.0, i == j);
                }
                r_C(NumNodes, slots[j]) = dt * flux;
            }
        } else if (rFaceTypes[i_face] != ParticleContainerType::FaceType::Wall) {
            // Open boundary face: upwind flux. The incoming fluid is the level set indicator at the inlets and air elsewhere (unless
            // requested); the outgoing fluid is the level set indicator (default) or the interior trace
            const bool is_inlet = rFaceTypes[i_face] == ParticleContainerType::FaceType::Inlet;
            std::array<double, TDim> normal_velocity;
            std::array<double, TDim> distance;
            for (std::size_t i = 0; i < TDim; ++i) {
                const std::size_t node = r_data.NodeIndices[face_nodes[i]];
                normal_velocity[i] = inner_prod(rFluxVelocity[node], r_area_normal) / area;
                distance[i] = rNodalDistance[node];
            }
            for (const auto& r_point : r_quadrature) {
                const auto& r_lambda = r_point.first;
                const double weight = dt * r_point.second * area;
                double u_n = 0.0;
                for (std::size_t i = 0; i < TDim; ++i) {
                    u_n += r_lambda[i] * normal_velocity[i];
                }
                if (u_n > 0.0 && !rSettings.OutflowFromLevelSet) {
                    for (std::size_t i = 0; i < TDim; ++i) {
                        r_outflow[face_nodes[i]] += weight * u_n * r_lambda[i];
                    }
                } else if (u_n < 0.0 && !is_inlet && !rSettings.BackflowFromLevelSet) {
                    // Backflow through an open face that is not an inlet: air comes in, there is no phase flux
                    continue;
                } else {
                    double point_distance = 0.0;
                    for (std::size_t i = 0; i < TDim; ++i) {
                        point_distance += r_lambda[i] * distance[i];
                    }
                    const double indicator = point_distance < 0.0 ? 1.0 : 0.0;
                    rLocalSystem.PrescribedBoundaryFlux += weight * u_n * indicator;
                }
            }
        }
    }

    for (std::size_t a = 0; a < NumNodes; ++a) {
        const double g_a = volume / static_cast<double>(NumNodes) + r_outflow[a];
        r_K(a, NumNodes) = g_a;
        r_K(NumNodes, a) = g_a;
    }
    r_R[NumNodes] -= rLocalSystem.PrescribedBoundaryFlux;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::CondenseLocalSystem(
    const LocalSystem& rLocalSystem,
    CondensedSystem& rCondensedSystem
    )
{
    // Solve K [X_C, X_R] = [C, R]
    BoundedMatrix<double, LocalSize, LocalSize> K = rLocalSystem.K;
    BoundedMatrix<double, LocalSize, MaxFacetDofs + 1> rhs;
    for (std::size_t i = 0; i < LocalSize; ++i) {
        for (std::size_t j = 0; j < MaxFacetDofs; ++j) {
            rhs(i, j) = rLocalSystem.C(i, j);
        }
        rhs(i, MaxFacetDofs) = rLocalSystem.R[i];
    }
    SolveDenseSystem<LocalSize, MaxFacetDofs + 1>(K, rhs);
    for (std::size_t i = 0; i < LocalSize; ++i) {
        for (std::size_t j = 0; j < MaxFacetDofs; ++j) {
            rCondensedSystem.KinvC(i, j) = rhs(i, j);
        }
        rCondensedSystem.KinvR[i] = rhs(i, MaxFacetDofs);
    }

    // Schur complement S = B - C^T K^-1 C and right hand side - C^T K^-1 R
    noalias(rCondensedSystem.S) = rLocalSystem.B - prod(trans(rLocalSystem.C), rCondensedSystem.KinvC);
    noalias(rCondensedSystem.RHS) = -prod(trans(rLocalSystem.C), rCondensedSystem.KinvR);
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::RecoverLocalSolution(
    const LocalSystem& rLocalSystem,
    const CondensedSystem& rCondensedSystem,
    const array_1d<double, MaxFacetDofs>& rLocalFacetValues,
    ElementalValuesType& rIndicator,
    double& rLagrangeMultiplier
    )
{
    const array_1d<double, LocalSize> solution = rCondensedSystem.KinvR - prod(rCondensedSystem.KinvC, rLocalFacetValues);
    for (std::size_t a = 0; a < NumNodes; ++a) {
        rIndicator[a] = solution[a];
    }
    rLagrangeMultiplier = solution[NumNodes];
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
double PdeConstrainedProjectionUtility<TDim>::ComputeCellResidual(
    const LocalSystem& rLocalSystem,
    const ElementalValuesType& rIndicator,
    const array_1d<double, MaxFacetDofs>& rLocalFacetValues
    )
{
    double residual = -rLocalSystem.R[NumNodes];
    for (std::size_t a = 0; a < NumNodes; ++a) {
        residual += rLocalSystem.K(NumNodes, a) * rIndicator[a];
    }
    for (std::size_t k = 0; k < rLocalSystem.NumberOfFacetDofs; ++k) {
        residual += rLocalSystem.C(NumNodes, k) * rLocalFacetValues[k];
    }
    return residual;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::L2ProjectDistance(
    const FacetTableType& rFacetTable,
    const ParticleContainerType& rParticles,
    std::vector<ElementalValuesType>& rDistance
    )
{
    KRATOS_TRY

    const std::size_t n_elements = rFacetTable.NumberOfElements();
    KRATOS_ERROR_IF(rDistance.size() != n_elements) << "Wrong elemental distance size." << std::endl;

    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const std::size_t n_particles = rParticles.NumberOfParticles(iElement);
        if (n_particles == 0) {
            return;
        }

        double mean = 0.0;
        for (auto it_particle = rParticles.ParticlesBegin(iElement); it_particle != rParticles.ParticlesEnd(iElement); ++it_particle) {
            mean += it_particle->Distance;
        }
        mean /= static_cast<double>(n_particles);

        auto& r_distance = rDistance[iElement];
        if (n_particles < NumNodes) {
            std::fill(r_distance.begin(), r_distance.end(), mean);
            return;
        }

        BoundedMatrix<double, NumNodes, NumNodes> A = ZeroMatrix(NumNodes, NumNodes);
        BoundedMatrix<double, NumNodes, 1> b = ZeroMatrix(NumNodes, 1);
        typename FacetTableType::ShapeFunctionsType N;
        for (auto it_particle = rParticles.ParticlesBegin(iElement); it_particle != rParticles.ParticlesEnd(iElement); ++it_particle) {
            rFacetTable.ShapeFunctions(iElement, it_particle->Coordinates, N);
            for (std::size_t a = 0; a < NumNodes; ++a) {
                b(a, 0) += N[a] * it_particle->Distance;
                for (std::size_t c = 0; c < NumNodes; ++c) {
                    A(a, c) += N[a] * N[c];
                }
            }
        }

        // Tikhonov regularization towards the particle mean
        double trace = 0.0;
        for (std::size_t a = 0; a < NumNodes; ++a) {
            trace += A(a, a);
        }
        const double epsilon = 1.0e-8 * trace / static_cast<double>(NumNodes);
        for (std::size_t a = 0; a < NumNodes; ++a) {
            A(a, a) += epsilon;
            b(a, 0) += epsilon * mean;
        }

        SolveDenseSystem<NumNodes, 1>(A, b);
        for (std::size_t a = 0; a < NumNodes; ++a) {
            r_distance[a] = b(a, 0);
        }
    });

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::InitializeIndicatorFromDistance(
    const FacetTableType& rFacetTable,
    ModelPart& rModelPart,
    const std::vector<double>& rNodalDistance,
    std::vector<ElementalValuesType>& rIndicator
    )
{
    KRATOS_TRY

    const std::size_t n_elements = rFacetTable.NumberOfElements();
    rIndicator.resize(n_elements);
    const auto mod_sh_func_factory = FluidAuxiliaryUtilities::GetStandardModifiedShapeFunctionsFactory(rModelPart.ElementsBegin()->GetGeometry());

    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const auto& r_data = rFacetTable.GetElementData(iElement);
        auto& r_indicator = rIndicator[iElement];
        // Same split criterion as the geometry division utilities (zero distances are ignored)
        Vector distances(NumNodes);
        std::size_t n_positive = 0;
        std::size_t n_negative = 0;
        for (std::size_t a = 0; a < NumNodes; ++a) {
            distances[a] = rNodalDistance[r_data.NodeIndices[a]];
            n_positive += distances[a] > 0.0 ? 1 : 0;
            n_negative += distances[a] < 0.0 ? 1 : 0;
        }

        if (n_negative == 0) {
            std::fill(r_indicator.begin(), r_indicator.end(), 0.0);
        } else if (n_positive == 0) {
            std::fill(r_indicator.begin(), r_indicator.end(), 1.0);
        } else {
            // Right hand side int_{K^-} N_a with the negative side quadrature
            auto p_mod_sh_func = mod_sh_func_factory((rModelPart.ElementsBegin() + iElement)->pGetGeometry(), distances);
            Matrix N_negative;
            ModifiedShapeFunctions::ShapeFunctionsGradientsType DN_DX_negative;
            Vector weights_negative;
            p_mod_sh_func->ComputeNegativeSideShapeFunctionsAndGradientsValues(N_negative, DN_DX_negative, weights_negative, GeometryData::IntegrationMethod::GI_GAUSS_2);
            array_1d<double, NumNodes> rhs = ZeroVector(NumNodes);
            for (std::size_t g = 0; g < weights_negative.size(); ++g) {
                for (std::size_t a = 0; a < NumNodes; ++a) {
                    rhs[a] += weights_negative[g] * N_negative(g, a);
                }
            }

            // Inverse of the P1 mass matrix V / ((n + 1) (n + 2)) (I + 1 1^T) is ((n + 1) (n + 2) / V) (I - 1 1^T / (n + 2))
            const double rhs_sum = std::accumulate(rhs.begin(), rhs.end(), 0.0);
            const double factor = static_cast<double>(NumNodes * (NumNodes + 1)) / r_data.Volume;
            for (std::size_t a = 0; a < NumNodes; ++a) {
                r_indicator[a] = factor * (rhs[a] - rhs_sum / static_cast<double>(NumNodes + 1));
            }
        }
    });

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
double PdeConstrainedProjectionUtility<TDim>::ComputeNegativeVolume(
    const FacetTableType& rFacetTable,
    ModelPart& rModelPart,
    const std::vector<double>& rNodalDistance,
    const double Shift
    )
{
    KRATOS_TRY

    const auto mod_sh_func_factory = FluidAuxiliaryUtilities::GetStandardModifiedShapeFunctionsFactory(rModelPart.ElementsBegin()->GetGeometry());
    const double local_volume = IndexPartition<std::size_t>(rFacetTable.NumberOfElements()).for_each<SumReduction<double>>(Vector(NumNodes), [&](const std::size_t iElement, Vector& rDistances) {
        const auto& r_data = rFacetTable.GetElementData(iElement);
        // Same split criterion as the geometry division utilities (zero distances are ignored)
        std::size_t n_positive = 0;
        std::size_t n_negative = 0;
        for (std::size_t a = 0; a < NumNodes; ++a) {
            rDistances[a] = rNodalDistance[r_data.NodeIndices[a]] + Shift;
            n_positive += rDistances[a] > 0.0 ? 1 : 0;
            n_negative += rDistances[a] < 0.0 ? 1 : 0;
        }
        if (n_negative == 0) {
            return 0.0;
        } else if (n_positive == 0) {
            return r_data.Volume;
        } else {
            auto p_mod_sh_func = mod_sh_func_factory((rModelPart.ElementsBegin() + iElement)->pGetGeometry(), rDistances);
            return p_mod_sh_func->ComputeNegativeSideDomainSize();
        }
    });
    return rFacetTable.GetDataCommunicator().SumAll(local_volume);

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
double PdeConstrainedProjectionUtility<TDim>::ComputeIntegral(
    const FacetTableType& rFacetTable,
    const std::vector<ElementalValuesType>& rValues
    )
{
    const double local_integral = IndexPartition<std::size_t>(rFacetTable.NumberOfElements()).for_each<SumReduction<double>>([&](const std::size_t iElement) {
        const auto& r_values = rValues[iElement];
        return rFacetTable.GetElementData(iElement).Volume * std::accumulate(r_values.begin(), r_values.end(), 0.0) / static_cast<double>(NumNodes);
    });
    return rFacetTable.GetDataCommunicator().SumAll(local_integral);
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
template<std::size_t TSize, std::size_t TNumberOfRHS>
void PdeConstrainedProjectionUtility<TDim>::SolveDenseSystem(
    BoundedMatrix<double, TSize, TSize>& rA,
    BoundedMatrix<double, TSize, TNumberOfRHS>& rB
    )
{
    // Gaussian elimination with partial pivoting
    for (std::size_t k = 0; k < TSize; ++k) {
        std::size_t pivot = k;
        double max_value = std::abs(rA(k, k));
        for (std::size_t i = k + 1; i < TSize; ++i) {
            if (std::abs(rA(i, k)) > max_value) {
                max_value = std::abs(rA(i, k));
                pivot = i;
            }
        }
        KRATOS_ERROR_IF(max_value == 0.0) << "Singular local system in the particle-mesh projection." << std::endl;
        if (pivot != k) {
            for (std::size_t j = 0; j < TSize; ++j) {
                std::swap(rA(k, j), rA(pivot, j));
            }
            for (std::size_t j = 0; j < TNumberOfRHS; ++j) {
                std::swap(rB(k, j), rB(pivot, j));
            }
        }
        for (std::size_t i = k + 1; i < TSize; ++i) {
            const double factor = rA(i, k) / rA(k, k);
            if (factor != 0.0) {
                for (std::size_t j = k; j < TSize; ++j) {
                    rA(i, j) -= factor * rA(k, j);
                }
                for (std::size_t j = 0; j < TNumberOfRHS; ++j) {
                    rB(i, j) -= factor * rB(k, j);
                }
            }
        }
    }

    // Back substitution
    for (std::size_t k = TSize; k-- > 0;) {
        for (std::size_t j = 0; j < TNumberOfRHS; ++j) {
            double value = rB(k, j);
            for (std::size_t i = k + 1; i < TSize; ++i) {
                value -= rA(k, i) * rB(i, j);
            }
            rB(k, j) = value / rA(k, k);
        }
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void PdeConstrainedProjectionUtility<TDim>::GatherLocalFacetValues(
    const FacetLinearSystem& rFacetSystem,
    const LocalSystem& rLocalSystem,
    array_1d<double, MaxFacetDofs>& rLocalFacetValues
    )
{
    noalias(rLocalFacetValues) = ZeroVector(MaxFacetDofs);
    for (std::size_t k = 0; k < rLocalSystem.NumberOfFacetDofs; ++k) {
        rLocalFacetValues[k] = rFacetSystem.GetValue(rLocalSystem.FacetDofs[k]);
    }
}

/***********************************************************************************/
/***********************************************************************************/

template class PdeConstrainedProjectionUtility<2>;
template class PdeConstrainedProjectionUtility<3>;

} // namespace Kratos
