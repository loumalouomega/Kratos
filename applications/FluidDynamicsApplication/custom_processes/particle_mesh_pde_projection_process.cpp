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
#include <numeric>

// External includes

// Project includes
#include "includes/variables.h"
#include "includes/dem_variables.h"
#include "utilities/atomic_utilities.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"

// Application includes
#include "fluid_dynamics_application_variables.h"
#include "particle_mesh_pde_projection_process.h"

namespace Kratos
{

template<std::size_t TDim>
ParticleMeshPdeProjectionProcess<TDim>::ParticleMeshPdeProjectionProcess(
    ModelPart& rModelPart,
    typename LinearSolverType::Pointer pLinearSolver,
    Parameters ThisParameters
    ) : ParticleMeshPdeProjectionProcess(
            rModelPart,
            Kratos::make_shared<SerialFacetLinearSystem>(pLinearSolver),
            ThisParameters)
{
    KRATOS_ERROR_IF(rModelPart.GetCommunicator().IsDistributed()) << "The serial linear solver constructor cannot be used in MPI runs. "
        << "Use TrilinosParticleMeshPdeProjectionProcess from the FluidDynamics TrilinosExtension instead." << std::endl;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
ParticleMeshPdeProjectionProcess<TDim>::ParticleMeshPdeProjectionProcess(
    ModelPart& rModelPart,
    FacetLinearSystem::Pointer pFacetSystem,
    Parameters ThisParameters
    ) : Process(),
        mrModelPart(rModelPart),
        mpFacetSystem(pFacetSystem),
        mParameters(ThisParameters)
{
    KRATOS_TRY

    mParameters.RecursivelyValidateAndAssignDefaults(GetDefaultParameters());
    KRATOS_ERROR_IF(mpFacetSystem == nullptr) << "A facet linear system is required for the PDE-constrained projection." << std::endl;

    const auto& r_counts = mParameters["particles_per_element"];
    KRATOS_ERROR_IF(r_counts["minimum"].GetInt() < static_cast<int>(TDim + 1))
        << "At least " << TDim + 1 << " particles per element are required. Got minimum " << r_counts["minimum"].GetInt() << "." << std::endl;
    KRATOS_ERROR_IF(r_counts["minimum"].GetInt() > r_counts["initial"].GetInt() || r_counts["initial"].GetInt() > r_counts["maximum"].GetInt())
        << "'particles_per_element' must satisfy minimum <= initial <= maximum." << std::endl;
    const int rk_order = mParameters["runge_kutta_order"].GetInt();
    KRATOS_ERROR_IF(rk_order < 1 || rk_order > 3) << "'runge_kutta_order' must be 1, 2 or 3. Got " << rk_order << "." << std::endl;
    KRATOS_ERROR_IF(mParameters["max_CFL"].GetDouble() <= 0.0) << "'max_CFL' must be positive." << std::endl;
    KRATOS_ERROR_IF(mParameters["beta"].GetDouble() <= 0.0) << "'beta' must be positive." << std::endl;
    KRATOS_ERROR_IF(mParameters["zeta"].GetDouble() < 0.0) << "'zeta' must be non-negative." << std::endl;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
ParticleMeshPdeProjectionProcess<TDim>::ParticleMeshPdeProjectionProcess(
    Model& rModel,
    typename LinearSolverType::Pointer pLinearSolver,
    Parameters ThisParameters
    ) : ParticleMeshPdeProjectionProcess(
            rModel.GetModelPart(ThisParameters["model_part_name"].GetString()),
            pLinearSolver,
            ThisParameters)
{
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
const Parameters ParticleMeshPdeProjectionProcess<TDim>::GetDefaultParameters() const
{
    // The gradient penalty and the particle counts follow the LEoPart recommendations (zeta close to the number of particles per element)
    // The facet penalty is larger than the LEoPart value (1e-6): with a very small beta most facet unknowns are only controlled by the
    // penalty, which makes the facet system too ill-conditioned for iterative solvers (LEoPart uses a direct solver)
    Parameters default_parameters(R"({
        "model_part_name"         : "",
        "echo_level"              : 0,
        "particles_per_element"   : {
            "initial" : 25,
            "minimum" : 15,
            "maximum" : 50
        },
        "runge_kutta_order"       : 3,
        "max_CFL"                 : 1.0,
        "beta"                    : 1.0e-2,
        "zeta"                    : 25.0,
        "volume_correction"       : {
            "enabled"            : true,
            "relative_tolerance" : 1.0e-10,
            "max_iterations"     : 50
        },
        "outflow_flux_from_level_set" : true,
        "update_particle_distance_after_redistancing" : true,
        "random_seed"             : 0
    })");
    if constexpr (TDim == 3) {
        default_parameters["particles_per_element"]["initial"].SetInt(40);
        default_parameters["particles_per_element"]["minimum"].SetInt(25);
        default_parameters["particles_per_element"]["maximum"].SetInt(80);
        default_parameters["zeta"].SetDouble(40.0);
    }
    return default_parameters;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
int ParticleMeshPdeProjectionProcess<TDim>::Check()
{
    KRATOS_TRY

    KRATOS_ERROR_IF(mrModelPart.NumberOfElements() == 0) << "Model part '" << mrModelPart.FullName() << "' has no elements." << std::endl;
    KRATOS_ERROR_IF_NOT(mrModelPart.HasNodalSolutionStepVariable(DISTANCE)) << "DISTANCE is not in the nodal database of '" << mrModelPart.FullName() << "'." << std::endl;
    KRATOS_ERROR_IF_NOT(mrModelPart.HasNodalSolutionStepVariable(VELOCITY)) << "VELOCITY is not in the nodal database of '" << mrModelPart.FullName() << "'." << std::endl;
    KRATOS_ERROR_IF(mrModelPart.GetBufferSize() < 2) << "A buffer size of at least 2 is required." << std::endl;
    for (const auto& r_element : mrModelPart.Elements()) {
        const auto& r_geometry = r_element.GetGeometry();
        KRATOS_ERROR_IF(r_geometry.PointsNumber() != TDim + 1 || r_geometry.LocalSpaceDimension() != TDim)
            << "Element " << r_element.Id() << " is not a linear simplex of dimension " << TDim << "." << std::endl;
    }

    return 0;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::ExecuteInitialize()
{
    KRATOS_TRY

    if (mIsInitialized) {
        return;
    }

    Check();

    mpFacetTable = std::make_unique<FacetTableType>(mrModelPart);
    mpParticles = std::make_unique<ParticleContainerType>(*mpFacetTable, mrModelPart, static_cast<std::uint64_t>(mParameters["random_seed"].GetInt()));
    mpProjection = std::make_unique<ProjectionUtilityType>(*mpFacetTable);

    // Initial fields from the current nodal distance
    std::vector<double> nodal_distance;
    GetNodalDistance(nodal_distance);
    ProjectionUtilityType::InitializeIndicatorFromDistance(*mpFacetTable, mrModelPart, nodal_distance, mIndicator);
    const std::size_t n_elements = mpFacetTable->NumberOfElements();
    mDistance.resize(n_elements);
    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const auto& r_data = mpFacetTable->GetElementData(iElement);
        for (std::size_t a = 0; a < TDim + 1; ++a) {
            mDistance[iElement][a] = nodal_distance[r_data.NodeIndices[a]];
        }
    });
    mpParticles->Seed(static_cast<std::size_t>(mParameters["particles_per_element"]["initial"].GetInt()), nodal_distance);
    mDistanceAfterExecute = nodal_distance;
    mLastProjectionInfo.Integral = ProjectionUtilityType::ComputeIntegral(*mpFacetTable, mIndicator);
    mLastProjectionInfo.PreviousIntegral = mLastProjectionInfo.Integral;
    WritePhaseFraction();

    mIsInitialized = true;

    const std::size_t n_particles = GetNumberOfParticles();
    KRATOS_INFO_IF("ParticleMeshPdeProjectionProcess", mParameters["echo_level"].GetInt() > 0 && IsLoggingRank())
        << "Initialized with " << n_particles << " particles, " << mpFacetTable->NumberOfFacetDofs() / TDim
        << " shared facets and phase volume " << mLastProjectionInfo.Integral << "." << std::endl;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::Execute()
{
    KRATOS_TRY

    if (!mIsInitialized) {
        ExecuteInitialize();
    }

    KRATOS_ERROR_IF(mrModelPart.NumberOfElements() != mpFacetTable->NumberOfElements() || mrModelPart.NumberOfNodes() != mpFacetTable->NumberOfNodes())
        << "The mesh of '" << mrModelPart.FullName() << "' changed. Remeshing is not supported by ParticleMeshPdeProjectionProcess." << std::endl;

    const double dt = mrModelPart.GetProcessInfo()[DELTA_TIME];
    const int echo_level = mParameters["echo_level"].GetInt();
    const std::size_t n_nodes = mpFacetTable->NumberOfNodes();

    // Nodal data
    typename ParticleContainerType::NodalVectorValuesType old_velocity, new_velocity;
    GetNodalVelocity(1, old_velocity);
    GetNodalVelocity(0, new_velocity);
    typename ParticleContainerType::NodalVectorValuesType flux_velocity(n_nodes);
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        noalias(flux_velocity[iNode]) = 0.5 * (old_velocity[iNode] + new_velocity[iNode]);
    });
    std::vector<double> nodal_distance;
    GetNodalDistance(nodal_distance);
    // Impermeable nodes: slip walls and no-slip walls (all velocity components fixed to zero)
    std::vector<bool> is_closed(n_nodes);
    for (std::size_t i_node = 0; i_node < n_nodes; ++i_node) {
        const auto it_node = mrModelPart.NodesBegin() + i_node;
        const bool is_fixed = it_node->IsFixed(VELOCITY_X) && it_node->IsFixed(VELOCITY_Y) && (TDim == 2 || it_node->IsFixed(VELOCITY_Z));
        is_closed[i_node] = it_node->Is(SLIP) || (is_fixed && norm_2(new_velocity[i_node]) == 0.0);
    }

    // Particle advection and population control
    std::vector<typename ParticleContainerType::FaceTypesType> face_types;
    ParticleContainerType::ClassifyFaces(*mpFacetTable, flux_velocity, is_closed, face_types);
    const std::size_t n_lost = mpParticles->Advect(old_velocity, new_velocity, face_types, dt, mParameters["max_CFL"].GetDouble(), mParameters["runge_kutta_order"].GetInt());
    const auto& r_counts = mParameters["particles_per_element"];
    const auto population = mpParticles->ControlPopulation(r_counts["minimum"].GetInt(), r_counts["maximum"].GetInt(), face_types, nodal_distance, mIndicator, mDistance);

    // Conservative projection of the indicator
    typename ProjectionUtilityType::ProjectionSettings settings;
    settings.Beta = mParameters["beta"].GetDouble();
    settings.Zeta = mParameters["zeta"].GetDouble();
    settings.DeltaTime = dt;
    settings.OutflowFromLevelSet = mParameters["outflow_flux_from_level_set"].GetBool();
    mLastProjectionInfo = mpProjection->Project(*mpParticles, flux_velocity, face_types, nodal_distance, settings, *mpFacetSystem, mIndicator);

    // Projection of the distance and volume correction
    ProjectionUtilityType::L2ProjectDistance(*mpFacetTable, *mpParticles, mDistance);
    AverageToNodes(mDistance, nodal_distance);
    mLastShift = 0.0;
    if (mParameters["volume_correction"]["enabled"].GetBool()) {
        mLastShift = ComputeVolumeCorrectionShift(nodal_distance, mLastProjectionInfo.Integral);
        if (mLastShift != 0.0) {
            for (auto& r_value : nodal_distance) {
                r_value += mLastShift;
            }
            for (auto& r_values : mDistance) {
                for (auto& r_value : r_values) {
                    r_value += mLastShift;
                }
            }
            mpParticles->IncrementDistance(std::vector<double>(n_nodes, mLastShift));
        }
    }

    // Write the nodal distance and the phase fraction
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        (mrModelPart.NodesBegin() + iNode)->FastGetSolutionStepValue(DISTANCE) = nodal_distance[iNode];
    });
    mDistanceAfterExecute = nodal_distance;
    WritePhaseFraction();

    const auto& r_info = mLastProjectionInfo;
    KRATOS_WARNING_IF("ParticleMeshPdeProjectionProcess", r_info.FacetSystemResidual > 1.0e-6 && IsLoggingRank())
        << "The facet system of the projection did not converge (relative residual " << r_info.FacetSystemResidual
        << "). The phase volume is still conserved, but the indicator may be inaccurate. Use a larger 'beta' or a direct linear solver." << std::endl;
    const double balance = r_info.Integral - r_info.PreviousIntegral + r_info.NetBoundaryOutflow;
    KRATOS_INFO_IF("ParticleMeshPdeProjectionProcess", echo_level > 0 && IsLoggingRank())
        << "Phase volume " << r_info.Integral << " (balance error " << balance << ", max cell residual " << r_info.MaxCellResidual
        << "), indicator range [" << r_info.MinValue << ", " << r_info.MaxValue << "], distance shift " << mLastShift
        << ", facet system relative residual " << r_info.FacetSystemResidual << "." << std::endl;
    if (echo_level > 1) {
        const auto& r_data_communicator = mrModelPart.GetCommunicator().GetDataCommunicator();
        const std::vector<unsigned long> counts = r_data_communicator.SumAll(std::vector<unsigned long>{
            mpParticles->NumberOfParticles(), n_lost, population.Added, population.Removed, population.EmptyElements});
        KRATOS_INFO_IF("ParticleMeshPdeProjectionProcess", IsLoggingRank())
            << counts[0] << " particles (" << counts[1] << " left the domain, " << counts[2] << " added, "
            << counts[3] << " removed, " << counts[4] << " empty elements before reseeding)." << std::endl;
    }

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::ExecuteFinalizeSolutionStep()
{
    KRATOS_TRY

    if (!mIsInitialized || !mParameters["update_particle_distance_after_redistancing"].GetBool()) {
        return;
    }

    std::vector<double> increment;
    GetNodalDistance(increment);
    IndexPartition<std::size_t>(increment.size()).for_each([&](const std::size_t iNode) {
        increment[iNode] -= mDistanceAfterExecute[iNode];
    });
    mpParticles->IncrementDistance(increment);
    IndexPartition<std::size_t>(mDistance.size()).for_each([&](const std::size_t iElement) {
        const auto& r_data = mpFacetTable->GetElementData(iElement);
        for (std::size_t a = 0; a < TDim + 1; ++a) {
            mDistance[iElement][a] += increment[r_data.NodeIndices[a]];
        }
    });
    GetNodalDistance(mDistanceAfterExecute);

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
double ParticleMeshPdeProjectionProcess<TDim>::GetIndicatorIntegral() const
{
    return mIsInitialized ? ProjectionUtilityType::ComputeIntegral(*mpFacetTable, mIndicator) : 0.0;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
std::size_t ParticleMeshPdeProjectionProcess<TDim>::GetNumberOfParticles() const
{
    const std::size_t n_local = mIsInitialized ? mpParticles->NumberOfParticles() : 0;
    return mrModelPart.GetCommunicator().GetDataCommunicator().SumAll(static_cast<unsigned long>(n_local));
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::GetNodalDistance(std::vector<double>& rDistance) const
{
    const std::size_t n_nodes = mrModelPart.NumberOfNodes();
    rDistance.resize(n_nodes);
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        rDistance[iNode] = (mrModelPart.NodesBegin() + iNode)->FastGetSolutionStepValue(DISTANCE);
    });
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::GetNodalVelocity(
    const std::size_t BufferIndex,
    typename ParticleContainerType::NodalVectorValuesType& rVelocity
    ) const
{
    const std::size_t n_nodes = mrModelPart.NumberOfNodes();
    rVelocity.resize(n_nodes);
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        noalias(rVelocity[iNode]) = (mrModelPart.NodesBegin() + iNode)->FastGetSolutionStepValue(VELOCITY, BufferIndex);
    });
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::AverageToNodes(
    const std::vector<ElementalValuesType>& rElementalValues,
    std::vector<double>& rNodalValues
    ) const
{
    // Volume weighted sums, assembled among the ranks sharing a node
    const std::size_t n_nodes = mpFacetTable->NumberOfNodes();
    const auto it_node_begin = mrModelPart.NodesBegin();
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        (it_node_begin + iNode)->SetValue(PARTICLE_MESH_NODAL_SUMS, ZeroVector(3));
    });
    IndexPartition<std::size_t>(mpFacetTable->NumberOfElements()).for_each([&](const std::size_t iElement) {
        const auto& r_data = mpFacetTable->GetElementData(iElement);
        for (std::size_t a = 0; a < TDim + 1; ++a) {
            auto& r_sums = (it_node_begin + r_data.NodeIndices[a])->GetValue(PARTICLE_MESH_NODAL_SUMS);
            AtomicAdd(r_sums[0], r_data.Volume * rElementalValues[iElement][a]);
            AtomicAdd(r_sums[1], r_data.Volume);
        }
    });
    mrModelPart.GetCommunicator().AssembleNonHistoricalData(PARTICLE_MESH_NODAL_SUMS);

    rNodalValues.resize(n_nodes);
    IndexPartition<std::size_t>(n_nodes).for_each([&](const std::size_t iNode) {
        const auto& r_sums = (it_node_begin + iNode)->GetValue(PARTICLE_MESH_NODAL_SUMS);
        rNodalValues[iNode] = r_sums[1] > 0.0 ? r_sums[0] / r_sums[1] : 0.0;
    });
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
double ParticleMeshPdeProjectionProcess<TDim>::ComputeVolumeCorrectionShift(
    const std::vector<double>& rNodalDistance,
    const double TargetVolume
    ) const
{
    KRATOS_TRY

    // Global domain volume (all the ranks must take the same decisions in the iterations below)
    const double local_volume = IndexPartition<std::size_t>(mpFacetTable->NumberOfElements()).for_each<SumReduction<double>>([&](const std::size_t iElement) {
        return mpFacetTable->GetElementData(iElement).Volume;
    });
    const double total_volume = mrModelPart.GetCommunicator().GetDataCommunicator().SumAll(local_volume);
    const double tolerance = mParameters["volume_correction"]["relative_tolerance"].GetDouble() * total_volume;
    const int max_iterations = mParameters["volume_correction"]["max_iterations"].GetInt();

    // Nothing to correct if the target is (numerically) empty or full
    if (TargetVolume <= tolerance || TargetVolume >= total_volume - tolerance) {
        KRATOS_WARNING_IF("ParticleMeshPdeProjectionProcess", mParameters["echo_level"].GetInt() > 0 && IsLoggingRank())
            << "Volume correction skipped: target phase volume " << TargetVolume << " is empty or fills the domain." << std::endl;
        return 0.0;
    }

    // The negative volume is non-increasing with the shift: f(s) = V^-(d + s) - V_target
    const auto residual = [&](const double Shift) {
        return ProjectionUtilityType::ComputeNegativeVolume(*mpFacetTable, mrModelPart, rNodalDistance, Shift) - TargetVolume;
    };

    double s_a = 0.0;
    double f_a = residual(s_a);
    if (std::abs(f_a) <= tolerance) {
        return 0.0;
    }

    // Bracket the root expanding the step from the minimum element size
    double step = (f_a > 0.0 ? 1.0 : -1.0) * 0.1 * mpFacetTable->GetMinimumElementSize();
    double s_b = s_a + step;
    double f_b = residual(s_b);
    int iteration = 0;
    while (f_a * f_b > 0.0 && iteration++ < max_iterations) {
        s_a = s_b;
        f_a = f_b;
        step *= 2.0;
        s_b = s_a + step;
        f_b = residual(s_b);
    }
    if (f_a * f_b > 0.0) {
        KRATOS_WARNING_IF("ParticleMeshPdeProjectionProcess", IsLoggingRank()) << "Volume correction could not bracket the shift. No correction applied." << std::endl;
        return 0.0;
    }
    if (std::abs(f_b) <= tolerance) {
        return s_b;
    }

    // Illinois (modified regula falsi) iterations
    int side = 0;
    double s_c = s_b;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        s_c = (s_a * f_b - s_b * f_a) / (f_b - f_a);
        const double f_c = residual(s_c);
        if (std::abs(f_c) <= tolerance) {
            return s_c;
        }
        if (f_c * f_b > 0.0) {
            s_b = s_c;
            f_b = f_c;
            if (side == -1) {
                f_a *= 0.5;
            }
            side = -1;
        } else {
            s_a = s_c;
            f_a = f_c;
            if (side == 1) {
                f_b *= 0.5;
            }
            side = 1;
        }
    }

    KRATOS_WARNING_IF("ParticleMeshPdeProjectionProcess", IsLoggingRank()) << "Volume correction did not converge in " << max_iterations << " iterations." << std::endl;
    return s_c;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void ParticleMeshPdeProjectionProcess<TDim>::WritePhaseFraction() const
{
    std::vector<double> nodal_indicator;
    AverageToNodes(mIndicator, nodal_indicator);
    IndexPartition<std::size_t>(nodal_indicator.size()).for_each([&](const std::size_t iNode) {
        (mrModelPart.NodesBegin() + iNode)->SetValue(PHASE_FRACTION, nodal_indicator[iNode]);
    });
    IndexPartition<std::size_t>(mIndicator.size()).for_each([&](const std::size_t iElement) {
        const auto& r_values = mIndicator[iElement];
        (mrModelPart.ElementsBegin() + iElement)->SetValue(PHASE_FRACTION, std::accumulate(r_values.begin(), r_values.end(), 0.0) / static_cast<double>(TDim + 1));
    });
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
bool ParticleMeshPdeProjectionProcess<TDim>::IsLoggingRank() const
{
    return mrModelPart.GetCommunicator().GetDataCommunicator().Rank() == 0;
}

/***********************************************************************************/
/***********************************************************************************/

template class ParticleMeshPdeProjectionProcess<2>;
template class ParticleMeshPdeProjectionProcess<3>;

} // namespace Kratos
