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
#include <memory>
#include <vector>

// External includes

// Project includes
#include "containers/model.h"
#include "includes/define.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "processes/process.h"

// Application includes
#include "custom_utilities/particle_mesh/lagrangian_particle_container.h"
#include "custom_utilities/particle_mesh/mesh_facet_table.h"
#include "custom_utilities/particle_mesh/pde_constrained_projection_utility.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class ParticleMeshPdeProjectionProcess
 * @ingroup FluidDynamicsApplication
 * @brief Particle-based level set transport with the LEoPart PDE-constrained particle-mesh projection
 * @details Alternative to the Eulerian level set convection of the two-fluid solvers. Lagrangian particles carry a binary
 * phase indicator c_p (1 where DISTANCE < 0) and a signed distance phi_p. Each call to Execute performs one time step:
 * 1. The particles are advected with the nodal VELOCITY (interpolated between the previous and the current step).
 * 2. The number of particles per element is kept between the given bounds.
 * 3. The indicator is projected onto a discontinuous P1 field c_h with the PDE-constrained projection (Maljaars et al.,
 *    CMAME 348, 2019), which makes the integral of c_h change exactly by the boundary fluxes, both per element and globally.
 *    By default the boundary fluxes use the level set indicator H(-DISTANCE) ("outflow_flux_from_level_set"), so the phase only
 *    leaves the domain where the level set says it does; set it to false to use the interior trace on outflow (LEoPart).
 * 4. The particle distance is projected elementwise (l2) and averaged to the nodal DISTANCE.
 * 5. The nodal DISTANCE is shifted so that the volume of its negative side equals the integral of c_h (global volume correction).
 * 6. The indicator is written to the elemental and nodal (non-historical) PHASE_FRACTION for visualization.
 * ExecuteFinalizeSolutionStep adds the DISTANCE change done by other processes (e.g. redistancing) to the particle distance,
 * without changing the particle indicator.
 * @note For fixed linear simplex meshes (no remeshing, no moving mesh). In MPI runs the particles migrate between the ranks, the facet
 * system is distributed (TrilinosParticleMeshPdeProjectionProcess in the FluidDynamics TrilinosExtension) and all the reductions are global.
 * @tparam TDim Working space dimension
 * @author Vicente Mataix Ferrandiz
 */
template<std::size_t TDim>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) ParticleMeshPdeProjectionProcess : public Process
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of ParticleMeshPdeProjectionProcess
    KRATOS_CLASS_POINTER_DEFINITION(ParticleMeshPdeProjectionProcess);

    /// Facet table type
    using FacetTableType = MeshFacetTable<TDim>;

    /// Particle container type
    using ParticleContainerType = LagrangianParticleContainer<TDim>;

    /// Projection utility type
    using ProjectionUtilityType = PdeConstrainedProjectionUtility<TDim>;

    /// Serial linear solver type
    using LinearSolverType = typename ProjectionUtilityType::LinearSolverType;

    /// Elemental P1 values
    using ElementalValuesType = typename ProjectionUtilityType::ElementalValuesType;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Construct a new process from a model part (serial and OpenMP runs)
     * @param rModelPart The model part of the fluid domain (linear simplices)
     * @param pLinearSolver Linear solver for the facet system of the projection
     * @param ThisParameters Settings
     */
    ParticleMeshPdeProjectionProcess(
        ModelPart& rModelPart,
        typename LinearSolverType::Pointer pLinearSolver,
        Parameters ThisParameters
        );

    /**
     * @brief Construct a new process from a model part and a facet linear system (e.g. a distributed one for MPI runs)
     * @param rModelPart The model part of the fluid domain (linear simplices)
     * @param pFacetSystem Global system of the facet unknowns of the projection
     * @param ThisParameters Settings
     */
    ParticleMeshPdeProjectionProcess(
        ModelPart& rModelPart,
        FacetLinearSystem::Pointer pFacetSystem,
        Parameters ThisParameters
        );

    /**
     * @brief Construct a new process from a model (the model part is taken from "model_part_name")
     * @param rModel The model
     * @param pLinearSolver Linear solver for the facet system of the projection
     * @param ThisParameters Settings
     */
    ParticleMeshPdeProjectionProcess(
        Model& rModel,
        typename LinearSolverType::Pointer pLinearSolver,
        Parameters ThisParameters
        );

    /// Destructor.
    ~ParticleMeshPdeProjectionProcess() override = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Builds the facet table and seeds the particles from the current DISTANCE
     * @details It is called automatically by the first Execute if it was not called before
     */
    void ExecuteInitialize() override;

    /// Performs the particle transport of one time step and updates DISTANCE
    void Execute() override;

    /// Adds the DISTANCE change done after Execute (e.g. redistancing) to the particle distance
    void ExecuteFinalizeSolutionStep() override;

    /// Checks the model part and the settings
    int Check() override;

    /// Returns the default parameters
    const Parameters GetDefaultParameters() const override;

    ///@}
    ///@name Access
    ///@{

    /// Returns the integral of the projected indicator (the conserved phase volume). Collective in MPI runs
    double GetIndicatorIntegral() const;

    /// Returns the current (global) number of particles. Collective in MPI runs
    std::size_t GetNumberOfParticles() const;

    /// Returns the shift applied to the distance in the last volume correction
    double GetLastVolumeCorrectionShift() const
    {
        return mLastShift;
    }

    /// Returns the information of the last projection
    const typename ProjectionUtilityType::ProjectionInfo& GetLastProjectionInfo() const
    {
        return mLastProjectionInfo;
    }

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        return "ParticleMeshPdeProjectionProcess" + std::to_string(TDim) + "D";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << Info();
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    ModelPart& mrModelPart;

    FacetLinearSystem::Pointer mpFacetSystem;

    Parameters mParameters;

    bool mIsInitialized = false;

    std::unique_ptr<FacetTableType> mpFacetTable;

    std::unique_ptr<ParticleContainerType> mpParticles;

    std::unique_ptr<ProjectionUtilityType> mpProjection;

    std::vector<ElementalValuesType> mIndicator;

    std::vector<ElementalValuesType> mDistance;

    std::vector<double> mDistanceAfterExecute;

    double mLastShift = 0.0;

    typename ProjectionUtilityType::ProjectionInfo mLastProjectionInfo;

    ///@}
    ///@name Private Operations
    ///@{

    /// Copies the current nodal DISTANCE to a vector indexed by node position
    void GetNodalDistance(std::vector<double>& rDistance) const;

    /// Copies the nodal VELOCITY of a buffer position to a vector indexed by node position
    void GetNodalVelocity(
        const std::size_t BufferIndex,
        typename ParticleContainerType::NodalVectorValuesType& rVelocity
        ) const;

    /// Averages an elemental P1 field to the nodes (volume weighted)
    void AverageToNodes(
        const std::vector<ElementalValuesType>& rElementalValues,
        std::vector<double>& rNodalValues
        ) const;

    /// Finds the shift of the nodal distance whose negative volume equals the target volume
    double ComputeVolumeCorrectionShift(
        const std::vector<double>& rNodalDistance,
        const double TargetVolume
        ) const;

    /// Writes the indicator to PHASE_FRACTION
    void WritePhaseFraction() const;

    /// Returns true in the rank that writes the log (rank 0)
    bool IsLoggingRank() const;

    ///@}
}; // Class ParticleMeshPdeProjectionProcess

///@}
///@} addtogroup block

} // namespace Kratos
