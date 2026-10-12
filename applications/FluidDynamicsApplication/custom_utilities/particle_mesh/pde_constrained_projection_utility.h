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
#include "includes/define.h"
#include "includes/model_part.h"
#include "linear_solvers/linear_solver.h"
#include "spaces/default_spaces.h"

// Application includes
#include "facet_linear_system.h"
#include "mesh_facet_table.h"
#include "lagrangian_particle_container.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class PdeConstrainedProjectionUtility
 * @ingroup FluidDynamicsApplication
 * @brief PDE-constrained particle-mesh projection (Maljaars et al., CMAME 348, 2019; LEoPart, CAMWA 81, 2021)
 * @details Projects the particle indicator onto a discontinuous P1 field c_h by minimizing
 * 1/2 sum_p (c_h(x_p) - c_p)^2 + 1/2 beta sum_K int_{dK} (cbar - c_h)^2 + 1/2 zeta sum_K int_K |grad c_h|^2
 * subject to the weak (HDG-type) advection constraint, tested with an elementwise constant multiplier lambda (l = 0)
 * int_K (c_h - c*) + dt sum_F int_F (u.n) ctilde = 0 for every element K,
 * where the flux value ctilde is the single-valued facet unknown cbar on interior facets and, on the boundary, the level set
 * indicator g = H(-d) on inflowing points and either g (default) or the interior trace c_h (LEoPart) on outflowing points.
 * Using g on the outflow lets the phase leave the domain only where the level set says it does, so the small indicator values
 * spread over the other phase by the projection cannot leak through open boundaries. The cell content therefore changes
 * exactly by its facet fluxes, which are single valued because the nodal velocity is continuous, so the projection is
 * conservative both locally and globally. This holds for any value of cbar, so it does not depend on the accuracy of the
 * global solve. Per element, the (c_h, lambda) unknowns are condensed and a global symmetric system is solved for the
 * shared facet unknowns only (through a FacetLinearSystem, serial or distributed). Then c_h and lambda are recovered element by element.
 * The penalty parameters are scaled with the element size h so that they do not depend on the mesh units:
 * beta_K = beta h^(1 - d) and zeta_K = zeta h^(2 - d).
 * The facet unknowns that carry no flux are only controlled by beta. A very small beta (1e-6 in LEoPart) therefore makes the
 * facet system very ill-conditioned (only direct solvers work) and spreads the particle/mesh mismatch over the domain, while a
 * large beta tends to a central flux with larger over/undershoots. The conservation does not depend on beta.
 * @tparam TDim Working space dimension
 * @author Vicente Mataix Ferrandiz
 */
template<std::size_t TDim>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) PdeConstrainedProjectionUtility
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of PdeConstrainedProjectionUtility
    KRATOS_CLASS_POINTER_DEFINITION(PdeConstrainedProjectionUtility);

    /// Facet table type
    using FacetTableType = MeshFacetTable<TDim>;

    /// Particle container type
    using ParticleContainerType = LagrangianParticleContainer<TDim>;

    /// Number of nodes of the simplex
    static constexpr std::size_t NumNodes = TDim + 1;

    /// Number of faces of the simplex
    static constexpr std::size_t NumFaces = TDim + 1;

    /// Size of the condensed local unknowns (c_h and lambda)
    static constexpr std::size_t LocalSize = NumNodes + 1;

    /// Maximum number of facet unknowns of an element
    static constexpr std::size_t MaxFacetDofs = NumFaces * TDim;

    /// Elemental P1 values (nodal basis)
    using ElementalValuesType = typename ParticleContainerType::ElementalValuesType;

    /// Nodal vector values
    using NodalVectorValuesType = typename ParticleContainerType::NodalVectorValuesType;

    /// Face types
    using FaceTypesType = typename ParticleContainerType::FaceTypesType;

    /// Serial linear solver type
    using LinearSolverType = SerialFacetLinearSystem::LinearSolverType;

    /// Settings of the projection
    struct ProjectionSettings
    {
        double Beta = 1.0e-6;   /// Facet coupling penalty
        double Zeta = 0.0;      /// Gradient penalty
        double DeltaTime = 1.0; /// Time step
        bool OutflowFromLevelSet = true; /// Outflow flux with the level set indicator (true) or the interior trace (false, LEoPart)
    };

    /// Local saddle point system of an element
    struct LocalSystem
    {
        BoundedMatrix<double, LocalSize, LocalSize> K;          /// [[A, g], [g^T, 0]]
        BoundedMatrix<double, LocalSize, MaxFacetDofs> C;       /// [L; h]
        BoundedMatrix<double, MaxFacetDofs, MaxFacetDofs> B;    /// Facet penalty block
        array_1d<double, LocalSize> R;                          /// [sum_p N(x_p) c_p; int_K c* - inflow]
        array_1d<double, NumNodes> OutflowCoefficients;         /// dt int_{dK} (u.n)^+ N if the outflow uses the trace (net outflow is its product with c_h)
        double PrescribedBoundaryFlux = 0.0;                    /// dt int_{dK} (u.n) g over the boundary points whose flux uses the level set indicator
        std::array<std::size_t, MaxFacetDofs> FacetDofs;        /// Global facet DOFs of the local facet unknowns
        std::size_t NumberOfFacetDofs = 0;                      /// Number of local facet unknowns
    };

    /// Condensed local system and back-substitution data
    struct CondensedSystem
    {
        BoundedMatrix<double, MaxFacetDofs, MaxFacetDofs> S;    /// Schur complement B - C^T K^-1 C
        array_1d<double, MaxFacetDofs> RHS;                     /// - C^T K^-1 R
        BoundedMatrix<double, LocalSize, MaxFacetDofs> KinvC;   /// K^-1 C
        array_1d<double, LocalSize> KinvR;                      /// K^-1 R
    };

    /// Information of a projection
    struct ProjectionInfo
    {
        double PreviousIntegral = 0.0;      /// int c*
        double Integral = 0.0;              /// int c_h
        double NetBoundaryOutflow = 0.0;    /// dt int_{dOmega} (u.n) ctilde
        double MaxCellResidual = 0.0;       /// max_K |int_K (c_h - c*) + dt int_{dK} (u.n) ctilde| / |K|
        double MinValue = 0.0;              /// Minimum nodal value of c_h
        double MaxValue = 0.0;              /// Maximum nodal value of c_h
        double FacetSystemResidual = 0.0;   /// Relative residual of the facet system solve (does not affect the conservation)
    };

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Construct a new projection utility
     * @param rFacetTable The facet table
     */
    explicit PdeConstrainedProjectionUtility(const FacetTableType& rFacetTable);

    /// Destructor.
    ~PdeConstrainedProjectionUtility() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Performs the PDE-constrained projection
     * @param rParticles The particles (their indicator is projected)
     * @param rFluxVelocity Nodal velocity used in the facet fluxes
     * @param rFaceTypes Face types of each element (wall faces carry no flux)
     * @param rNodalDistance Nodal distance (its negative side defines the boundary indicator g)
     * @param rSettings Projection settings
     * @param rFacetSystem Global system of the facet unknowns (initialized on the first call)
     * @param rIndicator Elemental indicator. On input c*, on output c_h
     * @return ProjectionInfo Information of the projection
     */
    ProjectionInfo Project(
        const ParticleContainerType& rParticles,
        const NodalVectorValuesType& rFluxVelocity,
        const std::vector<FaceTypesType>& rFaceTypes,
        const std::vector<double>& rNodalDistance,
        const ProjectionSettings& rSettings,
        FacetLinearSystem& rFacetSystem,
        std::vector<ElementalValuesType>& rIndicator
        );

    /**
     * @brief Builds the local saddle point system of an element
     * @param rFacetTable The facet table
     * @param rParticles The particles
     * @param ElementIndex Element position
     * @param rPreviousIndicator Elemental c*
     * @param rFluxVelocity Nodal velocity used in the facet fluxes
     * @param rFaceTypes Face types of the element (wall faces carry no flux)
     * @param rNodalDistance Nodal distance (boundary indicator)
     * @param rSettings Projection settings
     * @param rLocalSystem The local system
     */
    static void BuildLocalSystem(
        const FacetTableType& rFacetTable,
        const ParticleContainerType& rParticles,
        const std::size_t ElementIndex,
        const ElementalValuesType& rPreviousIndicator,
        const NodalVectorValuesType& rFluxVelocity,
        const FaceTypesType& rFaceTypes,
        const std::vector<double>& rNodalDistance,
        const ProjectionSettings& rSettings,
        LocalSystem& rLocalSystem
        );

    /**
     * @brief Condenses the element unknowns of a local system
     * @param rLocalSystem The local system
     * @param rCondensedSystem The condensed system
     */
    static void CondenseLocalSystem(
        const LocalSystem& rLocalSystem,
        CondensedSystem& rCondensedSystem
        );

    /**
     * @brief Recovers the element unknowns from the local facet values
     * @param rLocalSystem The local system
     * @param rCondensedSystem The condensed system
     * @param rLocalFacetValues Local facet values
     * @param rIndicator Recovered c_h
     * @param rLagrangeMultiplier Recovered lambda
     */
    static void RecoverLocalSolution(
        const LocalSystem& rLocalSystem,
        const CondensedSystem& rCondensedSystem,
        const array_1d<double, MaxFacetDofs>& rLocalFacetValues,
        ElementalValuesType& rIndicator,
        double& rLagrangeMultiplier
        );

    /**
     * @brief Computes the residual of the local conservation statement of an element
     * @return double int_K (c_h - c*) + dt int_{dK} (u.n) ctilde
     */
    static double ComputeCellResidual(
        const LocalSystem& rLocalSystem,
        const ElementalValuesType& rIndicator,
        const array_1d<double, MaxFacetDofs>& rLocalFacetValues
        );

    /**
     * @brief Elementwise least-squares (l2) projection of the particle distance
     * @details Elements with fewer particles than nodes get the mean particle value, and elements without particles keep their previous value.
     * A small Tikhonov term towards the particle mean avoids singular systems for badly distributed particles.
     * @param rFacetTable The facet table
     * @param rParticles The particles
     * @param rDistance Elemental distance. On input the previous values, on output the projected ones
     */
    static void L2ProjectDistance(
        const FacetTableType& rFacetTable,
        const ParticleContainerType& rParticles,
        std::vector<ElementalValuesType>& rDistance
        );

    /**
     * @brief Computes the elementwise L2 projection of the Heaviside function H(-d) of a nodal distance
     * @details Cut elements are integrated exactly on the negative side with the standard modified shape functions,
     * so the integral of the projected field equals the volume of the negative distance region
     * @param rFacetTable The facet table
     * @param rModelPart The model part
     * @param rNodalDistance Nodal distance
     * @param rIndicator Elemental indicator
     */
    static void InitializeIndicatorFromDistance(
        const FacetTableType& rFacetTable,
        ModelPart& rModelPart,
        const std::vector<double>& rNodalDistance,
        std::vector<ElementalValuesType>& rIndicator
        );

    /**
     * @brief Computes the volume of the region where a shifted nodal distance is negative
     * @param rFacetTable The facet table
     * @param rModelPart The model part
     * @param rNodalDistance Nodal distance
     * @param Shift Shift added to the nodal distance
     * @return double The negative volume
     */
    static double ComputeNegativeVolume(
        const FacetTableType& rFacetTable,
        ModelPart& rModelPart,
        const std::vector<double>& rNodalDistance,
        const double Shift
        );

    /**
     * @brief Computes the integral of an elemental P1 field
     * @param rFacetTable The facet table
     * @param rValues Elemental values
     * @return double The integral
     */
    static double ComputeIntegral(
        const FacetTableType& rFacetTable,
        const std::vector<ElementalValuesType>& rValues
        );

    /**
     * @brief Solves a small dense system with LU factorization and partial pivoting
     * @param rA System matrix (overwritten)
     * @param rB Right hand sides (overwritten with the solution)
     */
    template<std::size_t TSize, std::size_t TNumberOfRHS>
    static void SolveDenseSystem(
        BoundedMatrix<double, TSize, TSize>& rA,
        BoundedMatrix<double, TSize, TNumberOfRHS>& rB
        );

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const
    {
        return "PdeConstrainedProjectionUtility" + std::to_string(TDim) + "D";
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    const FacetTableType& mrFacetTable;

    ///@}
    ///@name Private Operations
    ///@{

    /// Gathers the local facet values from the global ones
    static void GatherLocalFacetValues(
        const FacetLinearSystem& rFacetSystem,
        const LocalSystem& rLocalSystem,
        array_1d<double, MaxFacetDofs>& rLocalFacetValues
        );

    ///@}
}; // Class PdeConstrainedProjectionUtility

///@}
///@} addtogroup block

} // namespace Kratos
