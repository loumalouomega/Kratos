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
#include <cstdint>
#include <memory>
#include <vector>

// External includes

// Project includes
#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/binbased_fast_point_locator.h"

// Application includes
#include "mesh_facet_table.h"

namespace Kratos
{
///@addtogroup FluidDynamicsApplication
///@{

///@name Kratos Classes
///@{

/**
 * @class LagrangianParticleContainer
 * @ingroup FluidDynamicsApplication
 * @brief Lagrangian particles carried through a fixed simplex mesh
 * @details Each particle carries a binary phase indicator and a signed distance. The particles are stored in a flat
 * vector sorted by host element (bucketed with a counting sort after each move), so the particles of each element are
 * contiguous. Particles are advected with explicit Runge-Kutta schemes (orders 1 to 3) using the nodal velocity linearly
 * interpolated in time between the previous and the current step. The host element is found by walking through the
 * face neighbours, with a bins search as fallback. In MPI runs a particle crossing a partition facet is sent, with the state
 * of its trajectory, to the rank behind the facet, which continues its integration (exchange rounds with the neighbour ranks). The per-element number of particles is kept between a minimum and a
 * maximum (LEoPart AddDelete style): new particles copy the indicator of the closest particle in the element, and the
 * particles with the smallest nearest-neighbour distance are deleted when an element is over-populated.
 * @tparam TDim Working space dimension
 * @author Vicente Mataix Ferrandiz
 */
template<std::size_t TDim>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) LagrangianParticleContainer
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of LagrangianParticleContainer
    KRATOS_CLASS_POINTER_DEFINITION(LagrangianParticleContainer);

    /// Facet table type
    using FacetTableType = MeshFacetTable<TDim>;

    /// Number of nodes of the simplex
    static constexpr std::size_t NumNodes = TDim + 1;

    /// Number of faces of the simplex
    static constexpr std::size_t NumFaces = TDim + 1;

    /// Shape functions values container
    using ShapeFunctionsType = array_1d<double, NumNodes>;

    /// Elemental P1 values container (nodal basis)
    using ElementalValuesType = array_1d<double, NumNodes>;

    /// Nodal vector values (indexed by the node position in the model part)
    using NodalVectorValuesType = std::vector<array_1d<double, 3>>;

    /// Value used to flag a deleted particle
    static constexpr std::size_t DeletedParticle = std::numeric_limits<std::size_t>::max();

    /// Classification of the element faces
    enum class FaceType : unsigned char
    {
        Interior,
        Wall,
        Outflow,
        Inflow
    };

    /// Face types of an element
    using FaceTypesType = std::array<FaceType, NumFaces>;

    /// Lagrangian particle
    struct Particle
    {
        array_1d<double, 3> Coordinates = ZeroVector(3);    /// Particle position
        double Indicator = 0.0;                             /// Binary phase indicator (1 in the negative distance fluid)
        double Distance = 0.0;                              /// Signed distance
        std::size_t Element = DeletedParticle;              /// Host element position
    };

    /// Result of a point location
    struct LocationResult
    {
        bool Found = false;     /// True if the point is inside the mesh
        std::size_t Element = 0;/// Host element (or last element visited if not found)
        int ExitFace = -1;      /// Boundary face through which the point left the mesh (-1 if unknown)
        int ExitRank = -1;      /// Rank behind the exit face if it is a partition facet (-1 otherwise)
    };

    /// Statistics of the population control
    struct PopulationStatistics
    {
        std::size_t Added = 0;          /// Number of added particles
        std::size_t Removed = 0;        /// Number of removed particles
        std::size_t EmptyElements = 0;  /// Number of elements without particles before the control
    };

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Construct a new particle container
     * @param rFacetTable The facet table of the mesh
     * @param rModelPart The model part (used for the bins search fallback and the nodes coordinates)
     * @param Seed Random seed
     */
    LagrangianParticleContainer(
        const FacetTableType& rFacetTable,
        ModelPart& rModelPart,
        const std::uint64_t Seed
        );

    /// Destructor.
    ~LagrangianParticleContainer() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Seeds a fixed number of random particles per element
     * @param ParticlesPerElement Number of particles per element
     * @param rNodalDistance Nodal signed distance (interpolated to set the particle distance; the indicator is 1 where it is negative)
     */
    void Seed(
        const std::size_t ParticlesPerElement,
        const std::vector<double>& rNodalDistance
        );

    /**
     * @brief Classifies the boundary faces of each element
     * @details Boundary faces whose nodes are all closed (impermeable) are Wall. The other boundary faces are open: Outflow if the
     * face averaged normal velocity is non-negative and Inflow otherwise. Wall faces carry no indicator flux and push particles back.
     * @param rFacetTable The facet table of the mesh
     * @param rVelocity Nodal velocity
     * @param rIsClosedNode Impermeable flag of each node (e.g. slip or no-slip wall nodes)
     * @param rFaceTypes Face types of each element
     */
    static void ClassifyFaces(
        const FacetTableType& rFacetTable,
        const NodalVectorValuesType& rVelocity,
        const std::vector<bool>& rIsClosedNode,
        std::vector<FaceTypesType>& rFaceTypes
        );

    /**
     * @brief Advects the particles
     * @param rOldVelocity Nodal velocity at the beginning of the step
     * @param rNewVelocity Nodal velocity at the end of the step
     * @param rFaceTypes Face types of each element (particles leaving through outflow faces are deleted)
     * @param DeltaTime Time step
     * @param MaxCFL Maximum particle CFL number per substep
     * @param RungeKuttaOrder Order of the explicit Runge-Kutta scheme (1, 2 or 3)
     * @return std::size_t Number of particles that left the domain
     */
    std::size_t Advect(
        const NodalVectorValuesType& rOldVelocity,
        const NodalVectorValuesType& rNewVelocity,
        const std::vector<FaceTypesType>& rFaceTypes,
        const double DeltaTime,
        const double MaxCFL,
        const unsigned int RungeKuttaOrder
        );

    /**
     * @brief Keeps the number of particles of each element between the given bounds
     * @param MinPerElement Minimum number of particles per element
     * @param MaxPerElement Maximum number of particles per element
     * @param rFaceTypes Face types (new particles in elements with inflow faces take their values from the nodal distance)
     * @param rNodalDistance Current nodal distance
     * @param rElementalIndicator Elemental indicator field (used for elements without particles)
     * @param rElementalDistance Elemental distance field (used to extrapolate the distance of new particles)
     * @return PopulationStatistics Statistics of the operation
     */
    PopulationStatistics ControlPopulation(
        const std::size_t MinPerElement,
        const std::size_t MaxPerElement,
        const std::vector<FaceTypesType>& rFaceTypes,
        const std::vector<double>& rNodalDistance,
        const std::vector<ElementalValuesType>& rElementalIndicator,
        const std::vector<ElementalValuesType>& rElementalDistance
        );

    /**
     * @brief Adds an interpolated nodal increment to the particles distance (FLIP-like mesh to particle update)
     * @param rNodalIncrement Nodal distance increment
     */
    void IncrementDistance(const std::vector<double>& rNodalIncrement);

    /**
     * @brief Locates a point in the mesh walking from a starting element
     * @param rCoordinates Point coordinates
     * @param StartElement Starting element
     * @return LocationResult The location result
     */
    LocationResult Locate(
        const array_1d<double, 3>& rCoordinates,
        const std::size_t StartElement
        ) const;

    /**
     * @brief Replaces the particles (they are located and bucketed)
     * @param rParticles The new particles (their host element is used as starting point of the location if it is valid)
     */
    void SetParticles(const std::vector<Particle>& rParticles);

    ///@}
    ///@name Access
    ///@{

    /// Returns the total number of particles
    std::size_t NumberOfParticles() const
    {
        return mParticles.size();
    }

    /// Returns the number of particles of an element
    std::size_t NumberOfParticles(const std::size_t ElementIndex) const
    {
        return mOffsets[ElementIndex + 1] - mOffsets[ElementIndex];
    }

    /// Returns a pointer to the first particle of an element
    const Particle* ParticlesBegin(const std::size_t ElementIndex) const
    {
        return mParticles.data() + mOffsets[ElementIndex];
    }

    /// Returns a pointer past the last particle of an element
    const Particle* ParticlesEnd(const std::size_t ElementIndex) const
    {
        return mParticles.data() + mOffsets[ElementIndex + 1];
    }

    /// Returns all the particles (sorted by host element)
    const std::vector<Particle>& GetParticles() const
    {
        return mParticles;
    }

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const
    {
        return "LagrangianParticleContainer" + std::to_string(TDim) + "D";
    }

    ///@}
private:
    ///@name Member Variables
    ///@{

    const FacetTableType& mrFacetTable;

    ModelPart& mrModelPart;

    std::uint64_t mSeed;

    std::uint64_t mCounter = 0;

    std::vector<Particle> mParticles;

    std::vector<std::size_t> mOffsets;

    std::unique_ptr<BinBasedFastPointLocator<TDim>> mpLocator;

    ///@}
    ///@name Private Operations
    ///@{

    /// Result of the integration of a trajectory
    enum class AdvectionResult
    {
        Completed,
        Deleted,
        Migrating
    };

    /// State of a trajectory being integrated (it can be sent to another rank)
    struct AdvectionState
    {
        array_1d<double, 3> Coordinates = ZeroVector(3);   /// Current position
        std::size_t Element = 0;                           /// Current host element
        std::size_t Substep = 0;                           /// Next substep to integrate
        std::size_t NumberOfSubsteps = 1;                  /// Total number of substeps
        int DestinationRank = -1;                          /// Rank to send the particle to
        std::size_t PartitionFacet = 0;                    /// Global partition facet crossed by the particle
    };

    /// Integrates the remaining substeps of a trajectory
    AdvectionResult IntegrateTrajectory(
        AdvectionState& rState,
        const NodalVectorValuesType& rOldVelocity,
        const NodalVectorValuesType& rNewVelocity,
        const std::vector<FaceTypesType>& rFaceTypes,
        const double DeltaTime,
        const unsigned int RungeKuttaOrder
        ) const;

    /// Locates the end point of a substep and handles the boundaries (deletion, push back or migration)
    AdvectionResult PlacePoint(
        array_1d<double, 3>& rNewCoordinates,
        const std::size_t StartElement,
        const std::vector<FaceTypesType>& rFaceTypes,
        AdvectionState& rState
        ) const;

    /// Sends the particles crossing partition facets to the neighbour ranks and continues their trajectories (returns the number of lost particles)
    std::size_t ExchangeMigrants(
        std::vector<std::pair<Particle, AdvectionState>>& rMigrants,
        const NodalVectorValuesType& rOldVelocity,
        const NodalVectorValuesType& rNewVelocity,
        const std::vector<FaceTypesType>& rFaceTypes,
        const double DeltaTime,
        const unsigned int RungeKuttaOrder
        );

    /// Removes the deleted particles and sorts the rest by host element
    void Rebucket();

    /// Locates a point with the bins search
    LocationResult LocateWithBins(const array_1d<double, 3>& rCoordinates) const;

    /// Evaluates the velocity at a point of an element (the shape functions are clipped if the point is outside)
    array_1d<double, 3> EvaluateVelocity(
        const std::size_t ElementIndex,
        const array_1d<double, 3>& rCoordinates,
        const double TimeFraction,
        const NodalVectorValuesType& rOldVelocity,
        const NodalVectorValuesType& rNewVelocity
        ) const;

    /// Samples a random point inside an element
    template<class TRandomGenerator>
    array_1d<double, 3> SamplePoint(
        const std::size_t ElementIndex,
        TRandomGenerator& rGenerator
        ) const;

    ///@}
}; // Class LagrangianParticleContainer

///@}
///@} addtogroup block

} // namespace Kratos
