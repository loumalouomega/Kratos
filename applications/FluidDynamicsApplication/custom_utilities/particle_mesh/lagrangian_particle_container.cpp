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
#include <mutex>

// External includes

// Project includes
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"

// Application includes
#include "lagrangian_particle_container.h"

namespace Kratos
{

namespace
{

/**
 * @brief Small and fast counter based random generator (SplitMix64)
 * @details It is used instead of the standard engines because one generator is created per element and step
 */
class SplitMix64
{
public:
    explicit SplitMix64(const std::uint64_t Seed) : mState(Seed) {}

    std::uint64_t Next()
    {
        std::uint64_t z = (mState += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    /// Uniform number in (0, 1]
    double Uniform()
    {
        return (static_cast<double>(Next() >> 11) + 1.0) * (1.0 / 9007199254740992.0);
    }

private:
    std::uint64_t mState;
};

std::uint64_t CombineSeed(
    const std::uint64_t Seed,
    const std::uint64_t Counter,
    const std::uint64_t Index
    )
{
    SplitMix64 generator(Seed ^ (Counter * 0xD1B54A32D192ED03ULL) ^ (Index * 0x8CB92BA72F3D8DD7ULL));
    return generator.Next();
}

} // namespace

template<std::size_t TDim>
LagrangianParticleContainer<TDim>::LagrangianParticleContainer(
    const FacetTableType& rFacetTable,
    ModelPart& rModelPart,
    const std::uint64_t Seed
    ) : mrFacetTable(rFacetTable),
        mrModelPart(rModelPart),
        mSeed(Seed)
{
    KRATOS_TRY

    KRATOS_ERROR_IF(rModelPart.NumberOfElements() != rFacetTable.NumberOfElements())
        << "The facet table does not correspond to model part '" << rModelPart.FullName() << "'." << std::endl;

    mOffsets.assign(rFacetTable.NumberOfElements() + 1, 0);
    mpLocator = std::make_unique<BinBasedFastPointLocator<TDim>>(rModelPart);
    mpLocator->UpdateSearchDatabase();

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void LagrangianParticleContainer<TDim>::Seed(
    const std::size_t ParticlesPerElement,
    const std::vector<double>& rNodalDistance
    )
{
    KRATOS_TRY

    KRATOS_ERROR_IF(rNodalDistance.size() != mrFacetTable.NumberOfNodes()) << "Wrong nodal distance size." << std::endl;

    const std::size_t n_elements = mrFacetTable.NumberOfElements();
    mParticles.resize(n_elements * ParticlesPerElement);
    const std::uint64_t counter = mCounter++;
    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        SplitMix64 generator(CombineSeed(mSeed, counter, mrFacetTable.GetElementId(iElement)));
        const auto& r_data = mrFacetTable.GetElementData(iElement);
        ShapeFunctionsType N;
        for (std::size_t i = 0; i < ParticlesPerElement; ++i) {
            auto& r_particle = mParticles[iElement * ParticlesPerElement + i];
            r_particle.Coordinates = SamplePoint(iElement, generator);
            r_particle.Element = iElement;
            mrFacetTable.ShapeFunctions(iElement, r_particle.Coordinates, N);
            double distance = 0.0;
            for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                distance += N[i_node] * rNodalDistance[r_data.NodeIndices[i_node]];
            }
            r_particle.Distance = distance;
            r_particle.Indicator = distance < 0.0 ? 1.0 : 0.0;
        }
    });

    Rebucket();

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void LagrangianParticleContainer<TDim>::ClassifyFaces(
    const FacetTableType& rFacetTable,
    const NodalVectorValuesType& rVelocity,
    const std::vector<bool>& rIsClosedNode,
    std::vector<FaceTypesType>& rFaceTypes
    )
{
    KRATOS_TRY

    const std::size_t n_elements = rFacetTable.NumberOfElements();
    rFaceTypes.resize(n_elements);
    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const auto& r_data = rFacetTable.GetElementData(iElement);
        auto& r_types = rFaceTypes[iElement];
        for (std::size_t i_face = 0; i_face < NumFaces; ++i_face) {
            if (r_data.Facets[i_face] != FacetTableType::Invalid) {
                r_types[i_face] = FaceType::Interior;
                continue;
            }
            const auto& r_area_normal = r_data.FaceAreaNormals[i_face];
            const double area = norm_2(r_area_normal);
            array_1d<double, 3> mean_velocity = ZeroVector(3);
            bool all_closed = true;
            for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                if (i_node != i_face) {
                    const std::size_t node = r_data.NodeIndices[i_node];
                    noalias(mean_velocity) += rVelocity[node] / static_cast<double>(TDim);
                    all_closed = all_closed && rIsClosedNode[node];
                }
            }
            // Impermeable walls carry no flux even if the nodal velocity has a small normal component (e.g. at slip wall edges)
            if (all_closed) {
                r_types[i_face] = FaceType::Wall;
            } else {
                const double normal_velocity = inner_prod(mean_velocity, r_area_normal) / area;
                r_types[i_face] = normal_velocity >= 0.0 ? FaceType::Outflow : FaceType::Inflow;
            }
        }
    });

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
std::size_t LagrangianParticleContainer<TDim>::Advect(
    const NodalVectorValuesType& rOldVelocity,
    const NodalVectorValuesType& rNewVelocity,
    const std::vector<FaceTypesType>& rFaceTypes,
    const double DeltaTime,
    const double MaxCFL,
    const unsigned int RungeKuttaOrder
    )
{
    KRATOS_TRY

    KRATOS_ERROR_IF(RungeKuttaOrder < 1 || RungeKuttaOrder > 3) << "Runge-Kutta order must be 1, 2 or 3. Got " << RungeKuttaOrder << "." << std::endl;
    KRATOS_ERROR_IF(MaxCFL <= 0.0) << "Maximum CFL must be positive." << std::endl;

    constexpr std::size_t max_substeps = 1000;

    // Local advection (particles crossing a partition facet are collected to be sent to the neighbour rank)
    std::vector<std::pair<Particle, AdvectionState>> migrants;
    std::mutex migrants_mutex;
    std::size_t n_lost = IndexPartition<std::size_t>(mParticles.size()).for_each<SumReduction<std::size_t>>([&](const std::size_t iParticle) -> std::size_t {
        auto& r_particle = mParticles[iParticle];
        AdvectionState state;
        noalias(state.Coordinates) = r_particle.Coordinates;
        state.Element = r_particle.Element;
        const array_1d<double, 3> initial_velocity = EvaluateVelocity(state.Element, state.Coordinates, 0.0, rOldVelocity, rNewVelocity);
        const double h = mrFacetTable.GetElementData(state.Element).Size;
        state.NumberOfSubsteps = std::min(max_substeps, std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(norm_2(initial_velocity) * DeltaTime / (MaxCFL * h)))));

        const auto result = IntegrateTrajectory(state, rOldVelocity, rNewVelocity, rFaceTypes, DeltaTime, RungeKuttaOrder);
        if (result == AdvectionResult::Completed) {
            noalias(r_particle.Coordinates) = state.Coordinates;
            r_particle.Element = state.Element;
            return 0;
        }
        if (result == AdvectionResult::Migrating) {
            std::lock_guard<std::mutex> lock(migrants_mutex);
            migrants.push_back({r_particle, state});
        }
        r_particle.Element = DeletedParticle;
        return result == AdvectionResult::Deleted ? 1 : 0;
    });

    // Send the particles to the neighbour ranks until none is in flight (a particle may cross several partitions in one step)
    n_lost += ExchangeMigrants(migrants, rOldVelocity, rNewVelocity, rFaceTypes, DeltaTime, RungeKuttaOrder);

    Rebucket();

    return n_lost;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename LagrangianParticleContainer<TDim>::AdvectionResult LagrangianParticleContainer<TDim>::IntegrateTrajectory(
    AdvectionState& rState,
    const NodalVectorValuesType& rOldVelocity,
    const NodalVectorValuesType& rNewVelocity,
    const std::vector<FaceTypesType>& rFaceTypes,
    const double DeltaTime,
    const unsigned int RungeKuttaOrder
    ) const
{
    const std::size_t n_substeps = rState.NumberOfSubsteps;
    const double dt = DeltaTime / static_cast<double>(n_substeps);
    array_1d<double, 3>& x = rState.Coordinates;
    std::size_t& element = rState.Element;

    for (std::size_t i_substep = rState.Substep; i_substep < n_substeps; ++i_substep) {
        const double t_0 = static_cast<double>(i_substep) / static_cast<double>(n_substeps);
        const double t_1 = static_cast<double>(i_substep + 1) / static_cast<double>(n_substeps);
        const double t_mid = 0.5 * (t_0 + t_1);

        // Explicit Runge-Kutta stages (stage points outside the local mesh use the clipped velocity of the last visited element)
        const array_1d<double, 3> k_1 = EvaluateVelocity(element, x, t_0, rOldVelocity, rNewVelocity);
        array_1d<double, 3> x_new;
        if (RungeKuttaOrder == 1) {
            noalias(x_new) = x + dt * k_1;
        } else if (RungeKuttaOrder == 2) {
            const array_1d<double, 3> x_1 = x + dt * k_1;
            const array_1d<double, 3> k_2 = EvaluateVelocity(Locate(x_1, element).Element, x_1, t_1, rOldVelocity, rNewVelocity);
            noalias(x_new) = x + 0.5 * dt * (k_1 + k_2);
        } else {
            const array_1d<double, 3> x_1 = x + 0.5 * dt * k_1;
            const array_1d<double, 3> k_2 = EvaluateVelocity(Locate(x_1, element).Element, x_1, t_mid, rOldVelocity, rNewVelocity);
            const array_1d<double, 3> x_2 = x - dt * k_1 + 2.0 * dt * k_2;
            const array_1d<double, 3> k_3 = EvaluateVelocity(Locate(x_2, element).Element, x_2, t_1, rOldVelocity, rNewVelocity);
            noalias(x_new) = x + (dt / 6.0) * (k_1 + 4.0 * k_2 + k_3);
        }

        rState.Substep = i_substep + 1;
        const auto result = PlacePoint(x_new, element, rFaceTypes, rState);
        if (result != AdvectionResult::Completed) {
            return result;
        }
    }

    return AdvectionResult::Completed;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename LagrangianParticleContainer<TDim>::AdvectionResult LagrangianParticleContainer<TDim>::PlacePoint(
    array_1d<double, 3>& rNewCoordinates,
    const std::size_t StartElement,
    const std::vector<FaceTypesType>& rFaceTypes,
    AdvectionState& rState
    ) const
{
    auto location = Locate(rNewCoordinates, StartElement);
    if (!location.Found && location.ExitRank == FacetTableType::Invalid) {
        if (location.ExitFace == -1 || rFaceTypes[location.Element][location.ExitFace] == FaceType::Outflow) {
            return AdvectionResult::Deleted;
        }

        // Closed boundary: put the particle back slightly inside the domain along the face normal
        const auto& r_data = mrFacetTable.GetElementData(location.Element);
        const auto& r_area_normal = r_data.FaceAreaNormals[location.ExitFace];
        const array_1d<double, 3> unit_normal = r_area_normal / norm_2(r_area_normal);
        const std::size_t face_node = (location.ExitFace + 1) % NumNodes;
        const auto& r_face_point = (mrModelPart.NodesBegin() + r_data.NodeIndices[face_node])->Coordinates();
        const double outside_distance = inner_prod(rNewCoordinates - r_face_point, unit_normal);
        array_1d<double, 3> pushed_back = rNewCoordinates - (outside_distance + 1.0e-6 * r_data.Size) * unit_normal;
        location = Locate(pushed_back, location.Element);
        if (location.Found || location.ExitRank != FacetTableType::Invalid) {
            noalias(rNewCoordinates) = pushed_back;
        } else if (rState.Element < mrFacetTable.NumberOfElements() && rState.Substep > 0 && StartElement == rState.Element) {
            // Keep the particle where it was
            location.Found = true;
            location.Element = rState.Element;
            noalias(rNewCoordinates) = rState.Coordinates;
        } else {
            return AdvectionResult::Deleted;
        }
    }

    if (!location.Found) {
        // The point is behind a partition facet: send the particle to the neighbour rank
        const auto& r_data = mrFacetTable.GetElementData(location.Element);
        rState.DestinationRank = location.ExitRank;
        rState.PartitionFacet = mrFacetTable.GetFacetGlobalId(static_cast<std::size_t>(r_data.Facets[location.ExitFace]));
        noalias(rState.Coordinates) = rNewCoordinates;
        return AdvectionResult::Migrating;
    }

    noalias(rState.Coordinates) = rNewCoordinates;
    rState.Element = location.Element;
    return AdvectionResult::Completed;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
std::size_t LagrangianParticleContainer<TDim>::ExchangeMigrants(
    std::vector<std::pair<Particle, AdvectionState>>& rMigrants,
    const NodalVectorValuesType& rOldVelocity,
    const NodalVectorValuesType& rNewVelocity,
    const std::vector<FaceTypesType>& rFaceTypes,
    const double DeltaTime,
    const unsigned int RungeKuttaOrder
    )
{
    KRATOS_TRY

    constexpr std::size_t max_rounds = 50;
    constexpr std::size_t buffer_size = 8; // x, y, z, indicator, distance, substep, number of substeps, partition facet

    const auto& r_data_communicator = mrFacetTable.GetDataCommunicator();
    if (!mrFacetTable.IsDistributed()) {
        KRATOS_ERROR_IF_NOT(rMigrants.empty()) << "Particles crossing partition facets in a non-distributed mesh." << std::endl;
        return 0;
    }

    std::vector<int> neighbours;
    for (const int neighbour : mrModelPart.GetCommunicator().NeighbourIndices()) {
        if (neighbour >= 0) {
            neighbours.push_back(neighbour);
        }
    }

    std::size_t n_lost = 0;
    for (std::size_t i_round = 0; i_round < max_rounds; ++i_round) {
        const std::size_t n_in_flight = r_data_communicator.SumAll(static_cast<unsigned long>(rMigrants.size()));
        if (n_in_flight == 0) {
            return n_lost;
        }

        // Pack and exchange with every neighbour (empty buffers included, so that all the ranks take part)
        std::vector<std::pair<Particle, AdvectionState>> arrivals;
        for (const int neighbour : neighbours) {
            std::vector<double> send_buffer;
            for (const auto& [r_particle, r_state] : rMigrants) {
                if (r_state.DestinationRank == neighbour) {
                    send_buffer.insert(send_buffer.end(), {r_state.Coordinates[0], r_state.Coordinates[1], r_state.Coordinates[2],
                        r_particle.Indicator, r_particle.Distance, static_cast<double>(r_state.Substep),
                        static_cast<double>(r_state.NumberOfSubsteps), static_cast<double>(r_state.PartitionFacet)});
                }
            }
            const std::vector<double> recv_buffer = r_data_communicator.SendRecv(send_buffer, neighbour, neighbour);
            for (std::size_t i = 0; i + buffer_size <= recv_buffer.size(); i += buffer_size) {
                Particle particle;
                AdvectionState state;
                for (std::size_t d = 0; d < 3; ++d) {
                    state.Coordinates[d] = recv_buffer[i + d];
                }
                particle.Indicator = recv_buffer[i + 3];
                particle.Distance = recv_buffer[i + 4];
                state.Substep = static_cast<std::size_t>(recv_buffer[i + 5]);
                state.NumberOfSubsteps = static_cast<std::size_t>(recv_buffer[i + 6]);
                state.PartitionFacet = static_cast<std::size_t>(recv_buffer[i + 7]);
                arrivals.push_back({particle, state});
            }
        }
        rMigrants.clear();

        // Continue the trajectories of the arrived particles from the element behind the partition facet
        for (auto& [r_particle, r_state] : arrivals) {
            std::size_t facet_element, facet_face;
            KRATOS_ERROR_IF_NOT(mrFacetTable.FindPartitionFacet(r_state.PartitionFacet, facet_element, facet_face))
                << "Particle received through unknown partition facet " << r_state.PartitionFacet << "." << std::endl;
            r_state.Element = std::numeric_limits<std::size_t>::max();
            array_1d<double, 3> coordinates = r_state.Coordinates;
            auto result = PlacePoint(coordinates, facet_element, rFaceTypes, r_state);
            if (result == AdvectionResult::Completed) {
                result = IntegrateTrajectory(r_state, rOldVelocity, rNewVelocity, rFaceTypes, DeltaTime, RungeKuttaOrder);
            }
            if (result == AdvectionResult::Completed) {
                noalias(r_particle.Coordinates) = r_state.Coordinates;
                r_particle.Element = r_state.Element;
                mParticles.push_back(r_particle);
            } else if (result == AdvectionResult::Migrating) {
                rMigrants.push_back({r_particle, r_state});
            } else {
                ++n_lost;
            }
        }
    }

    KRATOS_WARNING_IF("LagrangianParticleContainer", r_data_communicator.SumAll(static_cast<unsigned long>(rMigrants.size())) > 0)
        << "Particles still crossing partitions after " << max_rounds << " exchange rounds were deleted." << std::endl;
    n_lost += rMigrants.size();
    rMigrants.clear();
    return n_lost;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename LagrangianParticleContainer<TDim>::PopulationStatistics LagrangianParticleContainer<TDim>::ControlPopulation(
    const std::size_t MinPerElement,
    const std::size_t MaxPerElement,
    const std::vector<FaceTypesType>& rFaceTypes,
    const std::vector<double>& rNodalDistance,
    const std::vector<ElementalValuesType>& rElementalIndicator,
    const std::vector<ElementalValuesType>& rElementalDistance
    )
{
    KRATOS_TRY

    KRATOS_ERROR_IF(MinPerElement > MaxPerElement) << "Minimum number of particles per element is larger than the maximum." << std::endl;

    const std::size_t n_elements = mrFacetTable.NumberOfElements();

    // Compute the new number of particles of each element
    std::vector<std::size_t> new_offsets(n_elements + 1, 0);
    PopulationStatistics statistics;
    for (std::size_t i_element = 0; i_element < n_elements; ++i_element) {
        const std::size_t count = NumberOfParticles(i_element);
        const std::size_t new_count = std::clamp(count, MinPerElement, MaxPerElement);
        statistics.Added += new_count > count ? new_count - count : 0;
        statistics.Removed += count > new_count ? count - new_count : 0;
        statistics.EmptyElements += count == 0 ? 1 : 0;
        new_offsets[i_element + 1] = new_offsets[i_element] + new_count;
    }
    if (statistics.Added == 0 && statistics.Removed == 0) {
        return statistics;
    }

    std::vector<Particle> new_particles(new_offsets.back());
    const std::uint64_t counter = mCounter++;
    struct TLS
    {
        std::vector<Particle> Local;
        std::vector<double> NearestDistance;
    };
    IndexPartition<std::size_t>(n_elements).for_each(TLS(), [&](const std::size_t iElement, TLS& rTLS) {
        auto& r_local = rTLS.Local;
        r_local.assign(ParticlesBegin(iElement), ParticlesEnd(iElement));
        const std::size_t count = r_local.size();
        const std::size_t new_count = new_offsets[iElement + 1] - new_offsets[iElement];
        const auto& r_data = mrFacetTable.GetElementData(iElement);

        if (count > new_count) {
            // Remove the particles with the smallest nearest-neighbour distance
            auto& r_nearest = rTLS.NearestDistance;
            while (r_local.size() > new_count) {
                r_nearest.assign(r_local.size(), std::numeric_limits<double>::max());
                for (std::size_t i = 0; i < r_local.size(); ++i) {
                    for (std::size_t j = i + 1; j < r_local.size(); ++j) {
                        const double distance = norm_2(r_local[i].Coordinates - r_local[j].Coordinates);
                        r_nearest[i] = std::min(r_nearest[i], distance);
                        r_nearest[j] = std::min(r_nearest[j], distance);
                    }
                }
                const std::size_t i_remove = std::distance(r_nearest.begin(), std::min_element(r_nearest.begin(), r_nearest.end()));
                r_local.erase(r_local.begin() + i_remove);
            }
        } else if (count < new_count) {
            SplitMix64 generator(CombineSeed(mSeed, counter, mrFacetTable.GetElementId(iElement)));
            const bool is_inflow_element = std::any_of(rFaceTypes[iElement].begin(), rFaceTypes[iElement].end(), [](const FaceType Type) {return Type == FaceType::Inflow;});

            // Gradient of the elemental distance (used to extrapolate the distance of the closest particle)
            array_1d<double, 3> distance_gradient = ZeroVector(3);
            for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                for (std::size_t d = 0; d < TDim; ++d) {
                    distance_gradient[d] += rElementalDistance[iElement][i_node] * r_data.DN_DX(i_node, d);
                }
            }

            ShapeFunctionsType N;
            for (std::size_t i = count; i < new_count; ++i) {
                Particle new_particle;
                new_particle.Coordinates = SamplePoint(iElement, generator);
                new_particle.Element = iElement;
                mrFacetTable.ShapeFunctions(iElement, new_particle.Coordinates, N);
                if (is_inflow_element) {
                    // Values from the nodal distance (imposed by the inlet)
                    double distance = 0.0;
                    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                        distance += N[i_node] * rNodalDistance[r_data.NodeIndices[i_node]];
                    }
                    new_particle.Distance = distance;
                    new_particle.Indicator = distance < 0.0 ? 1.0 : 0.0;
                } else if (count > 0) {
                    // Values from the closest particle (snapping the indicator)
                    std::size_t i_closest = 0;
                    double closest_distance = std::numeric_limits<double>::max();
                    for (std::size_t j = 0; j < count; ++j) {
                        const double distance = norm_2(r_local[j].Coordinates - new_particle.Coordinates);
                        if (distance < closest_distance) {
                            closest_distance = distance;
                            i_closest = j;
                        }
                    }
                    const auto& r_closest = r_local[i_closest];
                    new_particle.Indicator = r_closest.Indicator;
                    new_particle.Distance = r_closest.Distance + inner_prod(distance_gradient, new_particle.Coordinates - r_closest.Coordinates);
                } else {
                    // Values from the elemental fields
                    double indicator = 0.0;
                    double distance = 0.0;
                    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                        indicator += N[i_node] * rElementalIndicator[iElement][i_node];
                        distance += N[i_node] * rElementalDistance[iElement][i_node];
                    }
                    new_particle.Indicator = indicator >= 0.5 ? 1.0 : 0.0;
                    new_particle.Distance = distance;
                }
                r_local.push_back(new_particle);
            }
        }

        std::copy(r_local.begin(), r_local.end(), new_particles.begin() + new_offsets[iElement]);
    });

    mParticles.swap(new_particles);
    mOffsets.swap(new_offsets);

    return statistics;

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void LagrangianParticleContainer<TDim>::IncrementDistance(const std::vector<double>& rNodalIncrement)
{
    KRATOS_TRY

    KRATOS_ERROR_IF(rNodalIncrement.size() != mrFacetTable.NumberOfNodes()) << "Wrong nodal increment size." << std::endl;

    IndexPartition<std::size_t>(mParticles.size()).for_each([&](const std::size_t iParticle) {
        auto& r_particle = mParticles[iParticle];
        const auto& r_data = mrFacetTable.GetElementData(r_particle.Element);
        ShapeFunctionsType N;
        mrFacetTable.ShapeFunctions(r_particle.Element, r_particle.Coordinates, N);
        for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
            r_particle.Distance += N[i_node] * rNodalIncrement[r_data.NodeIndices[i_node]];
        }
    });

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename LagrangianParticleContainer<TDim>::LocationResult LagrangianParticleContainer<TDim>::Locate(
    const array_1d<double, 3>& rCoordinates,
    const std::size_t StartElement
    ) const
{
    constexpr double tolerance = 1.0e-12;
    constexpr std::size_t max_walk_steps = 500;

    LocationResult result;
    std::size_t element = StartElement;
    ShapeFunctionsType N;
    for (std::size_t i_step = 0; i_step < max_walk_steps; ++i_step) {
        mrFacetTable.ShapeFunctions(element, rCoordinates, N);
        const std::size_t i_min = std::distance(N.begin(), std::min_element(N.begin(), N.end()));
        if (N[i_min] >= -tolerance) {
            result.Found = true;
            result.Element = element;
            return result;
        }
        const auto& r_data = mrFacetTable.GetElementData(element);
        const int neighbour = r_data.Neighbours[i_min];
        if (neighbour == FacetTableType::Invalid) {
            // The walk hit the boundary of the local mesh: confirm with the bins (the domain or the partition may be non-convex)
            result = LocateWithBins(rCoordinates);
            if (!result.Found) {
                result.Element = element;
                result.ExitFace = static_cast<int>(i_min);
                result.ExitRank = r_data.PartitionRanks[i_min];
            }
            return result;
        }
        element = static_cast<std::size_t>(neighbour);
    }

    // The walk did not converge (degenerate cases): use the bins
    result = LocateWithBins(rCoordinates);
    if (!result.Found) {
        result.Element = element;
    }
    return result;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void LagrangianParticleContainer<TDim>::SetParticles(const std::vector<Particle>& rParticles)
{
    KRATOS_TRY

    mParticles = rParticles;
    const std::size_t n_elements = mrFacetTable.NumberOfElements();
    IndexPartition<std::size_t>(mParticles.size()).for_each([&](const std::size_t iParticle) {
        auto& r_particle = mParticles[iParticle];
        const std::size_t start = r_particle.Element < n_elements ? r_particle.Element : 0;
        const auto location = Locate(r_particle.Coordinates, start);
        r_particle.Element = location.Found ? location.Element : DeletedParticle;
    });
    Rebucket();

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void LagrangianParticleContainer<TDim>::Rebucket()
{
    KRATOS_TRY

    // Counting sort by host element (deleted particles are dropped)
    const std::size_t n_elements = mrFacetTable.NumberOfElements();
    mOffsets.assign(n_elements + 1, 0);
    for (const auto& r_particle : mParticles) {
        if (r_particle.Element != DeletedParticle) {
            ++mOffsets[r_particle.Element + 1];
        }
    }
    for (std::size_t i_element = 0; i_element < n_elements; ++i_element) {
        mOffsets[i_element + 1] += mOffsets[i_element];
    }

    std::vector<Particle> sorted_particles(mOffsets.back());
    std::vector<std::size_t> position(mOffsets.begin(), mOffsets.end() - 1);
    for (const auto& r_particle : mParticles) {
        if (r_particle.Element != DeletedParticle) {
            sorted_particles[position[r_particle.Element]++] = r_particle;
        }
    }
    mParticles.swap(sorted_particles);

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
typename LagrangianParticleContainer<TDim>::LocationResult LagrangianParticleContainer<TDim>::LocateWithBins(const array_1d<double, 3>& rCoordinates) const
{
    constexpr std::size_t max_results = 1000;
    typename BinBasedFastPointLocator<TDim>::ResultContainerType results(max_results);
    Vector N;
    Element::Pointer p_element = nullptr;
    LocationResult result;
    result.Found = mpLocator->FindPointOnMesh(rCoordinates, N, p_element, results.begin(), max_results, 1.0e-9);
    if (result.Found) {
        result.Element = mrFacetTable.GetElementIndex(p_element->Id());
    }
    return result;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
array_1d<double, 3> LagrangianParticleContainer<TDim>::EvaluateVelocity(
    const std::size_t ElementIndex,
    const array_1d<double, 3>& rCoordinates,
    const double TimeFraction,
    const NodalVectorValuesType& rOldVelocity,
    const NodalVectorValuesType& rNewVelocity
    ) const
{
    ShapeFunctionsType N;
    mrFacetTable.ShapeFunctions(ElementIndex, rCoordinates, N);

    // Clip the shape functions if the point is outside the element
    double sum = 0.0;
    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
        N[i_node] = std::max(N[i_node], 0.0);
        sum += N[i_node];
    }
    N /= sum;

    const auto& r_data = mrFacetTable.GetElementData(ElementIndex);
    array_1d<double, 3> velocity = ZeroVector(3);
    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
        const std::size_t node = r_data.NodeIndices[i_node];
        noalias(velocity) += N[i_node] * ((1.0 - TimeFraction) * rOldVelocity[node] + TimeFraction * rNewVelocity[node]);
    }
    return velocity;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
template<class TRandomGenerator>
array_1d<double, 3> LagrangianParticleContainer<TDim>::SamplePoint(
    const std::size_t ElementIndex,
    TRandomGenerator& rGenerator
    ) const
{
    // Uniform sampling of the simplex from normalized exponential variates
    std::array<double, NumNodes> barycentric;
    double sum = 0.0;
    for (auto& r_value : barycentric) {
        r_value = -std::log(rGenerator.Uniform());
        sum += r_value;
    }

    const auto& r_data = mrFacetTable.GetElementData(ElementIndex);
    const auto it_node_begin = mrModelPart.NodesBegin();
    array_1d<double, 3> coordinates = ZeroVector(3);
    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
        noalias(coordinates) += (barycentric[i_node] / sum) * (it_node_begin + r_data.NodeIndices[i_node])->Coordinates();
    }
    return coordinates;
}

/***********************************************************************************/
/***********************************************************************************/

template class LagrangianParticleContainer<2>;
template class LagrangianParticleContainer<3>;

} // namespace Kratos
