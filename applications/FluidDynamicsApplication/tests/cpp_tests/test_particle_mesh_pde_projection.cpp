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
#include "containers/model.h"
#include "geometries/hexahedra_3d_8.h"
#include "geometries/quadrilateral_2d_4.h"
#include "linear_solvers/skyline_lu_factorization_solver.h"
#include "processes/structured_mesh_generator_process.h"
#include "testing/testing.h"

// Application includes
#include "custom_processes/particle_mesh_pde_projection_process.h"
#include "custom_utilities/particle_mesh/facet_linear_system.h"
#include "custom_utilities/particle_mesh/lagrangian_particle_container.h"
#include "custom_utilities/particle_mesh/mesh_facet_table.h"
#include "custom_utilities/particle_mesh/pde_constrained_projection_utility.h"
#include "tests/cpp_tests/fluid_dynamics_fast_suite.h"

namespace Kratos::Testing
{

namespace
{

using SolverType = SkylineLUFactorizationSolver<DefaultSparseSpaceType, DefaultLocalSpaceType>;

ModelPart& CreateUnitSquareModelPart(
    Model& rModel,
    const int Divisions
    )
{
    auto& r_model_part = rModel.CreateModelPart("Square", 2);
    r_model_part.AddNodalSolutionStepVariable(DISTANCE);
    r_model_part.AddNodalSolutionStepVariable(VELOCITY);
    auto p_point_1 = Kratos::make_intrusive<Node>(1, 0.0, 0.0, 0.0);
    auto p_point_2 = Kratos::make_intrusive<Node>(2, 0.0, 1.0, 0.0);
    auto p_point_3 = Kratos::make_intrusive<Node>(3, 1.0, 1.0, 0.0);
    auto p_point_4 = Kratos::make_intrusive<Node>(4, 1.0, 0.0, 0.0);
    Quadrilateral2D4<Node> geometry(p_point_1, p_point_2, p_point_3, p_point_4);
    Parameters mesher_parameters(R"({
        "number_of_divisions"        : 1,
        "element_name"               : "Element2D3N",
        "condition_name"             : "LineCondition2D2N",
        "create_skin_sub_model_part" : false
    })");
    mesher_parameters["number_of_divisions"].SetInt(Divisions);
    StructuredMeshGeneratorProcess(geometry, r_model_part, mesher_parameters).Execute();
    return r_model_part;
}

ModelPart& CreateUnitCubeModelPart(
    Model& rModel,
    const int Divisions
    )
{
    auto& r_model_part = rModel.CreateModelPart("Cube", 2);
    r_model_part.AddNodalSolutionStepVariable(DISTANCE);
    r_model_part.AddNodalSolutionStepVariable(VELOCITY);
    auto p_point_1 = Kratos::make_intrusive<Node>(1, 0.0, 0.0, 0.0);
    auto p_point_2 = Kratos::make_intrusive<Node>(2, 1.0, 0.0, 0.0);
    auto p_point_3 = Kratos::make_intrusive<Node>(3, 1.0, 1.0, 0.0);
    auto p_point_4 = Kratos::make_intrusive<Node>(4, 0.0, 1.0, 0.0);
    auto p_point_5 = Kratos::make_intrusive<Node>(5, 0.0, 0.0, 1.0);
    auto p_point_6 = Kratos::make_intrusive<Node>(6, 1.0, 0.0, 1.0);
    auto p_point_7 = Kratos::make_intrusive<Node>(7, 1.0, 1.0, 1.0);
    auto p_point_8 = Kratos::make_intrusive<Node>(8, 0.0, 1.0, 1.0);
    Hexahedra3D8<Node> geometry(p_point_1, p_point_2, p_point_3, p_point_4, p_point_5, p_point_6, p_point_7, p_point_8);
    Parameters mesher_parameters(R"({
        "number_of_divisions"        : 1,
        "element_name"               : "Element3D4N",
        "condition_name"             : "SurfaceCondition3D3N",
        "create_skin_sub_model_part" : false
    })");
    mesher_parameters["number_of_divisions"].SetInt(Divisions);
    StructuredMeshGeneratorProcess(geometry, r_model_part, mesher_parameters).Execute();
    return r_model_part;
}

template<class TFunction>
std::vector<double> EvaluateAtNodes(
    const ModelPart& rModelPart,
    TFunction&& rFunction
    )
{
    std::vector<double> values;
    for (const auto& r_node : rModelPart.Nodes()) {
        values.push_back(rFunction(r_node.Coordinates()));
    }
    return values;
}

template<std::size_t TDim>
void CheckFacetTableConsistency(const MeshFacetTable<TDim>& rTable)
{
    for (std::size_t i_element = 0; i_element < rTable.NumberOfElements(); ++i_element) {
        const auto& r_data = rTable.GetElementData(i_element);
        for (std::size_t i_face = 0; i_face < TDim + 1; ++i_face) {
            const int neighbour = r_data.Neighbours[i_face];
            if (neighbour == MeshFacetTable<TDim>::Invalid) {
                KRATOS_EXPECT_EQ(r_data.Facets[i_face], MeshFacetTable<TDim>::Invalid);
                continue;
            }
            const auto& r_neighbour_data = rTable.GetElementData(neighbour);
            const auto it_face = std::find(r_neighbour_data.Facets.begin(), r_neighbour_data.Facets.end(), r_data.Facets[i_face]);
            KRATOS_EXPECT_TRUE(it_face != r_neighbour_data.Facets.end());
            const std::size_t neighbour_face = std::distance(r_neighbour_data.Facets.begin(), it_face);
            KRATOS_EXPECT_EQ(r_neighbour_data.Neighbours[neighbour_face], static_cast<int>(i_element));
            const array_1d<double, 3> opposite_area_normal = -r_neighbour_data.FaceAreaNormals[neighbour_face];
            KRATOS_EXPECT_VECTOR_NEAR(r_data.FaceAreaNormals[i_face], opposite_area_normal, 0.0);

            // Both sides must give the same facet DOF to each shared vertex
            for (std::size_t a = 0; a < TDim + 1; ++a) {
                if (a == i_face) {
                    continue;
                }
                const auto it_node = std::find(r_neighbour_data.NodeIndices.begin(), r_neighbour_data.NodeIndices.end(), r_data.NodeIndices[a]);
                KRATOS_EXPECT_TRUE(it_node != r_neighbour_data.NodeIndices.end());
                const std::size_t b = std::distance(r_neighbour_data.NodeIndices.begin(), it_node);
                KRATOS_EXPECT_EQ(rTable.FacetDof(i_element, i_face, a), rTable.FacetDof(neighbour, neighbour_face, b));
            }
        }
    }
}

} // namespace

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionFacetTable2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 2);
    const MeshFacetTable<2> table(r_model_part);

    // Interior edges of the triangulated square: E - boundary edges = (V + F - 1) - 4 n
    const std::size_t n_edges = r_model_part.NumberOfNodes() + r_model_part.NumberOfElements() - 1;
    KRATOS_EXPECT_EQ(table.NumberOfFacets(), n_edges - 8);
    KRATOS_EXPECT_EQ(table.NumberOfFacetDofs(), 2 * table.NumberOfFacets());
    CheckFacetTableConsistency(table);

    // Each face area normal must be orthogonal to the face, outward, and scaled by the face length
    for (std::size_t i_element = 0; i_element < table.NumberOfElements(); ++i_element) {
        const auto& r_data = table.GetElementData(i_element);
        double total = 0.0;
        array_1d<double, 3> sum = ZeroVector(3);
        for (std::size_t i_face = 0; i_face < 3; ++i_face) {
            noalias(sum) += r_data.FaceAreaNormals[i_face];
            total += norm_2(r_data.FaceAreaNormals[i_face]);
        }
        KRATOS_EXPECT_NEAR(norm_2(sum), 0.0, 1.0e-14);
        KRATOS_EXPECT_NEAR(total, 0.5 * (2.0 + std::sqrt(2.0)), 1.0e-14);
        KRATOS_EXPECT_NEAR(r_data.Volume, 0.125, 1.0e-14);
    }
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionFacetTable3D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitCubeModelPart(model, 2);
    const MeshFacetTable<3> table(r_model_part);

    // Each of the 6 cube faces is split into 2 n^2 boundary triangles
    const std::size_t n_boundary_faces = 6 * 2 * 4;
    KRATOS_EXPECT_EQ(table.NumberOfFacets(), (4 * r_model_part.NumberOfElements() - n_boundary_faces) / 2);
    CheckFacetTableConsistency(table);

    const double total_volume = std::accumulate(r_model_part.ElementsBegin(), r_model_part.ElementsEnd(), 0.0, [&](const double Value, const Element& rElement) {
        return Value + table.GetElementData(table.GetElementIndex(rElement.Id())).Volume;
    });
    KRATOS_EXPECT_NEAR(total_volume, 1.0, 1.0e-12);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionNegativeVolume2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 8);
    const MeshFacetTable<2> table(r_model_part);
    using UtilityType = PdeConstrainedProjectionUtility<2>;

    // A planar level set is integrated exactly
    const auto plane = EvaluateAtNodes(r_model_part, [](const array_1d<double, 3>& rX) {return rX[0] - 0.3;});
    KRATOS_EXPECT_NEAR(UtilityType::ComputeNegativeVolume(table, r_model_part, plane, 0.0), 0.3, 1.0e-12);
    KRATOS_EXPECT_NEAR(UtilityType::ComputeNegativeVolume(table, r_model_part, plane, -0.25), 0.55, 1.0e-12);

    // The initial indicator integrates to the negative volume of the nodal distance
    const auto circle = EvaluateAtNodes(r_model_part, [](const array_1d<double, 3>& rX) {return std::sqrt(std::pow(rX[0] - 0.5, 2) + std::pow(rX[1] - 0.5, 2)) - 0.3;});
    std::vector<UtilityType::ElementalValuesType> indicator;
    UtilityType::InitializeIndicatorFromDistance(table, r_model_part, circle, indicator);
    KRATOS_EXPECT_NEAR(UtilityType::ComputeIntegral(table, indicator), UtilityType::ComputeNegativeVolume(table, r_model_part, circle, 0.0), 1.0e-12);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionLocalSystem2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 4);
    const MeshFacetTable<2> table(r_model_part);
    LagrangianParticleContainer<2> particles(table, r_model_part, 1234);
    const auto distance = EvaluateAtNodes(r_model_part, [](const array_1d<double, 3>& rX) {return rX[0] + 0.3 * rX[1] - 0.6;});
    particles.Seed(10, distance);

    using UtilityType = PdeConstrainedProjectionUtility<2>;
    std::vector<array_1d<double, 3>> velocity;
    for (const auto& r_node : r_model_part.Nodes()) {
        velocity.push_back(array_1d<double, 3>{1.0 - r_node.Y(), 0.5 * r_node.X(), 0.0});
    }
    UtilityType::ProjectionSettings settings;
    settings.Beta = 1.0e-6;
    settings.Zeta = 10.0;
    settings.DeltaTime = 0.05;
    std::vector<LagrangianParticleContainer<2>::FaceTypesType> face_types;
    LagrangianParticleContainer<2>::ClassifyFaces(table, velocity, std::vector<bool>(table.NumberOfNodes(), false), face_types);

    for (std::size_t i_element = 0; i_element < table.NumberOfElements(); ++i_element) {
        UtilityType::LocalSystem local;
        UtilityType::CondensedSystem condensed;
        const UtilityType::ElementalValuesType previous{0.2, 0.7, 0.4};
        UtilityType::BuildLocalSystem(table, particles, i_element, previous, velocity, face_types[i_element], distance, settings, local);
        UtilityType::CondenseLocalSystem(local, condensed);

        // The Schur complement is symmetric
        const double scale = norm_frobenius(condensed.S);
        for (std::size_t i = 0; i < local.NumberOfFacetDofs; ++i) {
            for (std::size_t j = 0; j < local.NumberOfFacetDofs; ++j) {
                KRATOS_EXPECT_NEAR(condensed.S(i, j), condensed.S(j, i), 1.0e-12 * scale);
            }
        }

        // The local conservation statement holds for any facet values
        array_1d<double, UtilityType::MaxFacetDofs> facet_values = ZeroVector(UtilityType::MaxFacetDofs);
        for (std::size_t k = 0; k < local.NumberOfFacetDofs; ++k) {
            facet_values[k] = std::sin(1.0 + 3.0 * static_cast<double>(k + i_element));
        }
        UtilityType::ElementalValuesType indicator;
        double lagrange_multiplier;
        UtilityType::RecoverLocalSolution(local, condensed, facet_values, indicator, lagrange_multiplier);
        const double volume = table.GetElementData(i_element).Volume;
        KRATOS_EXPECT_NEAR(UtilityType::ComputeCellResidual(local, indicator, facet_values) / volume, 0.0, 1.0e-12);
    }
}

namespace
{

template<std::size_t TDim>
void TestProjectionReproducesLinearField(ModelPart& rModelPart)
{
    const MeshFacetTable<TDim> table(rModelPart);
    LagrangianParticleContainer<TDim> particles(table, rModelPart, 42);
    const auto zero_distance = std::vector<double>(table.NumberOfNodes(), 1.0);
    particles.Seed(TDim == 2 ? 12 : 20, zero_distance);

    // Particles sampled from a linear field and c* equal to the same field
    const auto linear_field = [](const array_1d<double, 3>& rX) {return 0.3 + 0.5 * rX[0] - 0.2 * rX[1] + 0.1 * rX[2];};
    std::vector<typename LagrangianParticleContainer<TDim>::Particle> sampled(particles.GetParticles());
    for (auto& r_particle : sampled) {
        r_particle.Indicator = linear_field(r_particle.Coordinates);
    }
    particles.SetParticles(sampled);

    using UtilityType = PdeConstrainedProjectionUtility<TDim>;
    std::vector<typename UtilityType::ElementalValuesType> indicator(table.NumberOfElements());
    std::vector<typename UtilityType::ElementalValuesType> expected(table.NumberOfElements());
    const auto it_node_begin = rModelPart.NodesBegin();
    for (std::size_t i_element = 0; i_element < table.NumberOfElements(); ++i_element) {
        const auto& r_data = table.GetElementData(i_element);
        for (std::size_t a = 0; a < TDim + 1; ++a) {
            expected[i_element][a] = linear_field((it_node_begin + r_data.NodeIndices[a])->Coordinates());
        }
    }
    indicator = expected;

    // With zero velocity the constraint is inactive and P_E(P_L(c)) = c
    const std::vector<array_1d<double, 3>> velocity(table.NumberOfNodes(), ZeroVector(3));
    typename UtilityType::ProjectionSettings settings;
    settings.Zeta = 0.0;
    settings.DeltaTime = 0.1;
    SerialFacetLinearSystem facet_system(Kratos::make_shared<SolverType>());
    UtilityType projection(table);
    std::vector<typename LagrangianParticleContainer<TDim>::FaceTypesType> face_types;
    LagrangianParticleContainer<TDim>::ClassifyFaces(table, velocity, std::vector<bool>(table.NumberOfNodes(), false), face_types);
    const auto info = projection.Project(particles, velocity, face_types, zero_distance, settings, facet_system, indicator);
    for (std::size_t i_element = 0; i_element < table.NumberOfElements(); ++i_element) {
        KRATOS_EXPECT_VECTOR_NEAR(indicator[i_element], expected[i_element], 1.0e-9);
    }
    KRATOS_EXPECT_NEAR(info.Integral, info.PreviousIntegral, 1.0e-13);
    KRATOS_EXPECT_NEAR(info.MaxCellResidual, 0.0, 1.0e-12);
}

template<std::size_t TDim>
void TestProjectionIsConservative(
    ModelPart& rModelPart,
    const bool OutflowFromLevelSet
    )
{
    const MeshFacetTable<TDim> table(rModelPart);
    LagrangianParticleContainer<TDim> particles(table, rModelPart, 7);
    const auto particle_distance = EvaluateAtNodes(rModelPart, [](const array_1d<double, 3>& rX) {return rX[0] - 0.45;});
    particles.Seed(TDim == 2 ? 15 : 25, particle_distance);

    // Previous field disagreeing with the particles so that the constraint is active
    using UtilityType = PdeConstrainedProjectionUtility<TDim>;
    // Asymmetric so that the boundary fluxes of the rotation do not cancel
    const auto previous_distance = EvaluateAtNodes(rModelPart, [](const array_1d<double, 3>& rX) {return rX[1] + 0.3 * rX[0] - 0.5;});
    std::vector<typename UtilityType::ElementalValuesType> indicator;
    UtilityType::InitializeIndicatorFromDistance(table, rModelPart, previous_distance, indicator);
    const double previous_integral = UtilityType::ComputeIntegral(table, indicator);

    // Rigid rotation crossing the boundary of the box (inflow and outflow)
    std::vector<array_1d<double, 3>> velocity;
    for (const auto& r_node : rModelPart.Nodes()) {
        velocity.push_back(array_1d<double, 3>{-(r_node.Y() - 0.5), r_node.X() - 0.5, 0.0});
    }
    typename UtilityType::ProjectionSettings settings;
    settings.Zeta = 15.0;
    settings.DeltaTime = 0.02;
    settings.OutflowFromLevelSet = OutflowFromLevelSet;
    SerialFacetLinearSystem facet_system(Kratos::make_shared<SolverType>());
    UtilityType projection(table);
    std::vector<typename LagrangianParticleContainer<TDim>::FaceTypesType> face_types;
    LagrangianParticleContainer<TDim>::ClassifyFaces(table, velocity, std::vector<bool>(table.NumberOfNodes(), false), face_types);
    const auto info = projection.Project(particles, velocity, face_types, previous_distance, settings, facet_system, indicator);

    KRATOS_EXPECT_NEAR(info.PreviousIntegral, previous_integral, 1.0e-14);
    KRATOS_EXPECT_NEAR(info.Integral, UtilityType::ComputeIntegral(table, indicator), 1.0e-14);
    KRATOS_EXPECT_NEAR(info.MaxCellResidual, 0.0, 1.0e-11);
    KRATOS_EXPECT_NEAR(info.Integral - info.PreviousIntegral + info.NetBoundaryOutflow, 0.0, 1.0e-13);
    KRATOS_EXPECT_GT(std::abs(info.NetBoundaryOutflow), 1.0e-6);
}

} // namespace

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionLinearField2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    TestProjectionReproducesLinearField<2>(CreateUnitSquareModelPart(model, 6));
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionLinearField3D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    TestProjectionReproducesLinearField<3>(CreateUnitCubeModelPart(model, 3));
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionConservation2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    TestProjectionIsConservative<2>(CreateUnitSquareModelPart(model, 10), true);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionConservationTraceOutflow2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    TestProjectionIsConservative<2>(CreateUnitSquareModelPart(model, 10), false);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionConservation3D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    TestProjectionIsConservative<3>(CreateUnitCubeModelPart(model, 4), true);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionParticleAdvection2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 10);
    const MeshFacetTable<2> table(r_model_part);
    LagrangianParticleContainer<2> particles(table, r_model_part, 3);

    // A single particle advected by a rigid rotation returns to its position after one revolution
    LagrangianParticleContainer<2>::Particle particle;
    particle.Coordinates = array_1d<double, 3>{0.75, 0.5, 0.0};
    particle.Indicator = 1.0;
    particles.SetParticles({particle});
    std::vector<array_1d<double, 3>> velocity;
    for (const auto& r_node : r_model_part.Nodes()) {
        velocity.push_back(array_1d<double, 3>{-(r_node.Y() - 0.5), r_node.X() - 0.5, 0.0});
    }
    std::vector<LagrangianParticleContainer<2>::FaceTypesType> face_types;
    LagrangianParticleContainer<2>::ClassifyFaces(table, velocity, std::vector<bool>(table.NumberOfNodes(), false), face_types);
    const std::size_t n_steps = 100;
    const double dt = 2.0 * Globals::Pi / static_cast<double>(n_steps);
    for (std::size_t i_step = 0; i_step < n_steps; ++i_step) {
        KRATOS_EXPECT_EQ(particles.Advect(velocity, velocity, face_types, dt, 0.5, 3), 0);
    }
    KRATOS_EXPECT_EQ(particles.NumberOfParticles(), 1);
    // RK3 time integration error of the rotation (the velocity interpolation is exact for this linear field)
    KRATOS_EXPECT_VECTOR_NEAR(particles.GetParticles()[0].Coordinates, particle.Coordinates, 5.0e-5);
    const auto location = particles.Locate(particles.GetParticles()[0].Coordinates, 0);
    KRATOS_EXPECT_TRUE(location.Found);
    KRATOS_EXPECT_EQ(location.Element, particles.GetParticles()[0].Element);
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionProcessVolume2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 16);
    r_model_part.GetProcessInfo().SetValue(DELTA_TIME, 0.01);
    for (auto& r_node : r_model_part.Nodes()) {
        r_node.FastGetSolutionStepValue(DISTANCE) = std::sqrt(std::pow(r_node.X() - 0.5, 2) + std::pow(r_node.Y() - 0.6, 2)) - 0.2;
        // Divergence-free vortex with zero normal velocity on the box walls (closed domain)
        const double x = Globals::Pi * r_node.X();
        const double y = Globals::Pi * r_node.Y();
        const array_1d<double, 3> velocity{std::sin(x) * std::cos(y), -std::cos(x) * std::sin(y), 0.0};
        r_node.FastGetSolutionStepValue(VELOCITY) = velocity;
        r_node.FastGetSolutionStepValue(VELOCITY, 1) = velocity;
    }

    Parameters parameters(R"({
        "particles_per_element" : {"initial" : 16, "minimum" : 10, "maximum" : 30},
        "zeta" : 16.0
    })");
    ParticleMeshPdeProjectionProcess<2> process(r_model_part, Kratos::make_shared<SolverType>(), parameters);
    process.ExecuteInitialize();
    const double initial_volume = process.GetIndicatorIntegral();
    KRATOS_EXPECT_NEAR(initial_volume, Globals::Pi * 0.04, 5.0e-3);

    for (std::size_t i_step = 0; i_step < 10; ++i_step) {
        process.Execute();
        const auto& r_info = process.GetLastProjectionInfo();
        KRATOS_EXPECT_NEAR(r_info.Integral - r_info.PreviousIntegral + r_info.NetBoundaryOutflow, 0.0, 1.0e-13);
        KRATOS_EXPECT_NEAR(r_info.MaxCellResidual, 0.0, 1.0e-11);
    }

    // Closed domain: the phase volume is conserved and the negative volume of the distance follows it
    KRATOS_EXPECT_NEAR(process.GetIndicatorIntegral(), initial_volume, 1.0e-12);
    std::vector<double> distance;
    for (const auto& r_node : r_model_part.Nodes()) {
        distance.push_back(r_node.FastGetSolutionStepValue(DISTANCE));
    }
    const MeshFacetTable<2> table(r_model_part);
    KRATOS_EXPECT_NEAR(PdeConstrainedProjectionUtility<2>::ComputeNegativeVolume(table, r_model_part, distance, 0.0), initial_volume, 1.0e-9);
    for (const auto& r_element : r_model_part.Elements()) {
        KRATOS_EXPECT_TRUE(r_element.Has(PHASE_FRACTION));
    }
}

KRATOS_TEST_CASE_IN_SUITE(ParticleMeshPdeProjectionParticlesOutput2D, FluidDynamicsApplicationFastSuite)
{
    Model model;
    auto& r_model_part = CreateUnitSquareModelPart(model, 8);
    r_model_part.GetProcessInfo().SetValue(DELTA_TIME, 0.01);
    for (auto& r_node : r_model_part.Nodes()) {
        r_node.FastGetSolutionStepValue(DISTANCE) = r_node.X() - 0.5;
        const array_1d<double, 3> velocity{0.1, 0.0, 0.0};
        r_node.FastGetSolutionStepValue(VELOCITY) = velocity;
        r_node.FastGetSolutionStepValue(VELOCITY, 1) = velocity;
    }

    Parameters parameters(R"({
        "particles_per_element" : {"initial" : 12, "minimum" : 6, "maximum" : 24},
        "particles_output"      : {"model_part_name" : "Particles", "interval" : 2, "stride" : 3}
    })");
    ParticleMeshPdeProjectionProcess<2> process(r_model_part, Kratos::make_shared<SolverType>(), parameters);
    process.ExecuteInitialize();
    r_model_part.GetProcessInfo().SetValue(STEP, 1);
    process.Execute();

    // The particles model part is created in the same model, shares the process info and holds every third particle
    KRATOS_EXPECT_TRUE(model.HasModelPart("Particles"));
    const auto& r_particles = model.GetModelPart("Particles");
    KRATOS_EXPECT_EQ(&r_particles.GetProcessInfo(), &r_model_part.GetProcessInfo());
    const std::size_t n_expected = (process.GetNumberOfParticles() + 2) / 3;
    KRATOS_EXPECT_EQ(r_particles.NumberOfNodes(), n_expected);
    KRATOS_EXPECT_EQ(r_particles.NumberOfConditions(), n_expected);
    IndexType expected_id = 1;
    for (const auto& r_node : r_particles.Nodes()) {
        KRATOS_EXPECT_EQ(r_node.Id(), expected_id++);
        KRATOS_EXPECT_TRUE(r_node.X() >= -1.0e-12 && r_node.X() <= 1.0 + 1.0e-12 && r_node.Y() >= -1.0e-12 && r_node.Y() <= 1.0 + 1.0e-12);
        const double indicator = r_node.GetValue(PHASE_FRACTION);
        KRATOS_EXPECT_TRUE(indicator == 0.0 || indicator == 1.0);
        KRATOS_EXPECT_TRUE(std::isfinite(r_node.GetValue(DISTANCE)));
    }
    for (const auto& r_condition : r_particles.Conditions()) {
        KRATOS_EXPECT_EQ(r_condition.GetGeometry().PointsNumber(), 1);
        KRATOS_EXPECT_EQ(r_condition.GetGeometry()[0].Id(), r_condition.Id());
    }

    // Explicit update with another stride
    ModelPart& r_all_particles = model.CreateModelPart("AllParticles");
    process.UpdateParticlesModelPart(r_all_particles);
    KRATOS_EXPECT_EQ(r_all_particles.NumberOfNodes(), process.GetNumberOfParticles());
}

} // namespace Kratos::Testing
