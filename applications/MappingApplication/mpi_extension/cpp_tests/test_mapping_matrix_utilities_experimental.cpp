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

#ifdef HAVE_TPETRA

// System includes

// External includes

// Project includes
#include "containers/model.h"
#include "testing/testing.h"
#include "mpi/utilities/model_part_communicator_utilities.h"
#include "custom_utilities/mapping_matrix_utilities.h"
#include "custom_utilities/mapper_local_system.h"
#include "../custom_utilities/mapper_mpi_define_experimental.h"
#include "mapping_application_variables.h"

namespace Kratos::Testing {

namespace {

typedef typename MPIMapperDefinitionsExperimental::SparseSpaceType ExperimentalSparseSpaceType;
typedef typename MPIMapperDefinitionsExperimental::DenseSpaceType  ExperimentalDenseSpaceType;
typedef MappingMatrixUtilities<ExperimentalSparseSpaceType, ExperimentalDenseSpaceType> ExperimentalMappingMatrixUtilitiesType;

/// Minimal MapperLocalSystem returning prescribed ids and weights
class StubMapperLocalSystem : public MapperLocalSystem
{
public:
    StubMapperLocalSystem(
        EquationIdVectorType OriginIds,
        EquationIdVectorType DestinationIds,
        Matrix LocalMatrix)
        : mStubOriginIds(std::move(OriginIds)),
          mStubDestinationIds(std::move(DestinationIds)),
          mStubMatrix(std::move(LocalMatrix))
    {}

    CoordinatesArrayType& Coordinates() const override
    {
        return mCoordinates;
    }

    void PairingInfo(std::ostream& rOStream, const int EchoLevel) const override {}

private:
    void CalculateAll(MatrixType& rLocalMappingMatrix,
                      EquationIdVectorType& rOriginIds,
                      EquationIdVectorType& rDestinationIds,
                      MapperLocalSystem::PairingStatus& rPairingStatus) const override
    {
        rLocalMappingMatrix = mStubMatrix;
        rOriginIds = mStubOriginIds;
        rDestinationIds = mStubDestinationIds;
        rPairingStatus = MapperLocalSystem::PairingStatus::InterfaceInfoFound;
    }

    EquationIdVectorType mStubOriginIds;
    EquationIdVectorType mStubDestinationIds;
    Matrix mStubMatrix;
    mutable CoordinatesArrayType mCoordinates = ZeroVector(3);
};

/// Creates a distributed interface model part with NumLocalNodes local nodes whose
/// INTERFACE_EQUATION_IDs are [FirstEquationId, FirstEquationId + NumLocalNodes)
ModelPart& CreateInterfaceModelPart(
    Model& rModel,
    const std::string& rName,
    const int NumLocalNodes,
    const int FirstEquationId,
    const int Rank)
{
    ModelPart& r_model_part = rModel.CreateModelPart(rName);
    ModelPartCommunicatorUtilities::SetMPICommunicator(r_model_part);

    for (int i = 0; i < NumLocalNodes; ++i) {
        // Globally unique node ids
        auto p_node = r_model_part.CreateNewNode(FirstEquationId + i + 1, static_cast<double>(FirstEquationId + i), 0.0, 0.0);
        p_node->SetValue(INTERFACE_EQUATION_ID, FirstEquationId + i);
        r_model_part.GetCommunicator().LocalMesh().AddNode(p_node);
    }

    return r_model_part;
}

} // namespace

/**
 * Builds a distributed rectangular mapping matrix from prescribed local systems, including
 * contributions to rows owned by the neighbouring rank, and verifies M * x on a known x.
 */
KRATOS_DISTRIBUTED_TEST_CASE_IN_SUITE(TrilinosExperimentalBuildMappingMatrix, KratosMappingApplicationMPITestSuite)
{
    Model current_model;

    const auto& r_comm = Testing::GetDefaultDataCommunicator();
    const int rank = r_comm.Rank();
    const int world_size = r_comm.Size();

    constexpr int num_local = 2;
    const int global_size = num_local * world_size;
    const int first_id = rank * num_local;

    auto& r_model_part_origin = CreateInterfaceModelPart(current_model, "Origin", num_local, first_id, rank);
    auto& r_model_part_destination = CreateInterfaceModelPart(current_model, "Destination", num_local, first_id, rank);

    // Local systems:
    // 1) destination row (2 rank) <- origin cols {2 rank, 2 rank + 1} with weights {0.6, 0.4}
    // 2) destination row of the NEXT rank ((2 rank + 2) mod N) <- origin col {2 rank + 1} with
    //    weight 1.0 (exercises the insertion into rows owned by another rank)
    std::vector<Kratos::unique_ptr<MapperLocalSystem>> mapper_local_systems;

    Matrix local_matrix_1(1, 2);
    local_matrix_1(0, 0) = 0.6;
    local_matrix_1(0, 1) = 0.4;
    mapper_local_systems.push_back(Kratos::make_unique<StubMapperLocalSystem>(
        MapperLocalSystem::EquationIdVectorType{static_cast<std::size_t>(first_id), static_cast<std::size_t>(first_id + 1)},
        MapperLocalSystem::EquationIdVectorType{static_cast<std::size_t>(first_id)},
        local_matrix_1));

    Matrix local_matrix_2(1, 1);
    local_matrix_2(0, 0) = 1.0;
    const int remote_row = (first_id + 2) % global_size;
    mapper_local_systems.push_back(Kratos::make_unique<StubMapperLocalSystem>(
        MapperLocalSystem::EquationIdVectorType{static_cast<std::size_t>(first_id + 1)},
        MapperLocalSystem::EquationIdVectorType{static_cast<std::size_t>(remote_row)},
        local_matrix_2));

    // Build the mapping matrix and interface vectors
    Kratos::unique_ptr<typename ExperimentalSparseSpaceType::MatrixType> p_mapping_matrix;
    Kratos::unique_ptr<typename ExperimentalSparseSpaceType::VectorType> p_vector_origin;
    Kratos::unique_ptr<typename ExperimentalSparseSpaceType::VectorType> p_vector_destination;

    ExperimentalMappingMatrixUtilitiesType::BuildMappingMatrix(
        p_mapping_matrix, p_vector_origin, p_vector_destination,
        r_model_part_origin, r_model_part_destination,
        mapper_local_systems, 0);

    KRATOS_EXPECT_NE(nullptr, p_mapping_matrix.get());
    KRATOS_EXPECT_NE(nullptr, p_vector_origin.get());
    KRATOS_EXPECT_NE(nullptr, p_vector_destination.get());

    // Check the matrix dimensions (rectangular: destination x origin)
    KRATOS_EXPECT_EQ(static_cast<int>(p_mapping_matrix->getGlobalNumRows()), global_size);
    KRATOS_EXPECT_EQ(static_cast<int>(p_mapping_matrix->getGlobalNumCols()), global_size);

    // Fill the origin vector with x[gid] = gid
    using LO = typename ExperimentalSparseSpaceType::LO;
    using GO = typename ExperimentalSparseSpaceType::GO;
    for (int i = 0; i < num_local; ++i) {
        p_vector_origin->replaceLocalValue(static_cast<LO>(i), std::size_t(0), static_cast<double>(first_id + i));
    }

    // Multiply: y = M * x
    ExperimentalSparseSpaceType::Mult(*p_mapping_matrix, *p_vector_origin, *p_vector_destination);

    // Expected values:
    // y[2 r]     = 0.6 (2 r) + 0.4 (2 r + 1) + ((2 r - 1 + N) mod N)   (the last term comes from the previous rank)
    // y[2 r + 1] = 0
    const auto y_data = p_vector_destination->getData(0);
    const double expected_row_even = 0.6 * first_id + 0.4 * (first_id + 1) + static_cast<double>((first_id - 1 + global_size) % global_size);
    KRATOS_EXPECT_NEAR(static_cast<double>(y_data[0]), expected_row_even, 1e-12);
    KRATOS_EXPECT_NEAR(static_cast<double>(y_data[1]), 0.0, 1e-12);
}

}  // namespace Kratos::Testing

#endif // HAVE_TPETRA
