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
// Tpetra (experimental Trilinos space) counterpart of mapping_matrix_utilities_mpi.cpp

#ifdef HAVE_TPETRA

// System includes
#include <set>

// External includes

// Project includes
#include "utilities/parallel_utilities.h"
#include "custom_utilities/mapping_matrix_utilities.h"
#include "mapper_mpi_define_experimental.h"
#include "custom_utilities/mapper_utilities.h"
#include "mapping_application_variables.h"

namespace Kratos {

namespace {

typedef typename MPIMapperDefinitionsExperimental::SparseSpaceType MappingSparseSpaceType;
typedef typename MPIMapperDefinitionsExperimental::DenseSpaceType  DenseSpaceType;

typedef MappingMatrixUtilities<MappingSparseSpaceType, DenseSpaceType> MappingMatrixUtilitiesType;

typedef typename MapperLocalSystem::MatrixType MatrixType;
typedef typename MapperLocalSystem::EquationIdVectorType EquationIdVectorType;

typedef typename MappingSparseSpaceType::LO LO;
typedef typename MappingSparseSpaceType::GO GO;
typedef typename MappingSparseSpaceType::NT NT;
typedef typename MappingSparseSpaceType::MapType MapType;
typedef typename MappingSparseSpaceType::GraphType GraphType;

void ConstructRowColIdSets(std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems,
                           std::set<GO>& rRowEquationIds,
                           std::set<GO>& rColEquationIds)
{
    EquationIdVectorType origin_ids;
    EquationIdVectorType destination_ids;

    for (auto& rp_local_sys : rMapperLocalSystems) {
        rp_local_sys->EquationIdVectors(origin_ids, destination_ids);

        rRowEquationIds.insert(destination_ids.begin(), destination_ids.end());
        rColEquationIds.insert(origin_ids.begin(), origin_ids.end());
    }
}

void ConstructMatrixStructure(GraphType& rGraph,
                              std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems)
{
    EquationIdVectorType origin_ids;
    EquationIdVectorType destination_ids;

    std::vector<GO> col_gids;

    for (auto& rp_local_sys : rMapperLocalSystems) {
        rp_local_sys->EquationIdVectors(origin_ids, destination_ids);

        if (origin_ids.size() > 0) {
            col_gids.assign(origin_ids.begin(), origin_ids.end());
            for (const auto dest_id : destination_ids) {
                rGraph.insertGlobalIndices(static_cast<GO>(dest_id),
                                           Teuchos::ArrayView<const GO>(col_gids.data(), col_gids.size()));
            }
        }
    }
}

void BuildMatrix(Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType>& rpMdo,
                 std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems)
{
    MatrixType local_mapping_matrix;
    EquationIdVectorType origin_ids;
    EquationIdVectorType destination_ids;

    std::vector<GO> col_gids;
    std::vector<double> row_values;

    for (auto& rp_local_sys : rMapperLocalSystems) {
        rp_local_sys->CalculateLocalSystem(local_mapping_matrix, origin_ids, destination_ids);

        KRATOS_DEBUG_ERROR_IF(local_mapping_matrix.size1() != destination_ids.size()) << "MPI-MappingMatrixAssembly: DestinationID vector size mismatch: LocalMappingMatrix-Size1: " << local_mapping_matrix.size1() << " | DestinationIDs-size: " << destination_ids.size() << std::endl;
        KRATOS_DEBUG_ERROR_IF(local_mapping_matrix.size2() != origin_ids.size())<< "MPI-MappingMatrixAssembly: OriginID vector size mismatch: LocalMappingMatrix-Size2: " << local_mapping_matrix.size2() << " | OriginIDs-size: " << origin_ids.size() << std::endl;

        if (local_mapping_matrix.size1() > 0) {
            col_gids.assign(origin_ids.begin(), origin_ids.end());
            row_values.resize(origin_ids.size());
            for (std::size_t i = 0; i < destination_ids.size(); ++i) {
                for (std::size_t j = 0; j < origin_ids.size(); ++j) {
                    row_values[j] = local_mapping_matrix(i, j);
                }
                const int ierr = rpMdo->sumIntoGlobalValues(
                    static_cast<GO>(destination_ids[i]),
                    Teuchos::ArrayView<const GO>(col_gids.data(), col_gids.size()),
                    Teuchos::ArrayView<const double>(row_values.data(), row_values.size()));

                // sumIntoGlobalValues returns the number of successfully summed entries
                KRATOS_ERROR_IF(ierr != static_cast<int>(col_gids.size())) << "Tpetra failure in FECrsMatrix.sumIntoGlobalValues. "
                    << "Summed entries: " << ierr << " (expected " << col_gids.size() << ")" << std::endl;
            }
        }

        // The local-systems are always cleared since they would be recomputed
        // to fill a new MappingMatrix
        rp_local_sys->Clear();
    }
}

} // anonymous namespace

template<>
void MappingMatrixUtilitiesType::InitializeSystemVector(
    Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpVector,
    const std::size_t VectorSize)
{
    KRATOS_ERROR << "this function was not yet implemented in Trilinos!" << std::endl;
}

template<>
void MappingMatrixUtilitiesType::BuildMappingMatrix(
    Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType>& rpMappingMatrix,
    Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpInterfaceVectorOrigin,
    Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType>& rpInterfaceVectorDestination,
    const ModelPart& rModelPartOrigin,
    const ModelPart& rModelPartDestination,
    std::vector<Kratos::unique_ptr<MapperLocalSystem>>& rMapperLocalSystems,
    const int EchoLevel)
{
    KRATOS_TRY

    static_assert(MappingSparseSpaceType::IsDistributed(), "Using a non-distributed Space!");

    // ***** Creating vectors with information abt which IDs are local *****
    const auto& r_local_mesh_origin = rModelPartOrigin.GetCommunicator().LocalMesh();
    const auto& r_local_mesh_destination = rModelPartDestination.GetCommunicator().LocalMesh();

    const int num_local_nodes_orig = r_local_mesh_origin.NumberOfNodes();
    const int num_local_nodes_dest = r_local_mesh_destination.NumberOfNodes();

    std::vector<GO> global_equation_ids_origin(num_local_nodes_orig);
    std::vector<GO> global_equation_ids_destination(num_local_nodes_dest);

    const auto nodes_begin_orig = r_local_mesh_origin.NodesBegin();
    IndexPartition<std::size_t>(num_local_nodes_orig).for_each([&global_equation_ids_origin, &nodes_begin_orig](const std::size_t Index){
        global_equation_ids_origin[Index] = (nodes_begin_orig+Index)->GetValue(INTERFACE_EQUATION_ID);
    });

    const auto nodes_begin_dest = r_local_mesh_destination.NodesBegin();
    IndexPartition<std::size_t>(num_local_nodes_dest).for_each([&global_equation_ids_destination, &nodes_begin_dest](const std::size_t Index){
        global_equation_ids_destination[Index] = (nodes_begin_dest+Index)->GetValue(INTERFACE_EQUATION_ID);
    });

    // Construct vectors containing all the equation ids of rows and columns this processor contributes to
    std::set<GO> row_equation_ids_set;
    std::set<GO> col_equation_ids_set;
    ConstructRowColIdSets(rMapperLocalSystems, row_equation_ids_set, col_equation_ids_set);

    // ***** Creating the maps for the MappingMatrix and the SystemVectors *****
    auto raw_mpi_comm = MPIDataCommunicator::GetMPICommunicator(rModelPartOrigin.GetCommunicator().GetDataCommunicator());
    typename MappingSparseSpaceType::CommunicatorPointerType tpetra_comm =
        Teuchos::rcp(new typename MappingSparseSpaceType::CommunicatorType(raw_mpi_comm));

    const auto invalid_global_size = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();

    // One-to-one maps built from the locally owned interface equation ids
    // (declared with the exact RCP<const Map> type expected by the FECrsGraph constructor,
    // as the unconstrained Teuchos::RCP converting constructor makes overloads ambiguous otherwise)
    using MapPointerType = typename MappingSparseSpaceType::MapPointerType;
    const MapPointerType p_domain_map = Teuchos::rcp(new MapType(invalid_global_size,
        Teuchos::ArrayView<const GO>(global_equation_ids_origin.data(), global_equation_ids_origin.size()),
        0, tpetra_comm));
    const MapPointerType p_range_map = Teuchos::rcp(new MapType(invalid_global_size,
        Teuchos::ArrayView<const GO>(global_equation_ids_destination.data(), global_equation_ids_destination.size()),
        0, tpetra_comm));

    // Owned-plus-shared row map: the owned rows FIRST (same order as the owned map, as required
    // by Tpetra::FECrsGraph), followed by the non-owned rows this rank contributes to
    std::vector<GO> owned_plus_shared_row_gids(global_equation_ids_destination.begin(), global_equation_ids_destination.end());
    {
        const std::set<GO> owned_rows(global_equation_ids_destination.begin(), global_equation_ids_destination.end());
        for (const GO row_id : row_equation_ids_set) {
            if (owned_rows.find(row_id) == owned_rows.end()) {
                owned_plus_shared_row_gids.push_back(row_id);
            }
        }
    }
    const MapPointerType p_owned_plus_shared_row_map = Teuchos::rcp(new MapType(invalid_global_size,
        Teuchos::ArrayView<const GO>(owned_plus_shared_row_gids.data(), owned_plus_shared_row_gids.size()),
        0, tpetra_comm));

    // Owned-plus-shared domain (column) map: owned columns first, followed by the non-owned ones
    std::vector<GO> owned_plus_shared_col_gids(global_equation_ids_origin.begin(), global_equation_ids_origin.end());
    {
        const std::set<GO> owned_cols(global_equation_ids_origin.begin(), global_equation_ids_origin.end());
        for (const GO col_id : col_equation_ids_set) {
            if (owned_cols.find(col_id) == owned_cols.end()) {
                owned_plus_shared_col_gids.push_back(col_id);
            }
        }
    }
    const MapPointerType p_owned_plus_shared_domain_map = Teuchos::rcp(new MapType(invalid_global_size,
        Teuchos::ArrayView<const GO>(owned_plus_shared_col_gids.data(), owned_plus_shared_col_gids.size()),
        0, tpetra_comm));

    // ***** Creating the graph for the MappingMatrix *****
    const std::size_t num_indices_per_row = 5; // as in the Epetra implementation

    const Teuchos::RCP<const Tpetra::Import<LO, GO, NT>> p_null_importer; // computed internally by the graph
    auto p_graph = Teuchos::rcp(new GraphType(
        p_range_map,                    // owned row map (the mapping matrix rows live on the destination)
        p_owned_plus_shared_row_map,    // owned plus shared row map
        num_indices_per_row,
        p_owned_plus_shared_domain_map, // owned plus shared domain map
        p_null_importer,                // importer (computed internally)
        p_domain_map,                   // owned domain map
        p_range_map));                  // owned range map

    p_graph->beginAssembly();
    ConstructMatrixStructure(*p_graph, rMapperLocalSystems);
    p_graph->endAssembly();

    // ***** Creating the MappingMatrix *****
    Kratos::unique_ptr<typename MappingSparseSpaceType::MatrixType> p_Mdo =
        Kratos::make_unique<typename MappingSparseSpaceType::MatrixType>(p_graph);

    p_Mdo->beginAssembly();
    BuildMatrix(p_Mdo, rMapperLocalSystems);
    p_Mdo->endAssembly();

    if (EchoLevel > 2) {
        const std::string file_name = "TrilinosMappingMatrix_O_" + rModelPartOrigin.Name() + "__D_" + rModelPartDestination.Name() +".mm";
        MappingSparseSpaceType::WriteMatrixMarketMatrix(file_name.c_str(), *p_Mdo, false);
    }

    rpMappingMatrix.swap(p_Mdo);

    // ***** Creating the SystemVectors *****
    Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType> p_new_vector_destination =
        Kratos::make_unique<typename MappingSparseSpaceType::VectorType>(p_range_map, Teuchos::null, 1);
    Kratos::unique_ptr<typename MappingSparseSpaceType::VectorType> p_new_vector_origin =
        Kratos::make_unique<typename MappingSparseSpaceType::VectorType>(p_domain_map, Teuchos::null, 1);
    rpInterfaceVectorDestination.swap(p_new_vector_destination);
    rpInterfaceVectorOrigin.swap(p_new_vector_origin);

    KRATOS_CATCH("")
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Class template instantiation
template class MappingMatrixUtilities< MappingSparseSpaceType, DenseSpaceType >;

}  // namespace Kratos.

#endif // HAVE_TPETRA
