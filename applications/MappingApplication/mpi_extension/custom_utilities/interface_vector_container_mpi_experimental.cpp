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
// Tpetra (experimental Trilinos space) counterpart of interface_vector_container_mpi.cpp

#ifdef HAVE_TPETRA

// System includes

// External includes

// Project includes
#include "custom_utilities/interface_vector_container.h"
#include "mapper_mpi_define_experimental.h"
#include "custom_utilities/mapper_utilities.h"

namespace Kratos
{
namespace
{
typedef typename MPIMapperDefinitionsExperimental::SparseSpaceType ExperimentalSparseSpaceType;
typedef typename MPIMapperDefinitionsExperimental::DenseSpaceType  ExperimentalDenseSpaceType;
} // anonymous namespace

typedef InterfaceVectorContainer<ExperimentalSparseSpaceType, ExperimentalDenseSpaceType> ExperimentalVectorContainerType;

/***********************************************************************************/
/* PUBLIC Methods */
/***********************************************************************************/
template<>
void ExperimentalVectorContainerType::UpdateSystemVectorFromModelPart(const Variable<double>& rVariable,
                                                                      const Kratos::Flags& rMappingOptions)
{
    constexpr bool in_parallel = false; // accessing the Trilinos vectors is not threadsafe in the default configuration!

    auto vector_data = mpInterfaceVector->getDataNonConst(0);
    double* r_vector = vector_data.getRawPtr();

    switch (mInterfaceEntityType) {
        case InterfaceEntityType::NODES:
            MapperUtilities::UpdateSystemVectorFromModelPartNodes(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::ELEMENTS:
            MapperUtilities::UpdateSystemVectorFromModelPartElements(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::CONDITIONS:
            MapperUtilities::UpdateSystemVectorFromModelPartConditions(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::GEOMETRIES:
            MapperUtilities::UpdateSystemVectorFromModelPartGeometries(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;
    }
}

template<>
void ExperimentalVectorContainerType::UpdateModelPartFromSystemVector(const Variable<double>& rVariable,
                                                                      const Kratos::Flags& rMappingOptions)
{
    constexpr bool in_parallel = false; // accessing the Trilinos vectors is not threadsafe in the default configuration!

    const auto vector_data = mpInterfaceVector->getData(0);
    const double* r_vector = vector_data.getRawPtr();

    switch (mInterfaceEntityType) {
        case InterfaceEntityType::NODES:
            MapperUtilities::UpdateModelPartNodesFromSystemVector(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::ELEMENTS:
            MapperUtilities::UpdateModelPartElementsFromSystemVector(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::CONDITIONS:
            MapperUtilities::UpdateModelPartConditionsFromSystemVector(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;

        case InterfaceEntityType::GEOMETRIES:
            MapperUtilities::UpdateModelPartGeometriesFromSystemVector(
                r_vector, mrModelPart, rVariable, rMappingOptions, in_parallel);
            break;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Class template instantiation
template class InterfaceVectorContainer<ExperimentalSparseSpaceType, ExperimentalDenseSpaceType>;

}  // namespace Kratos

#endif // HAVE_TPETRA
