//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Philipp Bucher (https://github.com/philbucher)
//

#ifdef KRATOS_PYTHON
// System includes

// External includes

// Project includes
#include "includes/define_python.h"
#include "custom_mappers/nearest_neighbor_mapper.h"
#include "custom_mappers/nearest_element_mapper.h"
#include "custom_mappers/barycentric_mapper.h"
#include "custom_mappers/projection_3D_2D_mapper.h"
#include "custom_utilities/mapper_mpi_define.h"
#include "custom_utilities/mapper_mpi_backend.h"
#include "factories/mapper_factory.h"
#include "python/add_mapper_to_python.h"

namespace Kratos::Python {

PYBIND11_MODULE(KratosMappingMPIExtension,m)
{
    AddMappingToPython<MPIMapperDefinitions::SparseSpaceType, MPIMapperDefinitions::DenseSpaceType>(m);

    // Macros for registering mappers
    // will be removed once using the core factories
    #define KRATOS_REGISTER_MAPPER(MapperType, MapperName)                                                   \
        {                                                                                                    \
        Model current_model;                                                                                 \
        ModelPart& dummy_model_part = current_model.CreateModelPart("dummy");                                \
        MapperFactory<MPIMapperDefinitions::SparseSpaceType, MPIMapperDefinitions::DenseSpaceType>::Register \
            (MapperName, Kratos::make_shared<MapperType<                                                     \
            MPIMapperDefinitions::SparseSpaceType,MPIMapperDefinitions::DenseSpaceType, MapperMPIBackend<    \
                MPIMapperDefinitions::SparseSpaceType,MPIMapperDefinitions::DenseSpaceType>>>                \
            (dummy_model_part, dummy_model_part));                                                           \
        }

    KRATOS_REGISTER_MAPPER(NearestNeighborMapper, "nearest_neighbor");
    KRATOS_REGISTER_MAPPER(NearestElementMapper,  "nearest_element");
    KRATOS_REGISTER_MAPPER(BarycentricMapper,     "barycentric");
    KRATOS_REGISTER_MAPPER(Projection3D2DMapper,  "projection_3D_2D");
}

}

#endif // KRATOS_PYTHON defined
