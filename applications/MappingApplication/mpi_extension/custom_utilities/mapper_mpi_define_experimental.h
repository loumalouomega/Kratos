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

// External includes

// Project includes
// NOTE: Kept separate from mapper_mpi_define.h on purpose: the Epetra and Tpetra spaces are
// never mixed in the same translation unit of this extension (mirrors the TrilinosApplication
// convention of separate *_experimental translation units).
#include "trilinos_space_experimental.h"
#include "spaces/ublas_space.h"

namespace Kratos {

namespace MPIMapperDefinitionsExperimental {

    typedef TUblasDenseSpace<double> DenseSpaceType;

    typedef TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>> SparseSpaceType;

}  // namespace MPIMapperDefinitionsExperimental.

}  // namespace Kratos.
