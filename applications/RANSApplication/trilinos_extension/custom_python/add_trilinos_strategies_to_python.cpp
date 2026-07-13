//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Suneth Warnakulasuriya
//

// Trilinos includes
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "Epetra_MpiComm.h"

// KratosCore dependencies
#include "includes/model_part.h"
#include "spaces/ublas_space.h"

// TrilinosApplication dependencies
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

// RANS trilinos extensions
// schemes
#include "custom_strategies/algebraic_flux_corrected_steady_scalar_scheme.h"
#include "custom_strategies/bossak_relaxation_scalar_scheme.h"
#include "custom_strategies/steady_scalar_scheme.h"

// Include base h
#include "add_trilinos_strategies_to_python.h"

namespace Kratos
{
namespace Python
{

namespace
{

template<class TSparseSpace>
void RegisterTrilinosStrategies(pybind11::module& m, const std::string& Prefix)
{
    namespace py = pybind11;

    using LocalSpaceType = UblasSpace<double, Matrix, Vector>;
    using MPIBaseSchemeType = Scheme<TSparseSpace, LocalSpaceType>;

    // add schemes
    using MPIAlgebraicFluxCorrectedSteadyScalarSchemeType = AlgebraicFluxCorrectedSteadyScalarScheme<TSparseSpace, LocalSpaceType>;
    py::class_<MPIAlgebraicFluxCorrectedSteadyScalarSchemeType, typename MPIAlgebraicFluxCorrectedSteadyScalarSchemeType::Pointer, MPIBaseSchemeType>(m, (Prefix + "AlgebraicFluxCorrectedSteadyScalarScheme").c_str())
        .def(py::init<const double, const Flags&>())
        .def(py::init<const double, const Flags&, const Variable<int>&>());

    using MPISteadyScalarSchemeType = SteadyScalarScheme<TSparseSpace, LocalSpaceType>;
    py::class_<MPISteadyScalarSchemeType, typename MPISteadyScalarSchemeType::Pointer, MPIBaseSchemeType>(m, (Prefix + "SteadyScalarScheme").c_str())
        .def(py::init<const double>());

    using MPIBossakRelaxationScalarSchemeType = BossakRelaxationScalarScheme<TSparseSpace, LocalSpaceType>;
    py::class_<MPIBossakRelaxationScalarSchemeType, typename MPIBossakRelaxationScalarSchemeType::Pointer, MPIBaseSchemeType>(m, (Prefix + "BossakRelaxationScalarScheme").c_str())
        .def(py::init<const double, const double, const Variable<double>&>());
}

} // anonymous namespace

void AddTrilinosStrategiesToPython(pybind11::module& m)
{
    using MPISparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;

    RegisterTrilinosStrategies<MPISparseSpaceType>(m, "MPI");

#ifdef HAVE_TPETRA
    using MPIExperimentalSparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;

    RegisterTrilinosStrategies<MPIExperimentalSparseSpaceType>(m, "MPIExperimental");
#endif
}

} // namespace Python
} // namespace Kratos
