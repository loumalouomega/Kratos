//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Jordi Cotela
//

#include "add_trilinos_utilities_to_python.h"

// Trilinos includes
#include "Epetra_FEVector.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_MpiComm.h"

// KratosCore dependencies
#include "containers/variable.h"
#include "processes/process.h"
#include "solving_strategies/schemes/scheme.h"
#include "spaces/ublas_space.h"

// TrilinosApplication dependencies
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

// FluidDynamicsApplication dependencies
#include "custom_strategies/schemes/bdf2_turbulent_scheme.h"
#include "custom_strategies/schemes/residualbased_predictorcorrector_velocity_bossak_scheme_turbulent.h"
#include "custom_strategies/schemes/residualbased_simple_steady_scheme.h"

namespace Kratos {
namespace Python {

namespace {

template<class TSparseSpace>
void RegisterTrilinosSchemes(pybind11::module& m, const std::string& Prefix)
{
    namespace py = pybind11;

    using UblasLocalSpace = UblasSpace<double, Matrix, Vector>;

    using BaseSchemeType = Scheme< TSparseSpace, UblasLocalSpace >;

    using BDF2TurbulentSchemeType = BDF2TurbulentScheme<TSparseSpace, UblasLocalSpace>;
    py::class_< BDF2TurbulentSchemeType, typename BDF2TurbulentSchemeType::Pointer, BaseSchemeType >( m, (Prefix + "BDF2TurbulentScheme").c_str())
    .def(py::init<>()) // constructor without a turbulence model
    .def(py::init<Process::Pointer>() ) // constructor with a turbulence model
    .def(py::init<const Variable<int>&>()) // constructor for periodic conditions
    ;

    using VelocityBossakSchemeTurbulentType = ResidualBasedPredictorCorrectorVelocityBossakSchemeTurbulent<TSparseSpace, UblasLocalSpace>;
    py::class_ < VelocityBossakSchemeTurbulentType, typename VelocityBossakSchemeTurbulentType::Pointer, BaseSchemeType >
    (m, (Prefix + "PredictorCorrectorVelocityBossakSchemeTurbulent").c_str())
    .def(py::init<double, double, unsigned int, Process::Pointer >())
    .def(py::init<double, double, unsigned int, double, Process::Pointer >())
    .def(py::init<double,double,unsigned int >())
    .def(py::init<double,unsigned int, const Variable<int>&>())
    ;

    using ResidualBasedSimpleSteadySchemeType = ResidualBasedSimpleSteadyScheme<TSparseSpace, UblasLocalSpace>;
    py::class_ < ResidualBasedSimpleSteadySchemeType, typename ResidualBasedSimpleSteadySchemeType::Pointer, BaseSchemeType >
    (m, (Prefix + "ResidualBasedSimpleSteadyScheme").c_str())
    .def(py::init<double, double, unsigned int, Process::Pointer >())
    .def(py::init<double,double,unsigned int >())
    ;
}

} // anonymous namespace

void AddTrilinosSchemesToPython(pybind11::module& m)
{
    using TrilinosSparseSpace = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;

    RegisterTrilinosSchemes<TrilinosSparseSpace>(m, "Trilinos");

#ifdef HAVE_TPETRA
    using TrilinosExperimentalSparseSpace = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;

    RegisterTrilinosSchemes<TrilinosExperimentalSparseSpace>(m, "TrilinosExperimental");
#endif
}

}
}
