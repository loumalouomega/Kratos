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
#include "includes/model_part.h"
#include "linear_solvers/linear_solver.h"
#include "processes/process.h"
#include "spaces/ublas_space.h"

// TrilinosApplication dependencies
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

#include "custom_utilities/trilinos_fractional_step_settings.h"
#include "custom_utilities/trilinos_fractional_step_settings_periodic.h"

namespace Kratos {
namespace Python {

namespace {

template<class TSparseSpace, class TLinearSolverType>
void RegisterTrilinosUtilities(pybind11::module& m, const std::string& Prefix)
{
    namespace py = pybind11;

    using UblasLocalSpace = UblasSpace<double, Matrix, Vector>;
    using TrilinosCommunicatorType = typename TSparseSpace::CommunicatorType;

    using BaseSolverSettings = SolverSettings<TSparseSpace, UblasLocalSpace, TLinearSolverType>;
    typedef void (BaseSolverSettings::*BuildTurbulenceModel)(typename BaseSolverSettings::TurbulenceModelLabel const&, typename TLinearSolverType::Pointer, const double, const unsigned int);
    typedef void (BaseSolverSettings::*PassTurbulenceModel)(Process::Pointer);
    BuildTurbulenceModel set_turbulence_model_by_build_overload = &BaseSolverSettings::SetTurbulenceModel;
    PassTurbulenceModel set_turbulence_model_by_pass_overload = &BaseSolverSettings::SetTurbulenceModel;

    // Note: this class is just here to provide a basis for derived classes. It has no constructor and should not be creable from python.
    // The legacy (Epetra) registration keeps the historical unprefixed name
    const std::string base_settings_name = (Prefix == "Trilinos") ? "BaseSolverSettings" : Prefix + "BaseSolverSettings";
    py::class_< BaseSolverSettings, typename BaseSolverSettings::Pointer >(m, base_settings_name.c_str() )
    .def("SetTurbulenceModel",set_turbulence_model_by_build_overload)
    .def("SetTurbulenceModel",set_turbulence_model_by_pass_overload)
    ;

    py::enum_<typename BaseSolverSettings::StrategyLabel>(m, (Prefix + "StrategyLabel").c_str())
    .value("Velocity",BaseSolverSettings::Velocity)
    .value("Pressure",BaseSolverSettings::Pressure)
    ;

    py::enum_<typename BaseSolverSettings::TurbulenceModelLabel>(m, (Prefix + "TurbulenceModelLabel").c_str())
    .value("SpalartAllmaras",BaseSolverSettings::SpalartAllmaras)
    ;

    using FractionalStepSettings = TrilinosFractionalStepSettings<TSparseSpace,UblasLocalSpace,TLinearSolverType>;
    typedef void (FractionalStepSettings::*SetStrategyByParamsType)(typename FractionalStepSettings::StrategyLabel const&,typename TLinearSolverType::Pointer,const double,const unsigned int);
    SetStrategyByParamsType ThisSetStrategyOverload = &FractionalStepSettings::SetStrategy;

    py::class_< FractionalStepSettings, typename FractionalStepSettings::Pointer, BaseSolverSettings>(m, (Prefix + "FractionalStepSettings").c_str())
    .def(py::init<TrilinosCommunicatorType&,ModelPart&,unsigned int,unsigned int,bool,bool,bool>())
    .def("SetStrategy",ThisSetStrategyOverload)
    .def("GetStrategy",&FractionalStepSettings::pGetStrategy)
    .def("SetEchoLevel",&FractionalStepSettings::SetEchoLevel)
    ;

    using FractionalStepSettingsPeriodic = TrilinosFractionalStepSettingsPeriodic<TSparseSpace,UblasLocalSpace,TLinearSolverType>;
    typedef void (FractionalStepSettingsPeriodic::*SetStrategyByParamsPeriodicType)(typename BaseSolverSettings::StrategyLabel const&,typename TLinearSolverType::Pointer,const double,const unsigned int);
    SetStrategyByParamsPeriodicType ThatSetStrategyOverload = &FractionalStepSettingsPeriodic::SetStrategy;

    py::class_< FractionalStepSettingsPeriodic, typename FractionalStepSettingsPeriodic::Pointer, BaseSolverSettings>(m, (Prefix + "FractionalStepSettingsPeriodic").c_str())
    .def(py::init<TrilinosCommunicatorType&,ModelPart&,unsigned int,unsigned int,bool,bool,bool,const Kratos::Variable<int>&>())
    .def("SetStrategy",ThatSetStrategyOverload)
    .def("GetStrategy",&FractionalStepSettingsPeriodic::pGetStrategy)
    .def("SetEchoLevel",&FractionalStepSettingsPeriodic::SetEchoLevel)
    ;
}

} // anonymous namespace

void AddTrilinosUtilitiesToPython(pybind11::module& m)
{
    using UblasLocalSpace = UblasSpace<double, Matrix, Vector>;

    using TrilinosSparseSpace = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
    using TrilinosLinearSolver = LinearSolver<TrilinosSparseSpace, UblasLocalSpace>;

    RegisterTrilinosUtilities<TrilinosSparseSpace, TrilinosLinearSolver>(m, "Trilinos");

#ifdef HAVE_TPETRA
    using TrilinosExperimentalSparseSpace = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
    using TrilinosExperimentalLinearSolver = LinearSolver<TrilinosExperimentalSparseSpace, UblasLocalSpace>;

    RegisterTrilinosUtilities<TrilinosExperimentalSparseSpace, TrilinosExperimentalLinearSolver>(m, "TrilinosExperimental");
#endif
}

}
}
