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
#include "includes/model_part.h"
#include "linear_solvers/linear_solver.h"
#include "solving_strategies/strategies/implicit_solving_strategy.h"
#include "spaces/ublas_space.h"

// TrilinosApplication dependencies
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

// FluidDynamics trilinos extensions
#include "custom_strategies/strategies/fractional_step_strategy.h"
#include "custom_utilities/solver_settings.h"

// adjoint schemes
#include "custom_strategies/schemes/simple_steady_adjoint_scheme.h"
#include "custom_strategies/schemes/velocity_bossak_adjoint_scheme.h"

// sensitivity builder schemes
#include "custom_strategies/schemes/simple_steady_sensitivity_builder_scheme.h"
#include "custom_strategies/schemes/velocity_bossak_sensitivity_builder_scheme.h"

namespace Kratos {
namespace Python {

namespace {

template<class TSparseSpace, class TLinearSolverType>
void RegisterTrilinosStrategies(pybind11::module& m, const std::string& Prefix)
{
    namespace py = pybind11;

    using UblasLocalSpace = UblasSpace<double, Matrix, Vector>;

    using BaseSolvingStrategyType = ImplicitSolvingStrategy< TSparseSpace, UblasLocalSpace, TLinearSolverType >;
    using BaseSolverSettings = SolverSettings<TSparseSpace, UblasLocalSpace, TLinearSolverType>;
    using BaseSchemeType = Scheme<TSparseSpace, UblasLocalSpace>;

    using FractionalStepStrategyType = FractionalStepStrategy< TSparseSpace, UblasLocalSpace, TLinearSolverType>;
    py::class_< FractionalStepStrategyType, typename FractionalStepStrategyType::Pointer, BaseSolvingStrategyType >(m, (Prefix + "FractionalStepStrategy").c_str())
    .def(py::init< ModelPart&, BaseSolverSettings&, bool >())
    .def(py::init< ModelPart&, BaseSolverSettings&, bool, bool >())
    .def(py::init< ModelPart&, BaseSolverSettings&, bool, const Kratos::Variable<int>& >())
    .def(py::init< ModelPart&, BaseSolverSettings&, bool, bool, const Kratos::Variable<int>& >())
    .def("CalculateReactions", [Prefix](FractionalStepStrategyType& self) {
        KRATOS_WARNING(Prefix + "FractionalStepStrategy") << "\'CalculateReactions()\' exposure is deprecated. Use the constructor with the \'CalculateReactionsFlag\' instead." << std::endl;
        self.CalculateReactions();})
    .def("AddIterationStep",&FractionalStepStrategyType::AddIterationStep)
    .def("ClearExtraIterationSteps",&FractionalStepStrategyType::ClearExtraIterationSteps)
    ;

    using SimpleSteadyAdjointSchemeType = SimpleSteadyAdjointScheme<TSparseSpace, UblasLocalSpace>;
    py::class_<SimpleSteadyAdjointSchemeType, typename SimpleSteadyAdjointSchemeType::Pointer, BaseSchemeType>
        (m, (Prefix + "SimpleSteadyAdjointScheme").c_str())
        .def(py::init<AdjointResponseFunction::Pointer, const std::size_t, const std::size_t>())
        ;

    using VelocityBossakAdjointSchemeType = VelocityBossakAdjointScheme<TSparseSpace, UblasLocalSpace>;
    py::class_<VelocityBossakAdjointSchemeType, typename VelocityBossakAdjointSchemeType::Pointer, BaseSchemeType>
        (m, (Prefix + "VelocityBossakAdjointScheme").c_str())
        .def(py::init<Parameters, AdjointResponseFunction::Pointer, const std::size_t, const std::size_t>())
        ;
}

} // anonymous namespace

void AddTrilinosStrategiesToPython(pybind11::module& m)
{
    using UblasLocalSpace = UblasSpace<double, Matrix, Vector>;

    using TrilinosSparseSpace = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
    using TrilinosLinearSolver = LinearSolver<TrilinosSparseSpace, UblasLocalSpace>;

    RegisterTrilinosStrategies<TrilinosSparseSpace, TrilinosLinearSolver>(m, "Trilinos");

#ifdef HAVE_TPETRA
    using TrilinosExperimentalSparseSpace = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
    using TrilinosExperimentalLinearSolver = LinearSolver<TrilinosExperimentalSparseSpace, UblasLocalSpace>;

    RegisterTrilinosStrategies<TrilinosExperimentalSparseSpace, TrilinosExperimentalLinearSolver>(m, "TrilinosExperimental");
#endif
}

}
}
