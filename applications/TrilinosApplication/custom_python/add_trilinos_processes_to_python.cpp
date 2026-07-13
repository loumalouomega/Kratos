//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//

#if defined(KRATOS_PYTHON)

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "includes/model_part.h"
#include "linear_solvers/linear_solver.h"
#include "processes/process.h"
#include "processes/variational_distance_calculation_process.h"
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif
#include "spaces/ublas_space.h"
#include "custom_python/add_trilinos_processes_to_python.h"
#include "custom_processes/trilinos_levelset_convection_process.h"
#include "custom_strategies/builder_and_solvers/trilinos_block_builder_and_solver.h"

namespace Kratos::Python
{
namespace py = pybind11;

namespace {

template<class TSparseSpace, class TLocalSpace, class TLinearSolverType, class TBinder, unsigned int TDim>
void DistanceCalculatorConstructionHelper(TBinder& rBinder)
{
    using TrilinosCommunicatorType = typename TSparseSpace::CommunicatorType;
    using TrilinosBlockBuilderAndSolverType = TrilinosBlockBuilderAndSolver<TSparseSpace, TLocalSpace, TLinearSolverType>;
    using VariationalDistanceCalculationType = VariationalDistanceCalculationProcess<TDim, TSparseSpace, TLocalSpace, TLinearSolverType>;

    rBinder.def(py::init([](
        TrilinosCommunicatorType& rCommunicator, Model& rModel, typename TLinearSolverType::Pointer pLinearSolver, Parameters ThisParameters)
        {
            constexpr int row_size_guess = TDim == 2 ? 15 : 40;
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolverType>(
                rCommunicator, row_size_guess, pLinearSolver);
            return Kratos::make_shared<VariationalDistanceCalculationType>(rModel, pLinearSolver, p_builder_solver, ThisParameters);
        }));
    rBinder.def(py::init([](
        TrilinosCommunicatorType& rComm, ModelPart& rModelPart, typename TLinearSolverType::Pointer pLinearSolver,
        unsigned int MaxIter, Flags TheFlags)
        {
            constexpr int RowSizeGuess = (TDim == 2 ? 15 : 40);
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolverType>(
                rComm, RowSizeGuess, pLinearSolver);
            return Kratos::make_shared<VariationalDistanceCalculationType>(rModelPart, pLinearSolver, p_builder_solver, MaxIter, TheFlags);
        }));
    rBinder.def(py::init([](
        TrilinosCommunicatorType& rComm, ModelPart& rModelPart, typename TLinearSolverType::Pointer pLinearSolver,
        unsigned int MaxIter, Flags TheFlags, std::string& rAuxName)
        {
            constexpr int RowSizeGuess = (TDim == 2 ? 15 : 40);
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolverType>(
                rComm, RowSizeGuess, pLinearSolver);
            return Kratos::make_shared<VariationalDistanceCalculationType>(rModelPart, pLinearSolver, p_builder_solver, MaxIter, TheFlags, rAuxName);
        }));
    rBinder.def(py::init([](
        TrilinosCommunicatorType& rComm, ModelPart& rModelPart, typename TLinearSolverType::Pointer pLinearSolver,
        unsigned int MaxIter, Flags TheFlags, std::string& rAuxName, double Coefficient1)
        {
            constexpr int RowSizeGuess = (TDim == 2 ? 15 : 40);
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolverType>(
                rComm, RowSizeGuess, pLinearSolver);
            return Kratos::make_shared<VariationalDistanceCalculationType>(rModelPart, pLinearSolver, p_builder_solver, MaxIter, TheFlags, rAuxName, Coefficient1);
        }));
    rBinder.def(py::init([](
        TrilinosCommunicatorType& rComm, ModelPart& rModelPart, typename TLinearSolverType::Pointer pLinearSolver,
        unsigned int MaxIter, Flags TheFlags, std::string& rAuxName, double Coefficient1, double Coefficient2)
        {
            constexpr int RowSizeGuess = (TDim == 2 ? 15 : 40);
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolverType>(
                rComm, RowSizeGuess, pLinearSolver);
            return Kratos::make_shared<VariationalDistanceCalculationType>(rModelPart, pLinearSolver, p_builder_solver, MaxIter, TheFlags, rAuxName, Coefficient1, Coefficient2);
        }));
}

template<class TSparseSpace, class TLocalSpace, class TLinearSolverType>
void RegisterProcesses(pybind11::module& m, const std::string& Prefix)
{
    using TrilinosCommunicatorType = typename TSparseSpace::CommunicatorType;

    // Variational distance calculation processes
    using DistanceCalculator2DType = VariationalDistanceCalculationProcess<2, TSparseSpace, TLocalSpace, TLinearSolverType>;
    using DistanceCalculator3DType = VariationalDistanceCalculationProcess<3, TSparseSpace, TLocalSpace, TLinearSolverType>;
    using DistanceCalculator2DBinderType = py::class_<DistanceCalculator2DType, typename DistanceCalculator2DType::Pointer, Process>;
    using DistanceCalculator3DBinderType = py::class_<DistanceCalculator3DType, typename DistanceCalculator3DType::Pointer, Process>;

    auto distance_calculator_2d_binder = DistanceCalculator2DBinderType(m, (Prefix + "VariationalDistanceCalculationProcess2D").c_str());
    auto distance_calculator_3d_binder = DistanceCalculator3DBinderType(m, (Prefix + "VariationalDistanceCalculationProcess3D").c_str());

    DistanceCalculatorConstructionHelper<TSparseSpace, TLocalSpace, TLinearSolverType, DistanceCalculator2DBinderType, 2>(distance_calculator_2d_binder);
    DistanceCalculatorConstructionHelper<TSparseSpace, TLocalSpace, TLinearSolverType, DistanceCalculator3DBinderType, 3>(distance_calculator_3d_binder);

    // Level set convection processes
    using BaseLevelSetConvectionProcess2D = LevelSetConvectionProcess<2, TSparseSpace, TLocalSpace, TLinearSolverType>;
    using BaseLevelSetConvectionProcess3D = LevelSetConvectionProcess<3, TSparseSpace, TLocalSpace, TLinearSolverType>;

    py::class_<BaseLevelSetConvectionProcess2D, typename BaseLevelSetConvectionProcess2D::Pointer, Process>(m, ("Base" + Prefix + "LevelSetConvectionProcess2D").c_str());
    py::class_<BaseLevelSetConvectionProcess3D, typename BaseLevelSetConvectionProcess3D::Pointer, Process>(m, ("Base" + Prefix + "LevelSetConvectionProcess3D").c_str());

    using LevelSetConvectionProcess2D = TrilinosLevelSetConvectionProcess<2, TSparseSpace, TLocalSpace, TLinearSolverType>;
    py::class_<LevelSetConvectionProcess2D, typename LevelSetConvectionProcess2D::Pointer, BaseLevelSetConvectionProcess2D>(m, (Prefix + "LevelSetConvectionProcess2D").c_str())
        .def(py::init<TrilinosCommunicatorType&, Model&, typename TLinearSolverType::Pointer, Parameters>())
        .def(py::init<TrilinosCommunicatorType&, ModelPart&, typename TLinearSolverType::Pointer, Parameters>())
        ;

    using LevelSetConvectionProcess3D = TrilinosLevelSetConvectionProcess<3, TSparseSpace, TLocalSpace, TLinearSolverType>;
    py::class_<LevelSetConvectionProcess3D, typename LevelSetConvectionProcess3D::Pointer, BaseLevelSetConvectionProcess3D>(m, (Prefix + "LevelSetConvectionProcess3D").c_str())
        .def(py::init<TrilinosCommunicatorType&, Model&, typename TLinearSolverType::Pointer, Parameters>())
        .def(py::init<TrilinosCommunicatorType&, ModelPart&, typename TLinearSolverType::Pointer, Parameters>())
        ;
}

} // anonymous namespace

void AddProcesses(pybind11::module& m)
{
    using TrilinosSparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
    using TrilinosLocalSpaceType = UblasSpace<double, Matrix, Vector>;
    using TrilinosLinearSolverType = LinearSolver<TrilinosSparseSpaceType, TrilinosLocalSpaceType>;

    RegisterProcesses<TrilinosSparseSpaceType, TrilinosLocalSpaceType, TrilinosLinearSolverType>(m, "Trilinos");

#ifdef HAVE_TPETRA
    using TrilinosExperimentalSparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
    using TrilinosExperimentalLinearSolverType = LinearSolver<TrilinosExperimentalSparseSpaceType, TrilinosLocalSpaceType>;

    RegisterProcesses<TrilinosExperimentalSparseSpaceType, TrilinosLocalSpaceType, TrilinosExperimentalLinearSolverType>(m, "TrilinosExperimental");
#endif
}

}

#endif
