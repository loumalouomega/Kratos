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
#include "processes/process.h"
#include "spaces/ublas_space.h"

// TrilinosApplication dependencies
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

// FluidDynamics trilinos extensions
#include "custom_processes/trilinos_spalart_allmaras_turbulence_model.h"
#include "custom_processes/trilinos_stokes_initialization_process.h"

// FluidDynamicsApplication dependencies
#include "custom_processes/distance_smoothing_process.h"

namespace Kratos {
namespace Python {

namespace py = pybind11;

typedef UblasSpace<double, Matrix, Vector> UblasLocalSpaceType;

namespace {

template<class TSparseSpace, class TLinearSolverType, class TBinder, unsigned int TDim>
void DistanceSmoothingConstructionHelper(TBinder& rBinder)
{
    using TrilinosCommunicatorType = typename TSparseSpace::CommunicatorType;
    using DistanceSmoothingProcessType = DistanceSmoothingProcess<TDim, TSparseSpace, UblasLocalSpaceType, TLinearSolverType>;

    rBinder.def(py::init([](
        TrilinosCommunicatorType& rComm, ModelPart& rModelPart, typename TLinearSolverType::Pointer pLinearSolver)
        {
            constexpr int RowSizeGuess = (TDim == 2 ? 15 : 40);
            auto p_builder_solver = Kratos::make_shared<TrilinosBlockBuilderAndSolver<TSparseSpace, UblasLocalSpaceType, TLinearSolverType > >(
                rComm, RowSizeGuess, pLinearSolver);
            return Kratos::make_shared<DistanceSmoothingProcessType>(rModelPart, pLinearSolver, p_builder_solver);
        }));
}

template<class TSparseSpace, class TLinearSolverType>
void RegisterTrilinosProcesses(pybind11::module& m, const std::string& Prefix)
{
    using TrilinosCommunicatorType = typename TSparseSpace::CommunicatorType;

    // Turbulence models
    using SpalartAllmarasProcess = TrilinosSpalartAllmarasTurbulenceModel<TSparseSpace, UblasLocalSpaceType, TLinearSolverType>;
    py::class_< SpalartAllmarasProcess, typename SpalartAllmarasProcess::Pointer, Process>(m, (Prefix + "SpalartAllmarasTurbulenceModel").c_str())
    .def(py::init < TrilinosCommunicatorType&, ModelPart&, typename TLinearSolverType::Pointer, unsigned int, double, unsigned int, bool, unsigned int>())
    .def("ActivateDES", &SpalartAllmarasProcess::ActivateDES)
    .def("AdaptForFractionalStep", &SpalartAllmarasProcess::AdaptForFractionalStep)
    ;

    // Stokes initialization processes
    using StokesInitializationProcess = TrilinosStokesInitializationProcess<TSparseSpace, UblasLocalSpaceType, TLinearSolverType>;
    py::class_< StokesInitializationProcess, typename StokesInitializationProcess::Pointer, Process>(m, (Prefix + "StokesInitializationProcess").c_str())
    .def(py::init<TrilinosCommunicatorType&, ModelPart&, typename TLinearSolverType::Pointer, unsigned int, const Kratos::Variable<int>& >())
    .def("SetConditions",&StokesInitializationProcess::SetConditions)
    ;

    // Distance smoothing processes
    using DistanceSmoothing2DType = DistanceSmoothingProcess<2, TSparseSpace, UblasLocalSpaceType, TLinearSolverType>;
    using DistanceSmoothing3DType = DistanceSmoothingProcess<3, TSparseSpace, UblasLocalSpaceType, TLinearSolverType>;
    using DistanceSmoothing2DBinderType = py::class_<DistanceSmoothing2DType, typename DistanceSmoothing2DType::Pointer, Process >;
    using DistanceSmoothing3DBinderType = py::class_<DistanceSmoothing3DType, typename DistanceSmoothing3DType::Pointer, Process >;

    auto distance_smoothing_2d_binder = DistanceSmoothing2DBinderType(m, (Prefix + "DistanceSmoothingProcess2D").c_str());
    auto distance_smoothing_3d_binder = DistanceSmoothing3DBinderType(m, (Prefix + "DistanceSmoothingProcess3D").c_str());

    DistanceSmoothingConstructionHelper<TSparseSpace, TLinearSolverType, DistanceSmoothing2DBinderType, 2>(distance_smoothing_2d_binder);
    DistanceSmoothingConstructionHelper<TSparseSpace, TLinearSolverType, DistanceSmoothing3DBinderType, 3>(distance_smoothing_3d_binder);
}

} // anonymous namespace

void AddTrilinosProcessesToPython(pybind11::module& m)
{
    using TrilinosSparseSpaceType = TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>;
    using TrilinosLinearSolverType = LinearSolver<TrilinosSparseSpaceType, UblasLocalSpaceType>;

    RegisterTrilinosProcesses<TrilinosSparseSpaceType, TrilinosLinearSolverType>(m, "Trilinos");

#ifdef HAVE_TPETRA
    using TrilinosExperimentalSparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
    using TrilinosExperimentalLinearSolverType = LinearSolver<TrilinosExperimentalSparseSpaceType, UblasLocalSpaceType>;

    RegisterTrilinosProcesses<TrilinosExperimentalSparseSpaceType, TrilinosExperimentalLinearSolverType>(m, "TrilinosExperimental");
#endif
}

}
}
