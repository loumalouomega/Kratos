//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Andreas Winterstein (a.winterstein@tum.de)
//

// System includes

#if defined(KRATOS_PYTHON)
// External includes

// Project includes
#include "includes/define_python.h"
#include "add_trilinos_strategies_to_python.h"

//Trilinos includes
#include "Epetra_FEVector.h"

// Project includes
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif
#include "spaces/ublas_space.h"

#include "solving_strategies/strategies/solving_strategy.h"
#include "linear_solvers/linear_solver.h"

// Strategies
#include "custom_strategies/trilinos_laplacian_meshmoving_strategy.h"
#include "custom_strategies/trilinos_structural_meshmoving_strategy.h"


namespace Kratos {
namespace Python {

namespace {

template<class TSparseSpace>
void RegisterMeshMovingStrategies(pybind11::module& m, const std::string& Prefix)
{
    namespace py = pybind11;
    typedef UblasSpace<double, Matrix, Vector> TrilinosLocalSpaceType;
    typedef LinearSolver<TSparseSpace, TrilinosLocalSpaceType > TrilinosLinearSolverType;
    typedef typename TSparseSpace::CommunicatorType TrilinosCommunicatorType;

    typedef ImplicitSolvingStrategy< TSparseSpace, TrilinosLocalSpaceType, TrilinosLinearSolverType > TrilinosImplicitSolvingStrategyType;

    using TrilinosLaplacianMeshMovingStrategyType = TrilinosLaplacianMeshMovingStrategy< TSparseSpace, TrilinosLocalSpaceType, TrilinosLinearSolverType>;
    py::class_<TrilinosLaplacianMeshMovingStrategyType, typename TrilinosLaplacianMeshMovingStrategyType::Pointer, TrilinosImplicitSolvingStrategyType>
    (m, (Prefix + "LaplacianMeshMovingStrategy").c_str()).def(py::init<TrilinosCommunicatorType&, ModelPart&, typename TrilinosLinearSolverType::Pointer, int, bool, bool, bool, int>());

    using TrilinosStructuralMeshMovingStrategyType = TrilinosStructuralMeshMovingStrategy< TSparseSpace, TrilinosLocalSpaceType, TrilinosLinearSolverType>;
    py::class_<TrilinosStructuralMeshMovingStrategyType, typename TrilinosStructuralMeshMovingStrategyType::Pointer, TrilinosImplicitSolvingStrategyType>
    (m, (Prefix + "StructuralMeshMovingStrategy").c_str()).def(py::init<TrilinosCommunicatorType&, ModelPart&, typename TrilinosLinearSolverType::Pointer, int, bool, bool, bool, int>());
}

} // anonymous namespace

void AddMeshMovingStrategies(pybind11::module& m)
{
    typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;

    RegisterMeshMovingStrategies<TrilinosSparseSpaceType>(m, "Trilinos");

#ifdef HAVE_TPETRA
    typedef TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>> TrilinosExperimentalSparseSpaceType;

    RegisterMeshMovingStrategies<TrilinosExperimentalSparseSpaceType>(m, "TrilinosExperimental");
#endif
}

} // namespace Python.
} // namespace Kratos.

#endif // KRATOS_PYTHON defined
