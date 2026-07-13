//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:        BSD License
//                  Kratos default license: kratos/license.txt
//
//  Main authors:   Vicente Mataix Ferrandiz
//                  Philipp Bucher
//

// Project includes

// Linear solvers
#include "trilinos_linear_solver_factory.h"
#include "linear_solvers/fallback_linear_solver.h"

#ifndef TRILINOS_EXCLUDE_AZTEC_SOLVER
#include "external_includes/aztec_solver.h"
#endif

#ifndef TRILINOS_EXCLUDE_AMESOS_SOLVER
#include "external_includes/amesos_solver.h"
#endif

#ifndef TRILINOS_EXCLUDE_AMESOS2_SOLVER
#include "external_includes/amesos2_solver.h"
#endif

#ifndef TRILINOS_EXCLUDE_ML_SOLVER
#include "external_includes/ml_solver.h"
#endif

#include "external_includes/amgcl_mpi_solver.h"
#include "external_includes/amgcl_mpi_schur_complement_solver.h"
#include "external_includes/trilinos_monotonicity_preserving_solver.h"

#if defined(HAVE_TPETRA) && defined(HAVE_BELOS)
#include "external_includes/belos_solver.h"
#endif

#if defined(HAVE_TPETRA) && defined(HAVE_BELOS) && defined(HAVE_MUELU)
#include "external_includes/muelu_solver.h"
#endif

namespace Kratos {

namespace {
    template<class TSparseSpace>
    void RegisterSolvers()
    {
        using TrilinosLocalSpaceType = UblasSpace<double, Matrix, Vector>;

        using TrilinosFallbackLinearSolverType = FallbackLinearSolver<TSparseSpace, TrilinosLocalSpaceType>;
        const static auto TrilinosFallbackLinearSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, TrilinosFallbackLinearSolverType>();
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("fallback_linear_solver", TrilinosFallbackLinearSolverFactory);
        } else if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#ifdef HAVE_TPETRA
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("fallback_linear_solver", TrilinosFallbackLinearSolverFactory);
#endif
        } else {
            KRATOS_ERROR << "Only EPETRA and TPETRA are supported for now" << std::endl;
        }

#ifndef TRILINOS_EXCLUDE_AZTEC_SOLVER
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            using AztecSolverType = AztecSolver<TSparseSpace, TrilinosLocalSpaceType >;
            static auto AztecSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, AztecSolverType>();
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("aztec",    AztecSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("cg",       AztecSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("bicgstab", AztecSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("gmres",    AztecSolverFactory);
        }
#endif

#ifndef TRILINOS_EXCLUDE_AMESOS_SOLVER
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            using AmesosSolverType = AmesosSolver<TSparseSpace, TrilinosLocalSpaceType >;
            static auto AmesosSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, AmesosSolverType>();
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("amesos",        AmesosSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("klu",           AmesosSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("super_lu_dist", AmesosSolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("mumps",         AmesosSolverFactory);
        }
#endif

#ifndef TRILINOS_EXCLUDE_AMESOS2_SOLVER
        using Amesos2SolverType = Amesos2Solver<TSparseSpace, TrilinosLocalSpaceType >;
        static auto Amesos2SolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, Amesos2SolverType>();
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("amesos2",        Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("klu2",           Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("mumps2",         Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("super_lu_dist2", Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("basker",         Amesos2SolverFactory);
        } else if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#ifdef HAVE_TPETRA
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("amesos2",        Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("klu2",           Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("mumps2",         Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("super_lu_dist2", Amesos2SolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("basker",         Amesos2SolverFactory);
#endif
        } else {
            KRATOS_ERROR << "Only EPETRA and TPETRA are supported for now" << std::endl;
        }
#endif

#ifndef TRILINOS_EXCLUDE_ML_SOLVER
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            using MLSolverType = MultiLevelSolver<TSparseSpace, TrilinosLocalSpaceType >;
            static auto MultiLevelSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, MLSolverType>();
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("multi_level", MultiLevelSolverFactory);
        }
#endif

        // Belos iterative solvers (Tpetra-era replacement for AztecOO): registered under
        // the same names the Aztec solver uses for Epetra so existing settings work unchanged
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#if defined(HAVE_TPETRA) && defined(HAVE_BELOS)
            using BelosSolverType = BelosSolver<TSparseSpace, TrilinosLocalSpaceType >;
            static auto BelosSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, BelosSolverType>();
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("belos",    BelosSolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("cg",       BelosSolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("bicgstab", BelosSolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("gmres",    BelosSolverFactory);
#endif
        }

        // MueLu algebraic multigrid (Tpetra-era replacement for ML): registered under the
        // same name the ML solver uses for Epetra so existing settings work unchanged
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#if defined(HAVE_TPETRA) && defined(HAVE_BELOS) && defined(HAVE_MUELU)
            using MueLuSolverType = MueLuSolver<TSparseSpace, TrilinosLocalSpaceType >;
            static auto MueLuSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, MueLuSolverType>();
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("muelu",       MueLuSolverFactory);
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("multi_level", MueLuSolverFactory);
#endif
        }

        using AmgclMPISolverType = AmgclMPISolver<TSparseSpace, TrilinosLocalSpaceType >;
        static auto AmgclMPISolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, AmgclMPISolverType>();
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("amgcl", AmgclMPISolverFactory);
        } else if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#ifdef HAVE_TPETRA
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("amgcl", AmgclMPISolverFactory);
#endif
        } else {
            KRATOS_ERROR << "Only EPETRA and TPETRA are supported for now" << std::endl;
        }

        using AmgclMPISchurComplementSolverType = AmgclMPISchurComplementSolver<TSparseSpace, TrilinosLocalSpaceType >;
        static auto AmgclMPISchurComplementSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, AmgclMPISchurComplementSolverType>();
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("amgcl_schur_complement", AmgclMPISchurComplementSolverFactory);
        } else if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#ifdef HAVE_TPETRA
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("amgcl_schur_complement", AmgclMPISchurComplementSolverFactory);
#endif
        } else {
            KRATOS_ERROR << "Only EPETRA and TPETRA are supported for now" << std::endl;
        }

        using TrilinosMonotonicityPreservingSolverType = TrilinosMonotonicityPreservingSolver<TSparseSpace, TrilinosLocalSpaceType >;
        static auto TrilinosMonotonicityPreservingSolverFactory = TrilinosLinearSolverFactory<TSparseSpace, TrilinosLocalSpaceType, TrilinosMonotonicityPreservingSolverType>();
        if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::EPETRA) {
            KRATOS_REGISTER_TRILINOS_LINEAR_SOLVER("monotonicity_preserving", TrilinosMonotonicityPreservingSolverFactory);
        } else if constexpr (TSparseSpace::LinearAlgebraLibrary() == TrilinosLinearAlgebraLibrary::TPETRA) {
#ifdef HAVE_TPETRA
            KRATOS_REGISTER_TRILINOS_EXPERIMENTAL_LINEAR_SOLVER("monotonicity_preserving", TrilinosMonotonicityPreservingSolverFactory);
#endif
        } else {
            KRATOS_ERROR << "Only EPETRA and TPETRA are supported for now" << std::endl;
        }
    }
}

void RegisterTrilinosLinearSolvers()
{
    RegisterSolvers<TrilinosSparseSpaceType>();
#ifdef HAVE_TPETRA
    RegisterSolvers<TrilinosExperimentalSparseSpaceType>();
#endif
}

template class KratosComponents<TrilinosLinearSolverFactoryType>;
#ifdef HAVE_TPETRA
template class KratosComponents<TrilinosExperimentalLinearSolverFactoryType>;
#endif
}
