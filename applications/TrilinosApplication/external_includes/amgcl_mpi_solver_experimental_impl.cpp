//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// Tpetra (experimental space) counterpart of "amgcl_mpi_solver_impl.cpp".
// See that file for the rationale of the impl-header split.

#ifdef HAVE_TPETRA

// External includes
#include "amgcl_tpetra_adapter.hpp"

// Project includes
#include "trilinos_space_experimental.h"
#include "amgcl_mpi_solver.h"
#include "custom_utilities/trilinos_solver_utilities.h"

#define KRATOS_AMGCL_MPI // <= avoid including mpi.h in KratosCore
#include "linear_solvers/amgcl_solver_impl.hpp"
#undef KRATOS_AMGCL_MPI

// System includes
#include <optional>



namespace Kratos {


template <>
struct AMGCLAdaptor<TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>>
{
    using SparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
    using MatrixType = typename SparseSpaceType::MatrixType;
    using VectorType = typename SparseSpaceType::VectorType;
    using AdaptorType = amgcl::adapter::tpetra_map_adapter<MatrixType>;

    template <int BlockSize>
    auto MakeMatrixAdaptor(const MatrixType& rMatrix)
    {
        mAdaptor.emplace(amgcl::adapter::map(rMatrix));
        if constexpr (BlockSize == 1) {
            return mAdaptor.value();
        } else {
            using BlockType = amgcl::static_matrix<
                double,
                BlockSize,
                BlockSize
            >;
            return amgcl::adapter::block_matrix<BlockType>(mAdaptor.value());
        }
    }

    template <class TStaticMatrix>
    std::size_t BlockSystemSize(const MatrixType& rMatrix) const noexcept
    {
        return amgcl::adapter::tpetra_detail::GetNumLocalRows(rMatrix) / AMGCLStaticVectorTraits<TStaticMatrix>::value;
    }

    auto MakeVectorIterator(const VectorType& rVector) const
    {
        // Raw host pointer, mirroring what Epetra_FEVector::Values() provides
        return const_cast<VectorType&>(rVector).getDataNonConst(0).getRawPtr();
    }

    auto MakeVectorIterator(VectorType& rVector) const
    {
        return rVector.getDataNonConst(0).getRawPtr();
    }

    MPI_Comm GetCommunicator(MatrixType& rMatrix) const noexcept
    {
        return TrilinosSolverUtilities::GetMPICommFromTeuchosComm(*rMatrix.getRowMap()->getComm());
    }

private:
    // amgcl::adapter::block_matrix constructs a class
    // that stores a reference to the "matrix" passed
    // into it, which in this case means the adaptor
    // defined below. We need to keep it alive until
    // the hierarchy construction finishes, hence the
    // convoluted member variable.
    // Optional is used here to represent an invalid state
    // of the matrix view, before InitializeSolutionStep is
    // called.
    std::optional<AdaptorType> mAdaptor;
};


template class KRATOS_API(TRILINOS_APPLICATION) AmgclMPISolver<
    TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>,
    UblasSpace<double, Matrix, Vector>
>;


} // namespace Kratos

#endif // HAVE_TPETRA
