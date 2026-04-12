//  KRATOS  _____     _ _ _
//         |_   _| __(_) (_)_ __   ___  ___
//           | || '__| | | | '_ \ / _ \/ __|
//           | || |  | | | | | | | (_) \__
//           |_||_|  |_|_|_|_| |_|\___/|___/ APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Denis Demidov
//                   Riccardo Rossi
//

// The implementation of AMGCLSolver is split between
// - an implementation header ("linear_solvers/amgcl_solver_impl.hpp")
// - implementation sources (this source file and "linear_solvers/amgcl_solver_impl.cpp")
//
// The reason is twofold:
// - includes from the AMGCL library are extremely heavy, so they are
//   avoided in the class declaration ("linear_solvers/amgcl_solver.h").
//   Instead, the implementation header includes them and defines logic
//   common to any matrix/vector representations. Each source file that
//   defines an instantiation of AMGCLSolver includes the implementation
//   header.
// - Shared memory and distributed memory matrix/vector representations
//   are handled in separate source files to avoid adding a Trilinos
//   dependency to core.

// External includes
#include "amgcl/adapter/epetra.hpp"

// Project includes
#include "trilinos_space.h"
#include "trilinos_space_experimental.h"
#include "amgcl_mpi_solver.h"
#include "custom_utilities/trilinos_solver_utilities.h"

#define KRATOS_AMGCL_MPI // <= avoid including mpi.h in KratosCore
#include "linear_solvers/amgcl_solver_impl.hpp"
#undef KRATOS_AMGCL_MPI

// System includes
#include <optional>

#ifdef HAVE_TPETRA
#include <Tpetra_FECrsMatrix.hpp>
#include <Tpetra_FEVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Import.hpp>
#endif

namespace Kratos {

#ifdef HAVE_TPETRA
namespace {
/// Adapts Tpetra::FECrsMatrix (or CrsMatrix)
template <class TpetraMatrixType>
class tpetra_map {
    public:
        typedef typename TpetraMatrixType::scalar_type value_type;
        typedef typename TpetraMatrixType::global_ordinal_type GO;
        typedef typename TpetraMatrixType::local_ordinal_type LO;

        tpetra_map(const TpetraMatrixType &A)
            : A(A), order(A.getColMap())
        {
            auto p_row_map = A.getRowMap();
            auto p_col_map = A.getColMap();

            GO local_entries = p_row_map->getLocalNumElements();
            GO entries_before;
            Teuchos::scan(*p_row_map->getComm(), Teuchos::REDUCE_SUM, local_entries, Teuchos::outArg(entries_before));
            entries_before -= local_entries;

            using TpetraVectorType = Tpetra::Vector<GO, LO, GO, typename TpetraMatrixType::node_type>;
            TpetraVectorType perm(p_row_map);
            auto perm_view = perm.getLocalViewHost(Tpetra::Access::ReadWrite);
            for(LO i = 0; i < static_cast<LO>(local_entries); ++i)
                perm_view(i, 0) = entries_before + i;

            using ImportType = Tpetra::Import<LO, GO, typename TpetraMatrixType::node_type>;
            ImportType importer(p_row_map, p_col_map);

            order.doImport(perm, importer, Tpetra::INSERT);
        }

        size_t rows() const {
            return A.getLocalNumRows();
        }

        size_t cols() const {
            return A.getGlobalNumCols();
        }

        size_t nonzeros() const {
            return A.getLocalNumEntries();
        }

        class row_iterator {
            public:
                typedef GO    col_type;
                typedef typename TpetraMatrixType::scalar_type val_type;

                row_iterator(
                        const TpetraMatrixType &A,
                        const Tpetra::Vector<GO, LO, GO, typename TpetraMatrixType::node_type> &order,
                        int row
                        )
                {
                    typename TpetraMatrixType::local_inds_host_view_type local_cols;
                    typename TpetraMatrixType::values_host_view_type vals;
                    A.getLocalRowView(row, local_cols, vals);
                    LO nnz = local_cols.extent(0);

                    col_copy.resize(nnz);
                    val_copy.resize(nnz);

                    auto order_view = order.getLocalViewHost(Tpetra::Access::ReadOnly);
                    for(LO i = 0; i < nnz; ++i) {
                        col_copy[i] = order_view(local_cols[i], 0);
                        val_copy[i] = vals[i];
                    }

                    m_col = col_copy.data();
                    m_val = val_copy.data();
                    m_end = m_col + nnz;

                    amgcl::detail::sort_row(m_col, m_val, nnz);
                }

                operator bool() const {
                    return m_col != m_end;
                }

                row_iterator& operator++() {
                    ++m_col;
                    ++m_val;
                    return *this;
                }

                col_type col() const {
                    return *m_col;
                }

                val_type value() const {
                    return *m_val;
                }

            private:
                col_type * m_col;
                col_type * m_end;
                val_type * m_val;

                std::vector<col_type> col_copy;
                std::vector<val_type> val_copy;
        };

        row_iterator row_begin(int row) const {
            return row_iterator(A, order, row);
        }
    private:
        const TpetraMatrixType &A;
        Tpetra::Vector<GO, LO, GO, typename TpetraMatrixType::node_type> order;
};
} // namespace
#endif

template <>
struct AMGCLAdaptor<TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>>
{
    template <int BlockSize>
    auto MakeMatrixAdaptor(const Epetra_FECrsMatrix& rMatrix)
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
    std::size_t BlockSystemSize(const Epetra_FECrsMatrix& rMatrix) const noexcept
    {
        return rMatrix.RowMap().NumMyElements() / AMGCLStaticVectorTraits<TStaticMatrix>::value;
    }

    auto MakeVectorIterator(const Epetra_FEVector& rVector) const
    {
        return rVector.Values();
    }

    auto MakeVectorIterator(Epetra_FEVector& rVector) const
    {
        return rVector.Values();
    }

    MPI_Comm GetCommunicator(Epetra_FECrsMatrix& rMatrix) const noexcept
    {
        return TrilinosSolverUtilities::GetMPICommFromEpetraComm(rMatrix.Comm());
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
    std::optional<amgcl::adapter::epetra_map> mAdaptor;
};


#ifdef HAVE_TPETRA
template <>
struct AMGCLAdaptor<TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEVector<>>>
{
    using TpetraMatrixType = Tpetra::FECrsMatrix<>;
    using TpetraVectorType = Tpetra::FEVector<>;

    template <int BlockSize>
    auto MakeMatrixAdaptor(const TpetraMatrixType& rMatrix)
    {
        mAdaptor.emplace(rMatrix);
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
    std::size_t BlockSystemSize(const TpetraMatrixType& rMatrix) const noexcept
    {
        return rMatrix.getLocalNumRows() / AMGCLStaticVectorTraits<TStaticMatrix>::value;
    }

    auto MakeVectorIterator(const TpetraVectorType& rVector) const
    {
        auto view = rVector.getLocalViewHost(Tpetra::Access::ReadOnly);
        return view.data();
    }

    auto MakeVectorIterator(TpetraVectorType& rVector) const
    {
        auto view = rVector.getLocalViewHost(Tpetra::Access::ReadWrite);
        return view.data();
    }

    MPI_Comm GetCommunicator(const TpetraMatrixType& rMatrix) const noexcept
    {
        Teuchos::RCP<const Teuchos::Comm<int>> p_comm = rMatrix.getMap()->getComm();
        Teuchos::RCP<const Teuchos::MpiComm<int>> p_mpi_comm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(p_comm);
        return *(p_mpi_comm->getRawMpiComm());
    }

private:
    std::optional<tpetra_map<TpetraMatrixType>> mAdaptor;
};

template class KRATOS_API(TRILINOS_APPLICATION) AmgclMPISolver<
    TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEVector<>>,
    UblasSpace<double, Matrix, Vector>
>;
#endif

template class KRATOS_API(TRILINOS_APPLICATION) AmgclMPISolver<
    TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector>,
    UblasSpace<double, Matrix, Vector>
>;


} // namespace Kratos
