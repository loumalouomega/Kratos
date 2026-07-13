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
//  Kratos-owned Tpetra counterpart of amgcl/adapter/epetra.hpp (Denis Demidov, Riccardo Rossi).
//  It lives in TrilinosApplication (NOT in external_libraries/amgcl) to keep the vendored
//  amgcl sources untouched.
//

#pragma once

// System includes
#include <vector>

// External includes
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Import.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <amgcl/backend/interface.hpp>
#include <amgcl/detail/sort_row.hpp>

namespace amgcl {
namespace adapter {

/// @cond
namespace tpetra_detail {

/// Prefer getLocalNumRows / getLocalNumEntries (Trilinos >= 13.2 new API); fall back to the deprecated getNode* names.
template <class TMatrix>
auto GetNumLocalRowsImpl(const TMatrix& rMatrix, int)
    -> decltype(rMatrix.getLocalNumRows(), std::size_t{})
{
    return static_cast<std::size_t>(rMatrix.getLocalNumRows());
}

template <class TMatrix>
std::size_t GetNumLocalRowsImpl(const TMatrix& rMatrix, long)
{
    return static_cast<std::size_t>(rMatrix.getNodeNumRows());
}

template <class TMatrix>
std::size_t GetNumLocalRows(const TMatrix& rMatrix)
{
    return GetNumLocalRowsImpl(rMatrix, 0);
}

template <class TMatrix>
auto GetNumLocalEntriesImpl(const TMatrix& rMatrix, int)
    -> decltype(rMatrix.getLocalNumEntries(), std::size_t{})
{
    return static_cast<std::size_t>(rMatrix.getLocalNumEntries());
}

template <class TMatrix>
std::size_t GetNumLocalEntriesImpl(const TMatrix& rMatrix, long)
{
    return static_cast<std::size_t>(rMatrix.getNodeNumEntries());
}

template <class TMatrix>
std::size_t GetNumLocalEntries(const TMatrix& rMatrix)
{
    return GetNumLocalEntriesImpl(rMatrix, 0);
}

} // namespace tpetra_detail
/// @endcond

/// Adapts a Tpetra::CrsMatrix (or anything derived from it, e.g. Tpetra::FECrsMatrix) for amgcl.
/** As in the Epetra adapter, the local rows of every rank are renumbered into a globally
 * consecutive chunk (amgcl-MPI requirement) and the column indices are translated accordingly
 * through an import over the column map.
 */
template <class TMatrix>
class tpetra_map_adapter {
public:
    typedef double value_type;

    using LO = typename TMatrix::local_ordinal_type;
    using GO = typename TMatrix::global_ordinal_type;
    using NT = typename TMatrix::node_type;

    using OrderVectorType = Tpetra::Vector<GO, LO, GO, NT>;

    tpetra_map_adapter(const TMatrix& rA)
        : mrA(rA)
    {
        const auto p_row_map = rA.getRowMap();
        const auto p_col_map = rA.getColMap();
        const auto p_comm = p_row_map->getComm();

        const GO local_entries = static_cast<GO>(tpetra_detail::GetNumLocalRows(rA));
        GO entries_before = 0;
        Teuchos::scan(*p_comm, Teuchos::REDUCE_SUM, local_entries, Teuchos::outArg(entries_before)); // inclusive scan
        entries_before -= local_entries;

        // Permutation: local row i -> globally consecutive id (entries_before + i)
        OrderVectorType perm(p_row_map);
        {
            auto perm_data = perm.getDataNonConst(0);
            for (GO i = 0; i < local_entries; ++i) {
                perm_data[i] = entries_before + i;
            }
        }

        // Translate the permutation onto the column map so column ids can be renumbered
        mpOrder = Teuchos::rcp(new OrderVectorType(p_col_map));
        Tpetra::Import<LO, GO, NT> importer(p_row_map, p_col_map);
        mpOrder->doImport(perm, importer, Tpetra::INSERT);
        mOrder = mpOrder->getData(0);
    }

    size_t rows() const {
        return tpetra_detail::GetNumLocalRows(mrA);
    }

    size_t cols() const {
        return static_cast<size_t>(mrA.getGlobalNumCols());
    }

    size_t nonzeros() const {
        return tpetra_detail::GetNumLocalEntries(mrA);
    }

    class row_iterator {
        public:
            typedef GO     col_type;
            typedef double val_type;

            row_iterator(
                    const TMatrix& rA,
                    const Teuchos::ArrayRCP<const GO>& rOrder,
                    LO Row
                    )
            {
                typename TMatrix::local_inds_host_view_type indices;
                typename TMatrix::values_host_view_type values;
                rA.getLocalRowView(Row, indices, values);

                const std::size_t nnz = indices.extent(0);
                mColCopy.resize(nnz);
                mValCopy.resize(nnz);
                for (std::size_t k = 0; k < nnz; ++k) {
                    mColCopy[k] = rOrder[indices(k)];
                    mValCopy[k] = static_cast<val_type>(values(k));
                }

                if (nnz > 0) {
                    amgcl::detail::sort_row(mColCopy.data(), mValCopy.data(), static_cast<int>(nnz));
                }

                mpCol = mColCopy.data();
                mpEnd = mColCopy.data() + nnz;
                mpVal = mValCopy.data();
            }

            operator bool() const {
                return mpCol != mpEnd;
            }

            row_iterator& operator++() {
                ++mpCol;
                ++mpVal;
                return *this;
            }

            col_type col() const {
                return *mpCol;
            }

            val_type value() const {
                return *mpVal;
            }

        private:
            col_type* mpCol;
            col_type* mpEnd;
            val_type* mpVal;

            std::vector<col_type> mColCopy;
            std::vector<val_type> mValCopy;
    };

    row_iterator row_begin(int Row) const {
        return row_iterator(mrA, mOrder, static_cast<LO>(Row));
    }

private:
    const TMatrix& mrA;
    Teuchos::RCP<OrderVectorType> mpOrder;
    Teuchos::ArrayRCP<const GO> mOrder;
};

/// Adapts a Tpetra::CrsMatrix (or derived) for amgcl.
/// SFINAE-constrained on the Tpetra ordinal typedefs so that Epetra matrices keep
/// resolving to the non-template overload of amgcl/adapter/epetra.hpp.
template <class TMatrix,
          class = typename TMatrix::local_ordinal_type,
          class = typename TMatrix::global_ordinal_type>
tpetra_map_adapter<TMatrix> map(const TMatrix& rA) {
    return tpetra_map_adapter<TMatrix>(rA);
}

} // namespace adapter
} // namespace amgcl
