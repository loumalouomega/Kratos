//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//
// Tpetra (experimental Trilinos space) counterpart of relaxed_dof_updater.cpp

#ifdef HAVE_TPETRA

// Project includes
#include "trilinos_space_experimental.h"

#include "custom_strategies/relaxed_dof_updater.h"

namespace Kratos
{

namespace
{
using ExperimentalSparseSpaceType = TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>>;
}

template <>
void RelaxedDofUpdater<ExperimentalSparseSpaceType>::Initialize(
    const DofsArrayType& rDofSet,
    const SystemVectorType& rDx)
{
    // CreateImport already performs the global DOF-count consistency check
    this->mpDofImport = ExperimentalSparseSpaceType::CreateImport(rDofSet, rDx);
    this->mImportIsInitialized = true;
}

template <>
void RelaxedDofUpdater<ExperimentalSparseSpaceType>::Clear()
{
    this->mpDofImport.reset();
    this->mImportIsInitialized = false;
}

template <>
void RelaxedDofUpdater<ExperimentalSparseSpaceType>::UpdateDofs(
    DofsArrayType& rDofSet,
    const SystemVectorType& rDx)
{
    KRATOS_TRY;

    if (!this->mImportIsInitialized)
        this->Initialize(rDofSet, rDx);

    using ST = typename ExperimentalSparseSpaceType::ST;
    using LO = typename ExperimentalSparseSpaceType::LO;
    using GO = typename ExperimentalSparseSpaceType::GO;
    using NT = typename ExperimentalSparseSpaceType::NT;
    using ImportType = typename ExperimentalSparseSpaceType::ImportType;

    const std::size_t system_size = ExperimentalSparseSpaceType::Size(rDx);

    // recovering the concrete import type from the type-erased member
    const auto p_dof_import = std::static_pointer_cast<ImportType>(this->mpDofImport);

    // defining a temporary vector to gather all of the values needed
    const auto p_target_map = p_dof_import->getTargetMap();
    Tpetra::MultiVector<ST, LO, GO, NT> local_dx(p_target_map, 1);

    // importing in the temporary vector the values
    local_dx.doImport(rDx, *p_dof_import, Tpetra::INSERT);

    // performing the update
    // NOTE: plain serial loop on purpose: mixing Kratos OpenMP parallel utilities with
    // Kokkos-backed Tpetra host views causes conflicts (see TrilinosBlockBuilderAndSolver)
    const auto data = local_dx.getData(0);
    for (auto it_dof = rDofSet.begin(); it_dof != rDofSet.end(); ++it_dof) {
        if (it_dof->IsFree()) {
            const std::size_t global_id = it_dof->EquationId();
            if (global_id < system_size) {
                const LO local_id = p_target_map->getLocalElement(static_cast<GO>(global_id));
                it_dof->GetSolutionStepValue() += this->mRelaxationFactor * data[local_id];
            }
        }
    }

    KRATOS_CATCH("");
}

template <>
std::string RelaxedDofUpdater<ExperimentalSparseSpaceType>::Info() const
{
    std::stringstream buffer;
    buffer << "RelaxedDofUpdater - TrilinosSpaceExperimental";
    return buffer.str();
}

template class RelaxedDofUpdater<ExperimentalSparseSpaceType>;

} // namespace Kratos

#endif // HAVE_TPETRA
