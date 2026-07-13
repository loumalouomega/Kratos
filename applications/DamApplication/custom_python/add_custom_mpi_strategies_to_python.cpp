//
//   Project Name:
//   Last modified by:    $Author:  $
//   Date:                $Date:  $
//   Revision:            $Revision: $
//

// External includes
#include "spaces/ublas_space.h"

// Project includes
#include "custom_python/add_custom_mpi_strategies_to_python.h"
#include "includes/kratos_parameters.h"

//Trilinos includes
#include "mpi.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "trilinos_space.h"
#ifdef HAVE_TPETRA
#include "trilinos_space_experimental.h"
#endif

//linear solvers

//strategies

//builders and solvers

//schemes
// NOTE: the historical trilinos_* scheme wrappers no longer exist; the generic
// space-templated schemes are registered with the Trilinos spaces instead
#include "custom_strategies/schemes/incrementalupdate_static_damped_smoothing_scheme.hpp"
#include "custom_strategies/schemes/dam_UP_scheme.hpp"


namespace Kratos
{

namespace Python
{

namespace py = pybind11;

namespace
{

template<class TSparseSpace>
void RegisterMPIStrategies(pybind11::module& m, const std::string& Prefix)
{
    typedef UblasSpace<double, Matrix, Vector> TrilinosLocalSpaceType;

    typedef Scheme< TSparseSpace, TrilinosLocalSpaceType > TrilinosBaseSchemeType;

    typedef IncrementalUpdateStaticDampedSmoothingScheme<TSparseSpace, TrilinosLocalSpaceType> TrilinosIncrementalUpdateStaticDampedSchemeType;
    typedef DamUPScheme<TSparseSpace, TrilinosLocalSpaceType> TrilinosDamUPSchemeType;

    // Schemes
    py::class_< TrilinosIncrementalUpdateStaticDampedSchemeType, typename TrilinosIncrementalUpdateStaticDampedSchemeType::Pointer, TrilinosBaseSchemeType>
    (m, (Prefix + "IncrementalUpdateStaticDampedScheme").c_str())
    .def(py::init< double, double >());
    py::class_< TrilinosDamUPSchemeType, typename TrilinosDamUPSchemeType::Pointer, TrilinosBaseSchemeType >
    (m, (Prefix + "DamUPScheme").c_str())
    .def(py::init< double, double, double, double >());
}

} // anonymous namespace

void  AddCustomMPIStrategiesToPython(pybind11::module& m)
{
    typedef TrilinosSpace<Epetra_FECrsMatrix, Epetra_FEVector> TrilinosSparseSpaceType;

    RegisterMPIStrategies<TrilinosSparseSpaceType>(m, "Trilinos");

#ifdef HAVE_TPETRA
    typedef TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::FEMultiVector<>> TrilinosExperimentalSparseSpaceType;

    RegisterMPIStrategies<TrilinosExperimentalSparseSpaceType>(m, "TrilinosExperimental");
#endif
}

}  // namespace Python.
} // Namespace Kratos
