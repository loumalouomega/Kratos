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

// System includes

// External includes

// Project includes
#include "custom_python/add_trilinos_space_experimental_to_python.h"
#include "custom_python/experimental_trilinos_pointer_wrapper.h"
#include "includes/define.h"
#include "mpi/includes/mpi_data_communicator.h"

namespace Kratos::Python {
namespace py = pybind11;

#ifdef HAVE_TPETRA

namespace {

double ExperimentalDot(ExperimentalTrilinosSparseSpaceType &dummy,
                       ExperimentalTrilinosSparseSpaceType::VectorType &rX,
                       ExperimentalTrilinosSparseSpaceType::VectorType &rY) {
  return dummy.Dot(rX, rY);
}
void ExperimentalScaleAndAdd(
    ExperimentalTrilinosSparseSpaceType &dummy, const double A,
    const ExperimentalTrilinosSparseSpaceType::VectorType &rX, const double B,
    ExperimentalTrilinosSparseSpaceType::VectorType &rY) {
  dummy.ScaleAndAdd(A, rX, B, rY);
}
void ExperimentalMult(ExperimentalTrilinosSparseSpaceType &dummy,
                      ExperimentalTrilinosSparseSpaceType::MatrixType &rA,
                      ExperimentalTrilinosSparseSpaceType::VectorType &rX,
                      ExperimentalTrilinosSparseSpaceType::VectorType &rY) {
  dummy.Mult(rA, rX, rY);
}
void ExperimentalTransposeMult(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::MatrixType &rA,
    ExperimentalTrilinosSparseSpaceType::VectorType &rX,
    ExperimentalTrilinosSparseSpaceType::VectorType &rY) {
  dummy.TransposeMult(rA, rX, rY);
}
ExperimentalTrilinosSparseSpaceType::IndexType
ExperimentalSize(ExperimentalTrilinosSparseSpaceType &dummy,
                 ExperimentalTrilinosSparseSpaceType::VectorType const &rV) {
  return dummy.Size(rV);
}
ExperimentalTrilinosSparseSpaceType::IndexType
ExperimentalSize1(ExperimentalTrilinosSparseSpaceType &dummy,
                  ExperimentalTrilinosSparseSpaceType::MatrixType const &rM) {
  return dummy.Size1(rM);
}
ExperimentalTrilinosSparseSpaceType::IndexType
ExperimentalSize2(ExperimentalTrilinosSparseSpaceType &dummy,
                  ExperimentalTrilinosSparseSpaceType::MatrixType const &rM) {
  return dummy.Size2(rM);
}
void ExperimentalClearMatrix(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::MatrixPointerType &pA) {
  dummy.Clear(pA);
}
void ExperimentalClearVector(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::VectorPointerType &pX) {
  dummy.Clear(pX);
}
void ExperimentalResizeVector(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::VectorType &x,
    ExperimentalTrilinosSparseSpaceType::IndexType size) {
  dummy.Resize(x, size);
}
void ExperimentalSetToZeroMatrix(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::MatrixType &A) {
  dummy.SetToZero(A);
}
void ExperimentalSetToZeroVector(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::VectorType &x) {
  dummy.SetToZero(x);
}
double ExperimentalTwoNorm(ExperimentalTrilinosSparseSpaceType &dummy,
                           ExperimentalTrilinosSparseSpaceType::VectorType &x) {
  return dummy.TwoNorm(x);
}
void ExperimentalUnaliasedAdd(
    ExperimentalTrilinosSparseSpaceType &dummy,
    ExperimentalTrilinosSparseSpaceType::VectorType &x, const double A,
    const ExperimentalTrilinosSparseSpaceType::VectorType &y) {
  dummy.UnaliasedAdd(x, A, y);
}
ExperimentalAuxiliaryMatrixWrapper
ExperimentalCreateEmptyMatrixPointer(ExperimentalTrilinosSparseSpaceType &dummy,
                                     Teuchos::MpiComm<int> &rComm) {
  return ExperimentalAuxiliaryMatrixWrapper(
      dummy.CreateEmptyMatrixPointer(rComm));
}
ExperimentalAuxiliaryVectorWrapper
ExperimentalCreateEmptyVectorPointer(ExperimentalTrilinosSparseSpaceType &dummy,
                                     Teuchos::MpiComm<int> &rComm) {
  return ExperimentalAuxiliaryVectorWrapper(
      dummy.CreateEmptyVectorPointer(rComm));
}
ExperimentalTrilinosSparseSpaceType::MatrixType &
ExperimentalGetMatRef(ExperimentalAuxiliaryMatrixWrapper &dummy) {
  return dummy.GetReference();
}
ExperimentalTrilinosSparseSpaceType::VectorType &
ExperimentalGetVecRef(ExperimentalAuxiliaryVectorWrapper &dummy) {
  return dummy.GetReference();
}

} // namespace

#endif

void AddBasicOperationsExperimental(pybind11::module &m) {
#ifdef HAVE_TPETRA

  py::class_<Teuchos::MpiComm<int>>(m, "Experimental_TeuchosMpiComm")
      //.def(py::init< Teuchos::MpiComm<int>& >())
      ;

  py::class_<Tpetra::FECrsMatrix<>>(m, "Experimental_FECrsMatrix")
      //.def(py::init< Tpetra::FECrsMatrix<>& >())
      ;

  py::class_<Tpetra::FEMultiVector<>>(m, "Experimental_FEMultiVector")
      //.def(py::init< Tpetra::FEMultiVector<>& >())
      ;

  py::class_<ExperimentalAuxiliaryMatrixWrapper>(
      m, "ExperimentalTrilinosMatrixPointer")
      .def("GetReference", ExperimentalGetMatRef,
           py::return_value_policy::reference_internal);

  py::class_<ExperimentalAuxiliaryVectorWrapper>(
      m, "ExperimentalTrilinosVectorPointer")
      .def("GetReference", ExperimentalGetVecRef,
           py::return_value_policy::reference_internal);

  py::class_<ExperimentalTrilinosSparseSpaceType>(
      m, "ExperimentalTrilinosSparseSpace")
      .def(py::init<>())
      .def("ClearMatrix", ExperimentalClearMatrix)
      .def("ClearVector", ExperimentalClearVector)
      .def("ResizeVector", ExperimentalResizeVector)
      .def("SetToZeroMatrix", ExperimentalSetToZeroMatrix)
      .def("SetToZeroVector", ExperimentalSetToZeroVector)
      .def("TwoNorm", ExperimentalTwoNorm)
      .def("Dot", ExperimentalDot)
      .def("Mult", ExperimentalMult)
      .def("TransposeMult", ExperimentalTransposeMult)
      .def("Size", ExperimentalSize)
      .def("Size1", ExperimentalSize1)
      .def("Size2", ExperimentalSize2)
      .def("UnaliasedAdd", ExperimentalUnaliasedAdd)
      .def("ScaleAndAdd", ExperimentalScaleAndAdd)
      .def("CreateEmptyMatrixPointer", ExperimentalCreateEmptyMatrixPointer)
      .def("CreateEmptyVectorPointer", ExperimentalCreateEmptyVectorPointer)
      .def_static("IsDistributed",
                  &ExperimentalTrilinosSparseSpaceType::IsDistributed)
      .def_static(
          "FastestDirectSolverList",
          &ExperimentalTrilinosSparseSpaceType::FastestDirectSolverList);

#endif
}

} // namespace Kratos::Python.
