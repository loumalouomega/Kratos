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

#pragma once

#ifdef HAVE_TPETRA

// System includes

// External includes

// Project includes
#include "trilinos_space_experimental.h"

namespace Kratos {

typedef TrilinosSpaceExperimental<Tpetra::FECrsMatrix<>, Tpetra::Vector<>>
    ExperimentalTrilinosSparseSpaceType;

class ExperimentalAuxiliaryMatrixWrapper {
public:
  typedef typename ExperimentalTrilinosSparseSpaceType::MatrixType
      TrilinosMatrixType;
  typedef typename ExperimentalTrilinosSparseSpaceType::MatrixPointerType
      TrilinosMatrixPointerType;

  ExperimentalAuxiliaryMatrixWrapper(TrilinosMatrixPointerType p) : mp(p) {};
  virtual ~ExperimentalAuxiliaryMatrixWrapper() {}
  TrilinosMatrixPointerType &GetPointer() { return mp; }
  TrilinosMatrixType &GetReference() { return *mp; }

private:
  TrilinosMatrixPointerType mp;
};

class ExperimentalAuxiliaryVectorWrapper {
public:
  typedef typename ExperimentalTrilinosSparseSpaceType::VectorType
      TrilinosVectorType;
  typedef typename ExperimentalTrilinosSparseSpaceType::VectorPointerType
      TrilinosVectorPointerType;

  ExperimentalAuxiliaryVectorWrapper(TrilinosVectorPointerType p) : mp(p) {};
  virtual ~ExperimentalAuxiliaryVectorWrapper() {}
  TrilinosVectorPointerType &GetPointer() { return mp; }
  TrilinosVectorType &GetReference() { return *mp; }

private:
  TrilinosVectorPointerType mp;
};

} // namespace Kratos

#endif