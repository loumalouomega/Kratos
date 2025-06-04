//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Pooyan Dadvand
//

#pragma once

// System includes
#include <string>
#include <iostream>

// External includes
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

// Project includes
#include "includes/define.h"

namespace Kratos
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{
    // using namespace boost::numeric::ublas; // This was causing namespace pollution

    template <typename TDataType> using DenseMatrix = boost::numeric::ublas::matrix<TDataType>;
    template <typename TDataType> using DenseVector = boost::numeric::ublas::vector<TDataType>;

    template <typename TDataType, std::size_t TSize1, std::size_t TSize2> using BoundedMatrix = boost::numeric::ublas::bounded_matrix<TDataType, TSize1, TSize2>;
    template <typename TDataType, std::size_t TSize> using BoundedVector = boost::numeric::ublas::bounded_vector<TDataType, TSize>;


    typedef boost::numeric::ublas::vector<double> Vector;
    typedef boost::numeric::ublas::unit_vector<double> UnitVector;
    typedef boost::numeric::ublas::zero_vector<double> ZeroVector;
    typedef boost::numeric::ublas::scalar_vector<double> ScalarVector;
    //typedef boost::numeric::ublas::sparse_vector<double> SparseVector;
    typedef boost::numeric::ublas::mapped_vector<double> SparseVector;

    typedef boost::numeric::ublas::compressed_vector<double> CompressedVector;
    typedef boost::numeric::ublas::coordinate_vector<double> CoordinateVector;
    typedef boost::numeric::ublas::vector_range<Vector> VectorRange; // Vector is already boost::numeric::ublas::vector
    typedef boost::numeric::ublas::vector_slice<Vector> VectorSlice; // Vector is already boost::numeric::ublas::vector

    typedef boost::numeric::ublas::matrix<double> Matrix;
    typedef boost::numeric::ublas::identity_matrix<double> IdentityMatrix;
    typedef boost::numeric::ublas::zero_matrix<double> ZeroMatrix;
    typedef boost::numeric::ublas::scalar_matrix<double> ScalarMatrix;
    typedef boost::numeric::ublas::triangular_matrix<double> TriangularMatrix;
    typedef boost::numeric::ublas::symmetric_matrix<double> SymmetricMatrix;
    typedef boost::numeric::ublas::hermitian_matrix<double> HermitianMatrix;
    typedef boost::numeric::ublas::banded_matrix<double> BandedMatrix;
    //typedef boost::numeric::ublas::sparse_matrix<double> SparseMatrix;
    typedef boost::numeric::ublas::mapped_matrix<double> SparseMatrix;
    typedef boost::numeric::ublas::coordinate_matrix<double> CoordinateMatrix;
    typedef boost::numeric::ublas::matrix_column<Matrix> MatrixColumn; // Matrix is already boost::numeric::ublas::matrix
    typedef boost::numeric::ublas::matrix_vector_range<Matrix> MatrixVectorRange; // Matrix is already boost::numeric::ublas::matrix
    typedef boost::numeric::ublas::matrix_vector_slice<Matrix> MatrixVectorSlice; // Matrix is already boost::numeric::ublas::matrix
    typedef boost::numeric::ublas::matrix_range<Matrix> MatrixRange; // Matrix is already boost::numeric::ublas::matrix
    typedef boost::numeric::ublas::matrix_slice<Matrix> MatrixSlice; // Matrix is already boost::numeric::ublas::matrix

	template <typename TExpressionType> using MatrixRow = boost::numeric::ublas::matrix_row<TExpressionType>;

    typedef boost::numeric::ublas::compressed_matrix<double> CompressedMatrix;

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{


///@}
///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

///@}
}  // namespace Kratos.