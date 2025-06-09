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
    using namespace boost::numeric::ublas; // This may cause namespace pollution

    template <typename TDataType> 
    using DenseMatrix = boost::numeric::ublas::matrix<TDataType>;

    template <typename TDataType> 
    using DenseVector = boost::numeric::ublas::vector<TDataType>;

    template <typename TDataType, std::size_t TSize1, std::size_t TSize2> 
    using BoundedMatrix = boost::numeric::ublas::bounded_matrix<TDataType, TSize1, TSize2>;

    template <typename TDataType, std::size_t TSize> 
    using BoundedVector = boost::numeric::ublas::bounded_vector<TDataType, TSize>;

    using Vector = boost::numeric::ublas::vector<double>;
    using UnitVector = boost::numeric::ublas::unit_vector<double>;
    using ZeroVector = boost::numeric::ublas::zero_vector<double>;
    using ScalarVector = boost::numeric::ublas::scalar_vector<double>;
    //using SparseVector = boost::numeric::ublas::sparse_vector<double>;
    using SparseVector = boost::numeric::ublas::mapped_vector<double>;

    using CompressedVector = boost::numeric::ublas::compressed_vector<double>;
    using CoordinateVector = boost::numeric::ublas::coordinate_vector<double>;
    using VectorRange = boost::numeric::ublas::vector_range<Vector>;
    using VectorSlice = boost::numeric::ublas::vector_slice<Vector>;

    using Matrix = boost::numeric::ublas::matrix<double>;
    using IdentityMatrix = boost::numeric::ublas::identity_matrix<double>;
    using ZeroMatrix = boost::numeric::ublas::zero_matrix<double>;
    using ScalarMatrix = boost::numeric::ublas::scalar_matrix<double>;
    using TriangularMatrix = boost::numeric::ublas::triangular_matrix<double>;
    using SymmetricMatrix = boost::numeric::ublas::symmetric_matrix<double>;
    using HermitianMatrix = boost::numeric::ublas::hermitian_matrix<double>;
    using BandedMatrix = boost::numeric::ublas::banded_matrix<double>;
    //using SparseMatrix = boost::numeric::ublas::sparse_matrix<double>;
    using SparseMatrix = boost::numeric::ublas::mapped_matrix<double>;
    using CoordinateMatrix = boost::numeric::ublas::coordinate_matrix<double>;
    using MatrixColumn = boost::numeric::ublas::matrix_column<Matrix>;
    using MatrixVectorRange = boost::numeric::ublas::matrix_vector_range<Matrix>;
    using MatrixVectorSlice = boost::numeric::ublas::matrix_vector_slice<Matrix>;
    using MatrixRange = boost::numeric::ublas::matrix_range<Matrix>;
    using MatrixSlice = boost::numeric::ublas::matrix_slice<Matrix>;

    template <typename TExpressionType>
    using VectorExpression = boost::numeric::ublas::vector_expression<TExpressionType>;

    template <typename TExpressionType>
    using MatrixRow = boost::numeric::ublas::matrix_row<TExpressionType>;

    using CompressedMatrix = boost::numeric::ublas::compressed_matrix<double>;

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