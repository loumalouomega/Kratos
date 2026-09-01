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

#pragma once

// System includes
#include <cstddef>
#include <type_traits>

// External includes

// Project includes

namespace Kratos::Internals {

///@name Kratos Classes
///@{

/**
 * @class EigenArrayProxy
 * @brief Iterable view over a raw backend array.
 * @details The boost::numeric::ublas containers expose their storage as an
 * array object (unbounded_array/bounded_array) with begin()/end()/operator[],
 * whereas Eigen exposes a raw pointer. This view gives the Eigen storage the
 * uBLAS surface *and* decays back to the pointer, so both spellings
 *     data().begin()      &data()[0]
 * compile against either backend and no call site has to branch on it.
 *
 * It backs both the dense data() accessors and the CSR
 * value_data()/index1_data()/index2_data() accessors of the sparse wrapper.
 */
template<class T>
class EigenArrayProxy
{
public:
    ///@name Type Definitions
    ///@{

    using value_type = std::remove_const_t<T>;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = std::size_t;

    ///@}
    ///@name Life Cycle
    ///@{

    EigenArrayProxy(T* pData, const std::size_t Size) : mpData(pData), mSize(Size) {}

    ///@}
    ///@name Operations
    ///@{

    T* begin() { return mpData; }
    T* end() { return mpData + mSize; }
    const T* begin() const { return mpData; }
    const T* end() const { return mpData + mSize; }

    T& operator[](const std::size_t Index) { return mpData[Index]; }
    const T& operator[](const std::size_t Index) const { return mpData[Index]; }

    std::size_t size() const { return mSize; }

    /// Implicit decay to the raw buffer. The proxy is a non-owning view, so
    /// the conversion is shallow-const (as in std::span): it lets the call
    /// sites that hand the storage to a C API or an Eigen::Map keep passing
    /// data() directly, exactly as they do with the raw storage pointer.
    operator T*() const { return mpData; }

    ///@}

private:
    ///@name Member Variables
    ///@{

    T* mpData;
    std::size_t mSize;

    ///@}
};

///@}

} // namespace Kratos::Internals
