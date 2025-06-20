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
#include <iostream>

// External includes
#include <mdspan/mdspan.hpp>

// Project includes

namespace Kratos::Future {

/// @name Standard mdspan component aliases
/// @{

/**
 * @brief Alias for std::layout_left.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_left
 */
using layout_left = std::layout_left;

/**
 * @brief Alias for std::layout_right.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_right
 */
using layout_right = std::layout_right;

/**
 * @brief Alias for std::layout_stride.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_stride
 */
using layout_stride = std::layout_stride;

/**
 * @brief Alias for std::default_accessor.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/default_accessor
 */
template <class T>
using default_accessor = std::default_accessor<T>;

/**
 * @brief Alias for std::extents, representing compile-time extents.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/extents
 */
template <class IndexType, std::size_t... Extents>
using extents = std::extents<IndexType, Extents...>;

/**
 * @brief Alias for std::dextents, representing run-time extents.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/dextents
 */
template <class IndexType, std::size_t Rank>
using dextents = std::dextents<IndexType, Rank>;

/// @}
/// @name Common mdspan type aliases
/// @{

/**
 * @brief A non-owning view with dynamic extents for a given rank.
 *
 * @tparam T The type of the elements.
 * @tparam Rank The number of dimensions.
 * @tparam Layout The layout mapping (default: layout_right).
 */
template <class T, std::size_t Rank, class Layout = layout_right>
using mdspan_view = std::mdspan<T, dextents<std::size_t, Rank>, Layout>;

/**
 * @brief A non-owning 1D view, conceptually similar to std::span.
 * @tparam T The type of the elements.
 */
template <class T>
using mdspan_1d_view = mdspan_view<T, 1>;

/**
 * @brief A non-owning 2D view, conceptually a view of a matrix.
 * @tparam T The type of the elements.
 */
template <class T>
using mdspan_2d_view = mdspan_view<T, 2>;

/// @}
/// @name Stream Operators
/// @{

/**
 * @brief A utility to print the extents and basic info of an mdspan.
 */
template <class T, class E, class L, class A>
std::ostream& operator<<(std::ostream& rOStream, const std::mdspan<T, E, L, A>& rData)
{
    rOStream << "mdspan with extents: (";
    for (std::size_t i = 0; i < rData.rank(); ++i) {
        rOStream << rData.extent(i) << (i == rData.rank() - 1 ? "" : ", ");
    }
    rOStream << ")";
    return rOStream;
}

/// @}

} // namespace Kratos::Future