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
#include <cstddef>

// External includes
#include <mdspan.hpp>

// Project includes

namespace Kratos::Future {

/**
 * @file mdspan.h
 * @brief Provides type aliases and utilities for `std::mdspan`.
 * @details This header introduces a set of standardized aliases for the core
 *          components of `std::mdspan`, such as layouts, accessors, and extents.
 *          It also defines convenient aliases for common `mdspan` types (e.g.,
 *          1D and 2D views) to simplify their usage throughout the Kratos codebase.
 *          A stream insertion operator for easy debugging and inspection of `mdspan`
 *          instances is also included.
 */

///@name mdspan Component Aliases
///@{

/**
 * @brief Alias for a left-to-right (column-major) layout mapping.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_left
 */
using layout_left = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_left;

/**
 * @brief Alias for a right-to-left (row-major) layout mapping.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_right
 */
using layout_right = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_right;

/**
 * @brief Alias for a generalized strided layout mapping.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/layout_stride
 */
using layout_stride = MDSPAN_IMPL_STANDARD_NAMESPACE::layout_stride;

/**
 * @brief Alias for the default accessor policy, which provides element access via `operator[]`.
 * @tparam T The element type being accessed.
 */
template <typename T>
using default_accessor = MDSPAN_IMPL_STANDARD_NAMESPACE::default_accessor<T>;

/**
 * @brief Alias for compile-time extents of an `mdspan`.
 * @tparam IndexType The type used for indexing.
 * @tparam Extents A parameter pack of compile-time dimension extents.
 */
template <typename IndexType, std::size_t... Extents>
using extents = MDSPAN_IMPL_STANDARD_NAMESPACE::extents<IndexType, Extents...>;

/**
 * @brief Alias for run-time (dynamic) extents of an `mdspan`.
 * @tparam IndexType The type used for indexing.
 * @tparam Rank The number of dimensions (rank) of the `mdspan`.
 */
template <typename IndexType, std::size_t Rank>
using dextents = MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<IndexType, Rank>;

/**
 * @brief A constant representing the full extent in an `mdspan`, allowing for dynamic sizing.
 * @details This is used to specify that a dimension should take the full size of the underlying data.
 * @see https://en.cppreference.com/w/cpp/container/mdspan/full_extent
 */
inline constexpr auto full_extent = MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent;

///@}
///@name Common mdspan Type Aliases
///@{

/**
 * @brief A non-owning, multi-dimensional view over a contiguous sequence of objects.
 * @details This is the primary alias for `std::mdspan`, configured with custom
 * layout and accessor policies. It serves as a versatile, non-owning handle
 * to multi-dimensional data.
 * @tparam ElementType The type of elements in the view.
 * @tparam Extents An `extents` object specifying the dimensions.
 * @tparam LayoutPolicy The memory layout policy (e.g., `layout_right`).
 * @tparam AccessorPolicy The policy for accessing elements.
 */
template <
    typename ElementType,
    typename Extents,
    typename LayoutPolicy = layout_right,
    typename AccessorPolicy = default_accessor<ElementType>
>
using mdspan = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    ElementType,
    Extents,
    LayoutPolicy,
    AccessorPolicy
>;

/**
 * @brief A non-owning `mdspan` view with dynamic extents for a specified rank.
 * @tparam T The type of elements in the view.
 * @tparam Rank The number of dimensions (rank).
 * @tparam Layout The memory layout policy (default is row-major).
 */
template <typename T, std::size_t Rank, typename Layout = layout_right>
using mdspan_view = mdspan<T, dextents<std::size_t, Rank>, Layout>;

/**
 * @brief A non-owning 1D view, conceptually equivalent to `std::span`.
 * @tparam T The type of elements in the view.
 */
template <typename T>
using mdspan_1d_view = mdspan_view<T, 1>;

/**
 * @brief A non-owning 2D view, conceptually a view of a matrix.
 * @tparam T The type of elements in the view.
 */
template <typename T>
using mdspan_2d_view = mdspan_view<T, 2>;

///@}
///@name Stream Operators
///@{

/**
 * @brief Prints a summary of an `mdspan` to an output stream.
 * @details This operator provides a concise representation of an `mdspan`, including
 * its rank and the extent of each dimension. This is useful for debugging.
 * @param rOStream The output stream.
 * @param rData The `mdspan` to print.
 * @return The output stream.
 */
template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
std::ostream& operator<<(std::ostream& rOStream, const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& rData)
{
    rOStream << "mdspan(rank=" << rData.rank() << ", extents=[";
    for (std::size_t i = 0; i < rData.rank(); ++i) {
        rOStream << rData.extent(i) << (i == rData.rank() - 1 ? "" : ", ");
    }
    rOStream << "])";
    return rOStream;
}

/// @}

} // namespace Kratos::Future