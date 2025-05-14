/**
 * @file space.cuh
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-04-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include"../declarations.cuh"

/**
 * @brief A space is description of a linear space of n Dimensions where the
 * starting point is the coordinate at Index (0,0,0, ...). The space extents out
 * to defined by the domain.
 *
 * @tparam DIMENSIONS
 */
template<uint8_t DIMENSIONS>
class Space {
  public:

    /**
     * @brief The coordinates for the Index (0,0,0,...)
     *
     */
    Point<DIMENSIONS> starting_point;
    SquareMatrix<DIMENSIONS> basis;
    SquareMatrix<DIMENSIONS> inverted_basis;
    Extent<DIMENSIONS> extent;

  __device__ __host__ Point<DIMENSIONS> at_index(const Index<DIMENSIONS>& index) const {
    Point<DIMENSIONS> point{index};
    return point * basis + starting_point;
  }

  __device__ __host__ Point<DIMENSIONS> interpolate_point(const Point<DIMENSIONS>& point) const {
    return (point - starting_point) * inverted_basis;
  }

  __device__ __host__ Index<DIMENSIONS> index(const uint64_t& flat_index) const {
    return extent.from_flat_index(flat_index);
  }

  __device__ __host__ size_t elements() const {
    return extent.elements();
  }
};