/**
 * @file volume.cuh
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include<vector>

#include"../concepts.cuh"
#include"extent.cuh"
#include"point.cuh"
#include"index.cuh"

template<uint8_t DIMENSIONS, typename T>
struct Volume {
  T* data = nullptr;
  Extent<DIMENSIONS> m_extent;
  T default_value = 0;

  constexpr  __device__ __host__  size_t elements() const {
    return m_extent.elements();
  }

  constexpr  __device__ __host__  bool is_allocated() const {
    return data == nullptr;
  }

  dicomNodeError_t set_extent(const std::vector<ssize_t>& dims){
    return m_extent.set_dimensions(dims);
  }

  constexpr __device__ __host__ const Extent<DIMENSIONS>& extent() const {
    return m_extent;
  }

  constexpr __device__ __host__ const T& at(const Index<DIMENSIONS>& index) const {
    const cuda::std::optional<u64> flat_index = m_extent.flat_index(index);

    if(!flat_index.has_value()){
      return default_value;
    }

    return data[flat_index.value()];
  }

  __device__ __host__ T interpolate_at_index_point(const Point<DIMENSIONS>& p) const {
    T value = 0;

    float alphas[DIMENSIONS];
    Index<DIMENSIONS> index_lower_point;

    for (u8 d = 0; d < DIMENSIONS; d++) {
      alphas[d] = p[d] - floorf(p[d]);
      index_lower_point[d] = floorf(p[d]);
    }

    // A N-dimensional cube has 2^N points
    constexpr u16 MAX_POINTS = 1 << DIMENSIONS;

    // Coordinate index is the point index of points we need to visit
    for (u8 coordinate_index = 0; coordinate_index < MAX_POINTS; coordinate_index++) {
      float modifier = 1.0f;
      for (u8 d = 0; d < DIMENSIONS; d++ ) {
        modifier *= (coordinate_index >> d) & 1u ? alphas[d] : 1.0f - alphas[d];
      }
      Index<DIMENSIONS> index = index_lower_point + dimensional_offset<DIMENSIONS>(coordinate_index);

      value += modifier * this->at(index);
    }

    return value;
  }
};

//static_assert(CVolume<Volume, 3, float>, "Volume doesn't fulfill volume concepts");