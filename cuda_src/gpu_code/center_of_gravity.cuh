#pragma once

#include "map_reduce.cuh"

namespace CENTER_OF_GRAVITY {
  template<typename T, DIMENSION REDUCING_DIMENSION>
  struct COG_REDUCE_FUNCTION {


    constexpr static __device__ __host__ u32 index(u64 global_index, const Extent<3>& extent) noexcept {
      u32 ret = 0;

      if constexpr (REDUCING_DIMENSION == DIMENSION::X) { // X Dim
        ret = global_index / (extent.x() * extent.y());
      } else if constexpr  (REDUCING_DIMENSION == DIMENSION::Y) { // Y DIM
        ret = global_index / extent.x() % extent.y();
      } else if constexpr (REDUCING_DIMENSION == DIMENSION::Z) { // Z DIM
        ret = global_index % extent.x();
      }

      return ret;
    }

    constexpr static __device__ __host__ f32 map_to(u64 global_index, const Volume<3, T>& vol) noexcept {
      return static_cast<f32>(index(global_index, vol.extent())) * static_cast<f32>(vol.at(global_index));
    }

    constexpr static __device__ __host__ f32 apply(const f32& a, const f32& b) noexcept {
      return a + b;
    }

    constexpr static __device__ __host__ bool equals(const f32& a, const f32& b) noexcept {
      return a == b;
    }

    constexpr static __device__ __host__ f32 identity() noexcept {
      return 0;
    }

    constexpr static __device__ __host__ f32 remove_volatile(volatile f32& v) noexcept {
      f32 vv = v;
      return vv;
    }
  };
}


