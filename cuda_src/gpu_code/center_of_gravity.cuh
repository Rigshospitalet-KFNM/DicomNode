#pragma once

#include <array>

#include "core/core.cuh"
#include "map_reduce.cuh"

namespace CENTER_OF_GRAVITY {
  template<typename T, DIMENSION REDUCING_DIMENSION>
  struct COG_REDUCE_FUNCTION {
    constexpr static __device__ __host__ u32 index(const u64& global_index, const Extent<3>& extent) noexcept {
      if constexpr (REDUCING_DIMENSION == DIMENSION::X) { // X Dim
        return global_index / (extent.x() * extent.y());
      } else if constexpr  (REDUCING_DIMENSION == DIMENSION::Y) { // Y DIM
        return global_index / extent.x() % extent.y();
      } else  { // Z DIM
        static_assert(REDUCING_DIMENSION == DIMENSION::Z, "Not valid reducing dimension");
        return global_index % extent.x();
      }
    }

    constexpr static __device__ __host__ f32 map_to(const u64& global_index, const Volume<3, T>& vol) noexcept {
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


template<typename T>
  /**
   * finds the center of gravity in a device volume
   * @tparam T The datatype of the volume
   * @param volume - Volume of data, that we wish to find the center of gravity of
   * @param out_center_of_gravity - Out Variable for the point since the return
   * value have been taken by the error code
   */
  __host__ dicomNodeError_t center_of_gravity(
    const Volume<3, T>& volume,
    Point<3>& out_center_of_gravity
  ) noexcept {
  T sum = 0;

  DicomNodeRunner runner;

  runner | [&]() {
    return reduce_no_mem<8, VOLUME_SUM_OP<3, T>, T, Volume<3,T>>(
      volume.elements(), &sum, volume
    );
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::X>, float, Volume<3,T>>(
      runner, volume.elements(), &out_center_of_gravity[0], volume);
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::Y>, float, Volume<3, T>>(
      runner, volume.elements(), &out_center_of_gravity[1], volume);
  } | [&](){
    return reduce_no_mem<8, CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<T, DIMENSION::Z>, float, Volume<3, T>>(
      runner, volume.elements(), &out_center_of_gravity[2], volume);
  };

  f32 sumf = static_cast<f32>(sum);

  out_center_of_gravity[0] = out_center_of_gravity[0] / sumf;
  out_center_of_gravity[1] = out_center_of_gravity[1] / sumf;
  out_center_of_gravity[2] = out_center_of_gravity[2] / sumf;

  return runner.error();
}

}
