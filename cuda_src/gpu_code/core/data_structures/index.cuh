#pragma once

#include<cuda/std/utility> // make index_sequence

#include"../declarations.cuh"

/**
 * @brief An index in X,Y,Z,... coordinates
 *
 * @tparam DIMENSIONS - Number of Dimension of the index
 */
template<u8 DIMENSIONS>
struct Index {
  static_assert(DIMENSIONS > 0);
  // Negative number would indicate and out of image index
  i32 coordinates[DIMENSIONS]{};

  Index() noexcept = default;

  template<typename T, size_t... idx_seq>
  __device__ __host__ Index(const T& arr, cuda::std::index_sequence<idx_seq...>)
    noexcept : coordinates{static_cast<i32>(arr[idx_seq])...} { }

  template<typename T>
  __device__ __host__ Index(const T (&arr)[DIMENSIONS]) noexcept :
    Index(arr, cuda::std::make_index_sequence<DIMENSIONS>()) {}

  template<typename... Args>
  __device__ __host__ Index(const Args... args) noexcept
    : coordinates{static_cast<i32>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  __device__ __host__ Index(const u64& flat_index, const Extent<DIMENSIONS>& space){
    u64 dimension_temp = 1;

    #pragma unroll
    for(u8 dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = (flat_index % (dimension_temp * space[dim]))
        / dimension_temp;
      dimension_temp *= space[dim];
    }
  }

  __device__ __host__  i32& operator[](const u8 idx){
    // you can't put a static assert in here :(
    return coordinates[idx];
  }

  __device__ __host__ i32 operator[](const u8 idx) const {
    return coordinates[idx];
  }

  __device__ __host__ const i32& x() const {
    return coordinates[0];
  }

  __device__ __host__ const i32& y() const {
    static_assert(DIMENSIONS > 1);
    return coordinates[1];
  }

  __device__ __host__ const i32& z() const {
    static_assert(DIMENSIONS > 2);
    return coordinates[2];
  }
};
