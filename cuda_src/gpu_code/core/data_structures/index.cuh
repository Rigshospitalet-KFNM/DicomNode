#pragma once

#include"../declarations.cuh"

/**
 * @brief An index in X,Y,Z,... coordinates
 *
 * @tparam DIMENSIONS - Number of Dimension of the index
 */
template<uint8_t DIMENSIONS>
struct Index {
  static_assert(DIMENSIONS > 0);
  // Negative number would indicate and out of image index
  int32_t coordinates[DIMENSIONS]{};

  Index() noexcept = default;

  template<typename T, size_t... idx_seq>
  __device__ __host__ Index(const T& arr, cuda::std::index_sequence<idx_seq...>)
    noexcept : coordinates{static_cast<int32_t>(arr[idx_seq])...} { }

  template<typename T>
  __device__ __host__ Index(const T (&arr)[DIMENSIONS]) noexcept :
    Index(arr, cuda::std::make_index_sequence<DIMENSIONS>()) {}

  template<typename... Args>
  __device__ __host__ Index(const Args... args) noexcept
    : coordinates{static_cast<int32_t>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  __device__ __host__ Index(const uint64_t flat_index, const Extent<DIMENSIONS> space){
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = (flat_index % (dimension_temp * space[dim]))
        / dimension_temp;
      dimension_temp *= space[dim];
    }
  }

  __device__ __host__  int32_t& operator[](const uint8_t idx){
    // you can't put a static assert in here :(
    return coordinates[idx];
  }

  __device__ __host__ int32_t operator[](const uint8_t idx) const {
    return coordinates[idx];
  }

  __device__ __host__ const int32_t& x() const {
    return coordinates[0];
  }

  __device__ __host__ const int32_t& y() const {
    static_assert(DIMENSIONS > 1);
    return coordinates[1];
  }

  __device__ __host__ const int32_t& z() const {
    static_assert(DIMENSIONS > 2);
    return coordinates[2];
  }
};
