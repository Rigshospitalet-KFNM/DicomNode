#pragma once

#include"../declarations.cuh"

#include<cuda/std/utility>

/**
 * @brief A point in a N-dimensional Space
 *
 * @tparam DIMENSIONS - The number of dimensions in the point.
 */
template<uint8_t DIMENSIONS>
struct Point {
  float points[DIMENSIONS]{};

  Point() noexcept = default;

  template<typename T, size_t... idx_seq>
  constexpr __device__ __host__ Point(
    const T& arr, cuda::std::index_sequence<idx_seq...>
  ) noexcept : points{static_cast<float>(arr[idx_seq])...} {}

  constexpr __device__ __host__ Point(Index<DIMENSIONS> idx) noexcept
    : Point(idx.coordinates, cuda::std::make_index_sequence<DIMENSIONS>{}) {}

  template<typename... Args>
  constexpr __device__ __host__ Point(Args... args) noexcept
    : points{static_cast<float>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  }

  template<typename T>
  constexpr __device__ __host__ f32& operator[](const T i) noexcept {
    return points[i];
  }

  template<typename T>
  constexpr __device__ __host__ volatile f32& operator[](const T i) volatile {
    return points[i];
  }

  template<typename T>
  constexpr __device__ __host__ const f32& operator[](const T i) const noexcept {
    return points[i];
  }

  constexpr __device__ __host__ bool operator==(const Point& other) const noexcept {
    #pragma unroll
    for (u8 i = 0; i < DIMENSIONS; i++) {
      if (this->points[i] != other[i]) {
        return false;
      }
    }

    return true;
  }


  constexpr __device__ __host__ Point operator*(const SquareMatrix<DIMENSIONS>& m) const noexcept {
    Point v; // It's zero initialized!
    #pragma unroll
    for(u8 j = 0; j < DIMENSIONS; j++){
      #pragma unroll
      for(u8 i = 0; i < DIMENSIONS; i++){
        v[j] += points[i] * m[m.idx(j, i)];
      }
    }

    return v;
  }

  constexpr __device__ __host__ Point operator-(const Point& other) const noexcept {
    Point v; // It's zero initialized!
    #pragma unroll
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] - other[i];
    }

    return v;
  }

  __device__ __host__ Point<DIMENSIONS> operator+(const Point<DIMENSIONS>& other) const noexcept {
    Point<DIMENSIONS> v; // It's zero initialized!
    #pragma unroll
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] + other[i];
    }

    return v;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS;
  }
};
