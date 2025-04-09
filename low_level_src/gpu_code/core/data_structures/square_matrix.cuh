#pragma once

#include"../declarations.cuh"


template<uint8_t DIMENSIONS>
struct SquareMatrix {
  float points[DIMENSIONS * DIMENSIONS]{};

  static constexpr __device__ __host__ uint32_t idx(const int32_t row, const int32_t col){
    return row * DIMENSIONS + col;
  }

  __host__ __device__ float& operator[](const int32_t i){
    return points[i];
  }

  __host__ __device__ float& operator[](const uint32_t i){
    return points[i];
  }

  __device__ volatile float& operator[](const int32_t i) volatile {
    return points[i];
  }

  __device__ volatile float& operator[](const uint32_t i) volatile {
    return points[i];
  }

  __host__ __device__ const float& operator[](const int32_t i) const {
    return points[i];
  }

  __host__ __device__ const float& operator[](const uint32_t i) const {
    return points[i];
  }

  __host__ __device__ const Point<DIMENSIONS> operator*(
    const Point<DIMENSIONS>& other) const {
      // It's zero initialized!
      Point<DIMENSIONS> point;
      for(uint8_t j = 0; j < DIMENSIONS; j++){
        for(uint8_t i = 0; i < DIMENSIONS; i++){
          point[j] += other[i] * points[idx(i,j)];
        }
      }

    return point;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS * DIMENSIONS;
  }
};