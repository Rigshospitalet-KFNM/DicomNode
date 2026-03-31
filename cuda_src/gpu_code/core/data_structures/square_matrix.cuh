#pragma once

#include"../declarations.cuh"


template<u8 DIMENSIONS>
struct SquareMatrix {
  float points[DIMENSIONS * DIMENSIONS]{};

  static constexpr __device__ __host__ uint32_t idx(const i32 row, const i32 col){
    return row * DIMENSIONS + col;
  }

  __host__ __device__ f32& operator[](const i32 i){
    return points[i];
  }

  __host__ __device__ f32& operator[](const u32 i){
    return points[i];
  }

  __device__ volatile f32& operator[](const i32 i) volatile {
    return points[i];
  }

  __device__ volatile f32& operator[](const u32 i) volatile {
    return points[i];
  }

  __host__ __device__ const f32& operator[](const i32 i) const noexcept {
    return points[i];
  }

  constexpr __host__ __device__ const f32& operator[](const u32 i) const noexcept {
    return points[i];
  }

  constexpr __host__ __device__ Point<DIMENSIONS> operator*(
    const Point<DIMENSIONS>& other
  ) const noexcept {
      // It's zero initialized!
      Point<DIMENSIONS> point;
      #pragma unroll
      for(u8 j = 0; j < DIMENSIONS; j++){
        #pragma unroll
        for(u8 i = 0; i < DIMENSIONS; i++){
          point[j] += other[i] * points[idx(i,j)];
        }
      }

    return point;
  }

  constexpr __host__ __device__ SquareMatrix<DIMENSIONS> operator*(
    const SquareMatrix<DIMENSIONS>& other) const noexcept {

    SquareMatrix<DIMENSIONS> res;

    #pragma unroll
    for (u8 i = 0; i < DIMENSIONS; i++) {
      #pragma unroll
      for (u8 j = 0; j < DIMENSIONS; j++) {
        res[idx(i,j)] = 0.0f;
        #pragma unroll
        for (u8 k = 0; k < DIMENSIONS; k++) {
          res[idx(i,j)] += points[idx(i,k)] * other[idx(k,j)];
        }
      }
    }

    return res;
  }

  constexpr __device__ __host__ bool operator==(const SquareMatrix& other) const {
    for(u8 j = 0; j < DIMENSIONS * DIMENSIONS; j++) {
      if(other[j] != points[j]) {
        return false;
      }
    }
    return true;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS * DIMENSIONS;
  }
};