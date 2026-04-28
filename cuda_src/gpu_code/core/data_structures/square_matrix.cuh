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

  constexpr __device__ __host__ SquareMatrix<DIMENSIONS>& operator*=(
    const SquareMatrix<DIMENSIONS>& other) noexcept {
    SquareMatrix<DIMENSIONS> res = this->operator*(other);
    *this=res;
    return *this;
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

  constexpr __host__ __device__ SquareMatrix<DIMENSIONS> inverse() const noexcept {
    // Augmented matrix [A | I]
    f32 aug[DIMENSIONS * DIMENSIONS * 2]{};

    // Initialize augmented matrix
    #pragma unroll
    for (u8 i = 0; i < DIMENSIONS; i++) {
        #pragma unroll
        for (u8 j = 0; j < DIMENSIONS; j++) {
            aug[i * (DIMENSIONS * 2) + j] = points[idx(i, j)];
            aug[i * (DIMENSIONS * 2) + (DIMENSIONS + j)] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Gauss-Jordan elimination
    #pragma unroll
    for (u8 col = 0; col < DIMENSIONS; col++) {

        // Find pivot row (partial pivoting)
        u8 pivotRow = col;
        f32 maxVal = (aug[col * (DIMENSIONS * 2) + col] < 0)
            ? -aug[col * (DIMENSIONS * 2) + col]
            :  aug[col * (DIMENSIONS * 2) + col];

        #pragma unroll
        for (u8 row = col + 1; row < DIMENSIONS; row++) {
            f32 val = (aug[row * (DIMENSIONS * 2) + col] < 0)
                ? -aug[row * (DIMENSIONS * 2) + col]
                :  aug[row * (DIMENSIONS * 2) + col];
            if (val > maxVal) {
                maxVal = val;
                pivotRow = row;
            }
        }

        // Swap rows
        if (pivotRow != col) {
            #pragma unroll
            for (u8 j = 0; j < DIMENSIONS * 2; j++) {
                f32 tmp = aug[col * (DIMENSIONS * 2) + j];
                aug[col * (DIMENSIONS * 2) + j] = aug[pivotRow * (DIMENSIONS * 2) + j];
                aug[pivotRow * (DIMENSIONS * 2) + j] = tmp;
            }
        }

        // Scale pivot row
        f32 pivotVal = aug[col * (DIMENSIONS * 2) + col];
        f32 invPivot = 1.0f / pivotVal;
        #pragma unroll
        for (u8 j = 0; j < DIMENSIONS * 2; j++) {
            aug[col * (DIMENSIONS * 2) + j] *= invPivot;
        }

        // Eliminate column entries in all other rows
        #pragma unroll
        for (u8 row = 0; row < DIMENSIONS; row++) {
            if (row == col) continue;
            f32 factor = aug[row * (DIMENSIONS * 2) + col];
            #pragma unroll
            for (u8 j = 0; j < DIMENSIONS * 2; j++) {
                aug[row * (DIMENSIONS * 2) + j] -= factor * aug[col * (DIMENSIONS * 2) + j];
            }
        }
    }

    // Extract right half as inverse
    SquareMatrix<DIMENSIONS> result;
    #pragma unroll
    for (u8 i = 0; i < DIMENSIONS; i++) {
        #pragma unroll
        for (u8 j = 0; j < DIMENSIONS; j++) {
            result[idx(i, j)] = aug[i * (DIMENSIONS * 2) + (DIMENSIONS + j)];
        }
    }

    return result;
  }
};