#pragma once

#include<stdio.h>

#include<assert.h>
#include<stdint.h>
#include"indexing.cuh"

template<uint32_t DIMENSIONS>
struct Point {
  float points[DIMENSIONS];

  __device__ Point<DIMENSIONS>& operator-=(const Point<DIMENSIONS>& other) volatile {
    points[threadIdx.x] -= other[threadIdx.x];

    return *this;
  }

  __host__ __device__ float& operator[](const uint32_t i){
    return points[i];
  }

  __host__ __device__ float& operator[](const int32_t i){
    return points[i];
  }

  __device__ volatile float& operator[](const uint32_t i) volatile {
    return points[i];
  }

  __device__ volatile float& operator[](const int32_t i) volatile {
    return points[i];
  }

  __device__ void to_shared_memory(volatile Point<DIMENSIONS>* other) const {
    if(threadIdx.x < DIMENSIONS){
      other->point[threadIdx.x] = points[threadIdx.x];
    }
    __syncthreads();
  }

  Index<DIMENSIONS> lower_bound() const {
    int tmp[DIMENSIONS];
    #pragma unroll
    for(int i=0; i<DIMENSIONS;i++){
      tmp[i] =__float2int_rd(points[i]);
    }

    return Index(tmp);
  }
};

template<uint32_t DIMENSIONS>
struct SquareMatrix {
  float points[DIMENSIONS * DIMENSIONS];

  static __device__ __host__ uint32_t idx(const int32_t row, const int32_t col){
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

  __device__ void to_shared_memory(volatile SquareMatrix<DIMENSIONS>* other) const {
    if(threadIdx.x < DIMENSIONS*DIMENSIONS){
      other->point[threadIdx.x] = points[threadIdx.x];
    }
  }
};

template<uint8_t DIMENSION>
__device__ void swapRow(volatile SquareMatrix<DIMENSION>* matrix_,
                        const uint8_t r1, const uint8_t r2){
  volatile SquareMatrix<DIMENSION>& matrix = *matrix_;
  const uint32_t matrixRow = threadIdx.x / DIMENSION;
  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const bool active = threadIdx.x < DIMENSION*DIMENSION;
  const float tmp = active ? matrix[threadIdx.x] : 0.0f;
  __syncthreads();
  if(matrixRow == r1){
    matrix[r2 * DIMENSION + matrixCol] = tmp;
  }
  if(matrixRow == r2){
    matrix[r1 * DIMENSION + matrixCol] = tmp;
  }
  __syncthreads();
}

template<uint8_t DIMENSION>
__device__ void swapRow(volatile SquareMatrix<DIMENSION>& matrix,
                        const uint8_t r1, const uint8_t r2){
  const uint32_t matrixRow = threadIdx.x / DIMENSION;
  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const bool active = threadIdx.x < DIMENSION*DIMENSION;
  const float tmp = active ? matrix[threadIdx.x] : 0.0f;
  __syncthreads();
  if(matrixRow == r1){
    matrix[r2 * DIMENSION + matrixCol] = tmp;
  }
  if(matrixRow == r2){
    matrix[r1 * DIMENSION + matrixCol] = tmp;
  }
  __syncthreads();
}


template<uint8_t DIMENSION>
__device__ void swapVector(volatile Point<DIMENSION>* vector,
                        const uint8_t r1, const uint8_t r2){
  const float tmp = threadIdx.x == r1 || threadIdx.x == r2
    ? vector->points[threadIdx.x] : 0.0f;
  __syncthreads();
  if(threadIdx.x == r1){
    vector->points[r2] = tmp;
  }
  if(threadIdx.x == r2){
    vector->points[r1] = tmp;
  }
  __syncthreads();
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t ForwardElemination(
   volatile SquareMatrix<DIMENSION>* matrix,
   volatile Point<DIMENSION>* vector
){
  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const uint32_t matrixRow = threadIdx.x / DIMENSION;
  const bool vectorThread = threadIdx.x < DIMENSION;
  const bool activeThread = threadIdx.x < (DIMENSION * DIMENSION);

  // Forward elemination
  #pragma unroll
  for(uint8_t i=0; i < DIMENSION - 1; i++){
    float pivot = matrix->points[DIMENSION * i + i];
    if(pivot == 0.0f){
      uint8_t j = i + 1;
      float newPivotRow = matrix->points[DIMENSION * j + i];
      while(newPivotRow == 0.0f){
        j++;
        if(j == DIMENSION){
          return dicomNodeError_t::NotLinearIndependant;
        }
        newPivotRow = matrix->points[DIMENSION * j + i];
      }
      __syncthreads();
      swapRow<DIMENSION>(matrix, i, j);
      swapVector<DIMENSION>(vector, i, j);
      pivot = matrix->points[DIMENSION * i + i];
    }
    const float row_value = vectorThread ? vector->points[i] :
                            activeThread ? matrix->points[DIMENSION * i + matrixCol] : 0.0f;
    const float ratio_value = vectorThread ? matrix->points[DIMENSION * threadIdx.x + i] :
                              (activeThread ? matrix->points[DIMENSION * matrixRow + i] : 0.0f);
    const float thread_value = vectorThread ? vector->points[threadIdx.x] :
                              (activeThread ? matrix->points[threadIdx.x] : 0.0f);
    const float ratio = ratio_value / pivot;
    __syncthreads();
    // Write back
    if(vectorThread && i < threadIdx.x){
      vector->points[threadIdx.x] = thread_value - row_value * ratio;
    } else if (activeThread && i < matrixRow) {
      matrix->points[threadIdx.x] = thread_value - row_value * ratio;
    }

    __syncthreads();
  }

  return dicomNodeError_t::success;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t BackwardsElemination(
  volatile SquareMatrix<DIMENSION>* matrix,
  volatile Point<DIMENSION>* point
){
  volatile SquareMatrix<DIMENSION>& mRef = *matrix;
  volatile Point<DIMENSION>& pRef = *point;

  const bool active = threadIdx.x < DIMENSION;

  #pragma unroll
  for(int8_t i=DIMENSION-1; 0 <= i; i--){
    const float pivot = mRef[DIMENSION * i + i];
    const float vector_pivot = pRef[i];
    const float matrix_value = active ? mRef[DIMENSION * threadIdx.x + i] : 0;
    const float ratio = matrix_value / pivot;
    const float vector_value = active ? pRef[threadIdx.x] : 0.0;

    __syncthreads();
    // write back
    if(threadIdx.x <= i){
      mRef[DIMENSION * threadIdx.x + i] = threadIdx.x == i ? 1 : 0;
      pRef[threadIdx.x] = threadIdx.x == i ? vector_value / pivot
        : vector_value - vector_pivot * ratio;
    }
    __syncthreads();
  }

  return dicomNodeError_t::success;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t GaussJordanElemination(
   volatile SquareMatrix<DIMENSION>* matrix,
   volatile Point<DIMENSION>* point
  ){
  static_assert(DIMENSION <= 32);
  dicomNodeError_t error;

  error = ForwardElemination<DIMENSION>(matrix, point);
  if(error){
    return error;
  }
  error = BackwardsElemination<DIMENSION>(matrix, point);

  return error;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t _invertMatrixForward(
  volatile SquareMatrix<DIMENSION>& matrix,
  volatile SquareMatrix<DIMENSION>& output
) {
  const bool active = threadIdx.x < 2 * DIMENSION * DIMENSION;
  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const uint32_t matrixRow = (threadIdx.x / DIMENSION) % DIMENSION;
  const bool outputThread = active && DIMENSION * DIMENSION <= threadIdx.x;
  const uint32_t otidx = threadIdx.x % (DIMENSION * DIMENSION);

  if(outputThread){
    output[otidx] = static_cast<float>(matrixCol == matrixRow); //
  }
  __syncthreads();

  // Forward elemination
  #pragma unroll
  for(uint8_t i=0; i < DIMENSION - 1; i++){
    float pivot = matrix[DIMENSION * i + i];
    if(pivot == 0.0f){
      uint8_t j = i + 1;
      float newPivotRow = matrix[DIMENSION * j + i];
      while(newPivotRow == 0.0f){
        j++;
        if(j == DIMENSION){
          return dicomNodeError_t::NotLinearIndependant;
        }
        newPivotRow = matrix[DIMENSION * j + i];
      }
      __syncthreads();
      swapRow<DIMENSION>(matrix, i, j);
      swapRow<DIMENSION>(output, i, j);
      pivot = matrix[DIMENSION * i + i];
    }

    const float ratio_value = matrix[DIMENSION * matrixRow + i];
    const float ratio = ratio_value / pivot;

    const float row_value = outputThread ? output[DIMENSION * i + matrixCol] :
                            active ? matrix[DIMENSION * i + matrixCol] : 0.0f;

    const float thread_value = outputThread ? output[otidx] :
                               active ? matrix[threadIdx.x] : 0.0f;


    __syncthreads();
    // Write back
    if(outputThread && i < matrixRow){
      output[otidx] = thread_value - row_value * ratio;
    } else if (active && i < matrixRow) {
      matrix[threadIdx.x] = thread_value - row_value * ratio;
    }
    __syncthreads();
  }

  return dicomNodeError_t::success;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t _invertMatrixBackwards(
  volatile SquareMatrix<DIMENSION>& matrix,
  volatile SquareMatrix<DIMENSION>& output
){

  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const uint32_t matrixRow = (threadIdx.x / DIMENSION);
  const bool active = threadIdx.x < DIMENSION * DIMENSION;

  #pragma unroll
  for(int8_t i= DIMENSION - 1; 0 <= i; i--){
    const float pivot = matrix[DIMENSION * i + i];
    const float ratio_value = active ? matrix[matrix.idx(matrixRow, i)] : 0.0f;
    const float ratio = ratio_value / pivot;

    const float matrix_value = active ? matrix[threadIdx.x] : 0.0f;
    const float output_value = active ? output[threadIdx.x] : 0.0f;
    const float output_row = output[DIMENSION * i + matrixCol];

    __syncthreads();
    // write back
    if(matrixRow == i){
      const float writeback_value = output_value / pivot;
      output[threadIdx.x] = writeback_value;
      if(matrixCol == i){
        matrix[threadIdx.x] = 1;
      }
    } else if(matrixRow < i){
      output[threadIdx.x] = output_value - ratio * output_row;
      if(matrixCol == i){
        matrix[threadIdx.x] = 0;
      }
    }
    __syncthreads();
  }

  return dicomNodeError_t::success;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t invertMatrix(
  volatile SquareMatrix<DIMENSION>& matrix,
  volatile SquareMatrix<DIMENSION>& output
){
  dicomNodeError_t error = _invertMatrixForward<DIMENSION>(matrix, output);

  if(error != dicomNodeError_t::success){
    return error;
  }
  return _invertMatrixBackwards<DIMENSION>(matrix, output);
}


template<uint8_t DIMENSIONS, typename T>
class LinearSpace {
  public:
    Point<DIMENSIONS> starting_point[DIMENSIONS];
    SquareMatrix<DIMENSIONS> basis;
    Domain<DIMENSIONS> space;
    T* data;
    T defaultValue = 0;

  __device__ dicomNodeError_t find_boundary_point(
    volatile SquareMatrix<DIMENSIONS>* shared_memory_matrix,
    volatile Point<DIMENSIONS>* point) const {
      basis.to_shared_memory(shared_memory_matrix);
      volatile Point<DIMENSIONS>& rPoint = *point;
      rPoint -= starting_point;
      dicomNodeError_t error = GaussJordanElemination(shared_memory_matrix, point);

      return error;
  }

};
