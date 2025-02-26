#pragma once

#include<algorithm>
#include<utility>

#include<stdio.h>
#include<assert.h>
#include<stdint.h>

#include"indexing.cuh"
#include"cuda_management.cuh"

template<uint32_t>
struct SquareMatrix;

/**
 * @brief A point in a N-dimensional Space
 *
 * @tparam DIMENSIONS - The number of
 */
template<uint32_t DIMENSIONS>
struct Point {
  float points[DIMENSIONS]{};

  Point<DIMENSIONS>() noexcept = default;


  template<typename T, size_t... idx_seq>
  __device__ __host__ Point<DIMENSIONS>(
    const T& arr, cuda::std::index_sequence<idx_seq...>
  ) noexcept : points{static_cast<float>(arr[idx_seq])...} {}

  __device__ __host__ Point<DIMENSIONS>(Index<DIMENSIONS> idx)
    : Point(idx.coordinates, cuda::std::make_index_sequence<DIMENSIONS>{}) {}

  template<typename... Args>
  __device__ __host__ Point<DIMENSIONS>(Args... args) noexcept
    : points{static_cast<float>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  }

  template<typename T>
  __device__ __host__ float& operator[](const T i){
    return points[i];
  }

  template<typename T>
  __device__ __host__ volatile float& operator[](const T i) volatile {
    return points[i];
  }

  template<typename T>
  __device__ __host__ const float& operator[](const T i) const {
    return points[i];
  }

  __device__ __host__ Point<DIMENSIONS> operator*(const SquareMatrix<DIMENSIONS>& m){
    Point<DIMENSIONS> v; // It's zero initialized!
    for(uint8_t j = 0; j < DIMENSIONS; j++){
      for(uint8_t i = 0; i < DIMENSIONS; i++){
        v[j] += points[i] * m[m.idx(j, i)];
      }
    }

    return v;
  }

  __device__ __host__ Point<DIMENSIONS> operator-(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] - other[i];
    }

    return v;
  }

  __device__ __host__ Point<DIMENSIONS> operator+(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] + other[i];
    }

    return v;
  }

  __device__ __host__ bool operator==(const Point<DIMENSIONS> other) const {
    bool ret = true;
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      ret = ret && points[i] == other[i];
    }
    return ret;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS;
  }
};

template<uint32_t DIMENSIONS>
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
__device__ dicomNodeError_t ForwardElimination(
   volatile SquareMatrix<DIMENSION>* matrix,
   volatile Point<DIMENSION>* vector
){
  const uint32_t matrixCol = threadIdx.x % DIMENSION;
  const uint32_t matrixRow = threadIdx.x / DIMENSION;
  const bool vectorThread = threadIdx.x < DIMENSION;
  const bool activeThread = threadIdx.x < (DIMENSION * DIMENSION);

  // Forward elimination
  #pragma unroll
  for(uint8_t i=0; i < DIMENSION - 1; i++){
    float pivot = matrix->points[DIMENSION * i + i];
    if(pivot == 0.0f){
      uint8_t j = i + 1;
      float newPivotRow = matrix->points[DIMENSION * j + i];
      while(newPivotRow == 0.0f){
        j++;
        if(j == DIMENSION){
          return dicomNodeError_t::NotLinearIndependent;
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

  return dicomNodeError_t::SUCCESS;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t BackwardsElimination(
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

  return dicomNodeError_t::SUCCESS;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t GaussJordanElimination(
   volatile SquareMatrix<DIMENSION>* matrix,
   volatile Point<DIMENSION>* point
  ){
  static_assert(DIMENSION <= 32);
  dicomNodeError_t error;

  error = ForwardElimination<DIMENSION>(matrix, point);
  if(error){
    return error;
  }
  error = BackwardsElimination<DIMENSION>(matrix, point);

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
  const uint32_t output_thread_idx = threadIdx.x % (DIMENSION * DIMENSION);

  if(outputThread){
    output[output_thread_idx] = static_cast<float>(matrixCol == matrixRow); //
  }
  __syncthreads();

  // Forward elimination
  #pragma unroll
  for(uint8_t i=0; i < DIMENSION - 1; i++){
    float pivot = matrix[DIMENSION * i + i];
    if(pivot == 0.0f){
      uint8_t j = i + 1;
      float newPivotRow = matrix[DIMENSION * j + i];
      while(newPivotRow == 0.0f){
        j++;
        if(j == DIMENSION){
          return dicomNodeError_t::NotLinearIndependent;
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

    const float thread_value = outputThread ? output[output_thread_idx] :
                               active ? matrix[threadIdx.x] : 0.0f;


    __syncthreads();
    // Write back
    if(outputThread && i < matrixRow){
      output[output_thread_idx] = thread_value - row_value * ratio;
    } else if (active && i < matrixRow) {
      matrix[threadIdx.x] = thread_value - row_value * ratio;
    }
    __syncthreads();
  }

  return dicomNodeError_t::SUCCESS;
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
      const float write_back_value = output_value / pivot;
      output[threadIdx.x] = write_back_value;
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

  return dicomNodeError_t::SUCCESS;
}

template<uint8_t DIMENSION>
__device__ dicomNodeError_t invertMatrix(
  volatile SquareMatrix<DIMENSION>& matrix,
  volatile SquareMatrix<DIMENSION>& output
){
  dicomNodeError_t error = _invertMatrixForward<DIMENSION>(matrix, output);

  if(error != dicomNodeError_t::SUCCESS){
    return error;
  }
  return _invertMatrixBackwards<DIMENSION>(matrix, output);
}

/**
 * @brief A space is description of a linear space of n Dimensions where the
 * starting point is the coordinate at Index (0,0,0, ...). The space extents out
 * to defined by the domain.
 *
 * @tparam DIMENSIONS
 */
template<uint8_t DIMENSIONS>
class Space {
  public:

    /**
     * @brief The coordinates for the Index (0,0,0,...)
     *
     */
    Point<DIMENSIONS> starting_point;
    SquareMatrix<DIMENSIONS> basis;
    SquareMatrix<DIMENSIONS> inverted_basis;
    Extent<DIMENSIONS> extent;

  __device__ __host__ Index<DIMENSIONS> index(const uint64_t& flat_index) const {
    return extent.from_flat_index(flat_index);
  }

  __device__ __host__ Point<DIMENSIONS> at_index(const Index<DIMENSIONS>& index) const {
    Point<DIMENSIONS> point{index};
    return point * basis + starting_point;
  }

  __device__ __host__ Point<DIMENSIONS> interpolate_point(const Point<DIMENSIONS>& point) const {
    return (point - starting_point) * inverted_basis;
  }
};

template<uint8_t DIMENSIONS, typename T>
class Image {
  public:
    Space<DIMENSIONS> space;
    T* data = nullptr;
    T defaultValue = 0;

  constexpr const uint32_t& num_cols() const {
    static_assert(0 < DIMENSIONS);
    return space.extent.sizes[0];
  }

  constexpr const uint32_t& num_rows () const {
    static_assert(1 < DIMENSIONS);
    return space.extent.sizes[1];
  }

  constexpr const uint32_t& num_slices() const {
    static_assert(1 < DIMENSIONS);
    return space.extent.sizes[2];
  }
};
