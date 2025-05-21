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
  __device__ __host__ f32& operator[](const T i){
    return points[i];
  }

  template<typename T>
  __device__ __host__ volatile f32& operator[](const T i) volatile {
    return points[i];
  }

  template<typename T>
  __device__ __host__ const f32& operator[](const T i) const {
    return points[i];
  }

  __device__ __host__ Point<DIMENSIONS> operator*(const SquareMatrix<DIMENSIONS>& m){
    Point<DIMENSIONS> v; // It's zero initialized!
    #pragma unroll
    for(u8 j = 0; j < DIMENSIONS; j++){
      #pragma unroll
      for(u8 i = 0; i < DIMENSIONS; i++){
        v[j] += points[i] * m[m.idx(j, i)];
      }
    }

    return v;
  }

  __device__ __host__ Point<DIMENSIONS> operator-(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    #pragma unroll
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] - other[i];
    }

    return v;
  }

  __device__ __host__ Point<DIMENSIONS> operator+(const Point<DIMENSIONS>& other) const {
    Point<DIMENSIONS> v; // It's zero initialized!
    for(u8 i = 0; i < DIMENSIONS; i++){
      v[i] = points[i] + other[i];
    }

    return v;
  }

  __device__ __host__ bool operator==(const Point<DIMENSIONS> other) const {
    bool ret = true;
    for(u8 i = 0; i < DIMENSIONS; i++){
      ret = ret && points[i] == other[i];
    }
    return ret;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS;
  }
};
