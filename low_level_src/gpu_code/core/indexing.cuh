#pragma once

// Standard library
#include<stdint.h>
#include<cuda/std/optional>

template<uint8_t>
struct Domain;

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

  __device__ __host__ Index(const uint64_t flat_index, const Domain<DIMENSIONS> space){
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

/**
 * @brief Struct containing the size information of an image
 *  Data is stored in Z,Y,X
 */
template<uint8_t DIMENSIONS>
struct Domain {
  static_assert(DIMENSIONS > 0);
  uint32_t sizes[DIMENSIONS]{};

  __device__ __host__ Domain(){
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      sizes[i] = 0;
    }
  }

  template<typename... Args>
  __device__ __host__ Domain(Args... args) noexcept : sizes{static_cast<uint32_t>(args)...}{
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  __device__ __host__ uint32_t& operator[](const uint8_t i) {
    return sizes[i];
  }

  __device__ __host__ const uint32_t& operator[](const uint8_t i) const {
    return sizes[i];
  }

  __device__ __host__ inline bool contains(const Index<DIMENSIONS> index) const {
    bool in = true;
    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      in &= 0 <= index[i] && index[i] < sizes[DIMENSIONS - (i + 1)];
    }

    return in;
  }

  template<typename... Args>
  __device__ __host__ bool contains(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    return contains(Index({static_cast<int32_t>(args)...}));
  }

  __device__ __host__ cuda::std::optional<uint64_t> flat_index(const Index<DIMENSIONS> index) const {
    if(!contains(index)){
      return {};
    }

    uint64_t return_val = 0;
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      return_val += index[i] * dimension_temp;
      dimension_temp *= sizes[DIMENSIONS - (i + 1)];
    }

    return return_val;
  }

  template<typename... Args>
  __device__ __host__ cuda::std::optional<uint64_t> flat_index(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    Index index = Index<DIMENSIONS>(args...);

    return flat_index(index);
  }

  __device__ __host__ Index<DIMENSIONS> from_flat_index(uint64_t flat_index) const {
    int32_t coordinates[DIMENSIONS];
    uint64_t dimension_temp = 1;

    #pragma unroll
    for(uint8_t dim = 0; dim < DIMENSIONS; dim++){
      const uint32_t& extent = sizes[DIMENSIONS - (dim + 1)];
      coordinates[dim] = (flat_index % (dimension_temp * extent))
        / dimension_temp;
      dimension_temp *= extent;
    }

    return Index<DIMENSIONS>(coordinates);
  }

  __device__ __host__ const uint32_t& x() const noexcept {
    return sizes[2];
  }

  __device__ __host__ const uint32_t& y() const noexcept{
    static_assert(DIMENSIONS > 1);
    return sizes[1];
  }

  __device__ __host__ const uint32_t& z() const noexcept {
    static_assert(DIMENSIONS > 2);
    return sizes[0];
  }

  __device__ __host__ constexpr uint32_t* begin() noexcept{
    return &sizes[0];
  }

  __device__ __host__ constexpr uint32_t* end() noexcept{
    return &sizes[DIMENSIONS - 1];
  }

  __device__ __host__ constexpr const uint32_t* begin() const noexcept{
    return &sizes[0];
  }

  __device__ __host__ constexpr const uint32_t* end() const noexcept{
    return &sizes[DIMENSIONS - 1];
  }

  __device__  __host__ size_t size() const noexcept {
    size_t size = 1;

    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      size *= sizes[i];
    }

    return size;
  }

  static constexpr __host__ __device__ size_t elements() {
    return DIMENSIONS;
  }
};
