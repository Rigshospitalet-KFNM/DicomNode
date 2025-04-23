#pragma once

#include<stdint.h>
#include<vector>
#include<array>

#include<cuda/std/optional>

#include"../declarations.cuh"
#include"../concepts.cuh"

//namespace dicomnode {


/**
 * @brief Struct containing the size information or extent of a volume, image or
 * texture.
 *  Data is stored in Z,Y,X
 */
template<uint8_t DIMENSIONS>
struct Extent {
  static_assert(DIMENSIONS > 0);
  uint32_t sizes[DIMENSIONS]{}; // this zero initializes the array

  // Default constructor is need because otherwise the next constructor is used
  // which fails as no arguments fails static assert
  Extent(){}

  template<typename... Args>
  __device__ __host__ Extent(Args... args) noexcept : sizes{static_cast<uint32_t>(args)...}{
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  __device__ __host__ uint32_t& operator[](const uint8_t i) {
    return sizes[i];
  }

  __device__ __host__ const uint32_t& operator[](const uint8_t i) const {
    return sizes[i];
  }

  /**
   * @brief Checks in the index inside of the extent
   *
   * @param index
   * @return __device__
   */
  __device__ __host__ inline bool contains(const Index<DIMENSIONS> index) const {
    bool in = true;
    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      in &= 0 <= index[i] && index[i] < sizes[DIMENSIONS - (i + 1)];
    }

    return in;
  }

  template<typename... Args>
  __device__ __host__ inline bool contains(const Args... args) const {
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

  /**
   * @brief Returns the number of elements that is covered by this extent
   *
   * @return __device__ the number of elements
   */
  __device__  __host__ size_t elements() const noexcept {
    size_t size = 1;

    #pragma unroll
    for(uint8_t i = 0; i < DIMENSIONS; i++){
      size *= sizes[i];
    }

    return size;
  }

  constexpr uint8_t dimensionality() const {
    return DIMENSIONS;
  }

  __host__ dicomNodeError_t set_dimensions(const std::vector<ssize_t>& dims){
    if(dims.size() != DIMENSIONS){
      return INPUT_SIZE_MISMATCH;
    }

    for(int i = 0; const ssize_t& dim : dims){
      if(dim <= 0){
        return dicomNodeError_t::NON_POSITIVE_SHAPE;
      }
      sizes[i] = dim;
      i++;
    }

    return dicomNodeError_t::SUCCESS;
  }
};

//static_assert(CExtent<Extent, 3>); // I think there's some template magic wrong here

//}