#pragma once

#include<array>
#include<optional>
#include<vector>

#include"../error.hpp"
#include"../declarations.hpp"

template<u8 DIMENSIONS>
struct Extent {
  using EXTENT_DATA_TYPE = u32;

  static_assert(DIMENSIONS > 0);

  std::array<EXTENT_DATA_TYPE, DIMENSIONS> sizes{};

  // Default constructor is need because otherwise the next constructor is used
  // which fails as no arguments fails static assert
  Extent(){}

  template<typename... Args>
  Extent(Args... args) noexcept : sizes{static_cast<u32>(args)...}{
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  u32& operator[](const u8 i) {
    return sizes[i];
  }

  constexpr const u32& operator[](const u8 i) const {
    return sizes[i];
  }

  /**
   * @brief Checks in the index inside of the extent
   *
   * @param index
   * @return __device__
   */
  inline bool contains(const Index<DIMENSIONS> index) const {
    bool in = true;

    for(u8 i = 0; i < DIMENSIONS; i++){
      in &= 0 <= index[i] && std::cmp_less_equal(index[i],sizes[DIMENSIONS - (i + 1)]);
    }

    return in;
  }

  template<typename... Args>
  inline bool contains(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    return contains(Index({static_cast<i32>(args)...}));
  }

  std::optional<u64> flat_index(const Index<DIMENSIONS> index) const {
    if(!contains(index)){
      return {};
    }

    u64 return_val = 0;
    u64 dimension_temp = 1;

    for(u8 i = 0; i < DIMENSIONS; i++){
      return_val += index[i] * dimension_temp;
      dimension_temp *= sizes[DIMENSIONS - (i + 1)];
    }

    return return_val;
  }

  template<typename... Args>
  std::optional<u64> flat_index(const Args... args) const {
    static_assert(sizeof...(args) == DIMENSIONS);
    Index index = Index<DIMENSIONS>(args...);

    return flat_index(index);
  }

  Index<DIMENSIONS> from_flat_index(u64 flat_index) const {
    i32 coordinates[DIMENSIONS];
    u64 dimension_temp = 1;

    for(u8 dim = 0; dim < DIMENSIONS; dim++){
      const u32& extent = sizes[DIMENSIONS - (dim + 1)];
      coordinates[dim] = (flat_index % (dimension_temp * extent))
        / dimension_temp;
      dimension_temp *= extent;
    }

    return Index<DIMENSIONS>(coordinates);
  }

  const u32& x() const noexcept {
    return sizes[DIMENSIONS - 1];
  }

  const u32& y() const noexcept{
    static_assert(DIMENSIONS > 1);
    return sizes[DIMENSIONS - 2];
  }

  const u32& z() const noexcept {
    static_assert(DIMENSIONS > 2);
    return sizes[DIMENSIONS - 3];
  }

  constexpr u32* begin() noexcept{
    return sizes.begin();
  }

  constexpr u32* end() noexcept{
    return sizes.end();
  }

  constexpr const u32* begin() const noexcept{
    return sizes.begin();
  }

  constexpr const u32* end() const noexcept{
    return sizes.end();
  }

  /**
   * @brief Returns the number of elements that is covered by this extent
   *
   * @return __device__ the number of elements
   */
  size_t elements() const noexcept {
    size_t size = 1;

    for(u8 i = 0; i < DIMENSIONS; i++){
      size *= sizes[i];
    }

    return size;
  }

  constexpr u8 dimensionality() const {
    return DIMENSIONS;
  }

  CppError_t set_dimensions(const std::vector<ssize_t>& dims){
    if(dims.size() != DIMENSIONS){
      return INPUT_SIZE_MISMATCH;
    }

    for(u8 i = 0; const ssize_t& dim : dims){
      if(dim <= 0){
        return CppError_t::NON_POSITIVE_SHAPE;
      }
      sizes[i] = dim;
      i++;
    }

    return CppError_t::SUCCESS;
  }

  template<u8 ARRAY_SIZE>
  CppError_t set_dimensions(const std::array<ssize_t, ARRAY_SIZE>& dims){
    static_assert(ARRAY_SIZE == DIMENSIONS);

    for(u8 i = 0; const ssize_t& dim : dims){
      if(dim <= 0){
        return CppError_t::NON_POSITIVE_SHAPE;
      }
      sizes[i] = dim;
      i++;
    }
  }

  std::array<size_t, DIMENSIONS> python_strides(size_t pixel_size) const{
    std::array<size_t, DIMENSIONS> out;

    for(u32 i = 1; i <= DIMENSIONS; i++){
      u32 inverted_index = DIMENSIONS - i;
      out[inverted_index] = pixel_size;
      pixel_size *= sizes[inverted_index];
    }

    return out;
  }
};