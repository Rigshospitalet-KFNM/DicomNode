#pragma once

#include<array>
#include<utility>

#include"../declarations.hpp"


/**
 * @brief An index in X,Y,Z,... coordinates
 *
 * @tparam DIMENSIONS - Number of Dimension of the index
 */
template<u8 DIMENSIONS>
struct Index {
  static_assert(DIMENSIONS > 0);
  // Negative number would indicate and out of image index
  std::array<i32, DIMENSIONS> coordinates{};

  Index() noexcept = default;

  template<typename T, size_t... idx_seq>
  Index(const T& arr, std::index_sequence<idx_seq...>)
    noexcept : coordinates{static_cast<i32>(arr[idx_seq])...} { }

  template<typename T>
  Index(const T (&arr)[DIMENSIONS]) noexcept :
    Index(arr, std::make_index_sequence<DIMENSIONS>()) {}

  template<typename... Args>
  Index(const Args... args) noexcept
    : coordinates{static_cast<i32>(args)...} {
    static_assert(sizeof...(args) == DIMENSIONS);
  };

  Index(const u64 flat_index, const Extent<DIMENSIONS> space){
    uint64_t dimension_temp = 1;

    for(u8 dim = 0; dim < DIMENSIONS; dim++){
      coordinates[dim] = (flat_index % (dimension_temp * space[dim]))
        / dimension_temp;
      dimension_temp *= space[dim];
    }
  }

  i32& operator[](const uint8_t idx){
    // you can't put a static assert in here :(
    return coordinates[idx];
  }

  i32 operator[](const uint8_t idx) const {
    return coordinates[idx];
  }

  const i32& x() const {
    return coordinates[0];
  }

  const i32& y() const {
    static_assert(DIMENSIONS > 1);
    return coordinates[1];
  }

  i32& z() const {
    static_assert(DIMENSIONS > 2);
    return coordinates[2];
  }
};
