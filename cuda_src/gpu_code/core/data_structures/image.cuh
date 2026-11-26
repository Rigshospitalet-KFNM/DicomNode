/**
 * @file image.cuh
 * @author Demiguard
 * @brief Defines the Image class which represent a linear space with associated
 * data.
 * @version 0.1
 * @date 2025-04-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once


#include"../declarations.cuh"
#include"space.cuh"
#include"volume.cuh"


//template<template<uint8_t DIMENSIONS, typename T> typename VOLUME, uint8_t DIMENSIONS, typename T>
//  requires CVolume<VOLUME, DIMENSIONS, T>
template<u8 DIMENSIONS,typename T>
struct Image final {
  static_assert(0 < DIMENSIONS);

  Space<DIMENSIONS> space;
  Volume<DIMENSIONS, T> volume;

  //T operator()(Point<DIMENSIONS> point){}
  [[nodiscard]] constexpr __device__ __host__ T*& data() noexcept {
    return volume.data;
  }

  [[nodiscard]] constexpr __device__ __host__ const T* data() const noexcept {
    return volume.data;
  }

  [[nodiscard]] constexpr __device__ __host__ const u32& num_cols() const noexcept {
    return volume.extent().sizes[0];
  }

  [[nodiscard]] constexpr __device__ __host__ const u32& num_rows () const noexcept {
    static_assert(1 < DIMENSIONS);
    return volume.extent().sizes[1];
  }

  [[nodiscard]] constexpr __device__ __host__ const u32& num_slices() const noexcept {
    static_assert(2 < DIMENSIONS);
    return volume.extent().sizes[2];
  }

  [[nodiscard]] constexpr __device__ __host__ const Extent<DIMENSIONS>& extent() const {
    return volume.extent();
  }

  [[nodiscard]] constexpr __host__ __device__  Point<DIMENSIONS> at(const Point<DIMENSIONS>& p) const noexcept  {
    return p * space.basis + space.starting_point;
  }


  /** Returns the amount of bytes contained in the volume
   *
   * @return
   */
  [[nodiscard]] constexpr __device__ __host__ size_t size() const noexcept {
    return volume.size();
  }

  /** Returns the number of Elements stored in the image
   *
   * @return
   */
  [[nodiscard]] constexpr __device__ __host__ size_t elements() const noexcept {
    return volume.elements();
  }


};

template<typename T>
__device__ __host__ Image<3, T> sub_image(
    const Image<3, T>& original,
    T* new_data_ptr,
    Extent<3> new_extent,
    Index<3> offset_index
  ) {
    return Image<3, T>(
      offset_space(
        original.space,
        offset_index
    ),
    sub_volume(
        original.volume,
        new_data_ptr,
        new_extent,
        offset_index
      )
    );
}

Image() -> Image<3, f32>;

template<typename T>
Image(T) -> Image<3, T>;
