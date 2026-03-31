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
class Image {
  static_assert(0 < DIMENSIONS);

  public:
    Space<DIMENSIONS> space;
    Volume<DIMENSIONS, T> volume;

  __device__ __host__ Image(){}
  __device__ __host__ Image(Space<DIMENSIONS> a_space, Volume<DIMENSIONS, T> v_volume):
    space(std::move(a_space)),
    volume(std::move(v_volume))
  {}

  //T operator()(Point<DIMENSIONS> point){}
  constexpr __device__ __host__ T*& data() noexcept {
    return volume.data;
  }

  constexpr __device__ __host__ const T* data() const noexcept {
    return volume.data;
  }

  constexpr const u32& num_cols() const {
    return space.extent.sizes[0];
  }

  constexpr const u32& num_rows () const {
    static_assert(1 < DIMENSIONS);
    return space.extent.sizes[1];
  }

  constexpr const u32& num_slices() const {
    static_assert(2 < DIMENSIONS);
    return space.extent.sizes[2];
  }

  constexpr const Extent<DIMENSIONS>& extent() const {
    return space.extent;
  }



  /** Returns the amount of bytes contained in the volume
   *
   * @return
   */
  constexpr size_t size() const {
    return volume.size();
  }

  /** Returns the number of Elements stored in the image
   *
   * @return
   */
  constexpr __device__ __host__ size_t elements() const {
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
        new_extent,
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
