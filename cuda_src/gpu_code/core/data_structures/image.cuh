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

#include<stdint.h>

#include"../concepts.cuh"
#include"space.cuh"
#include"volume.cuh"


//template<template<uint8_t DIMENSIONS, typename T> typename VOLUME, uint8_t DIMENSIONS, typename T>
//  requires CVolume<VOLUME, DIMENSIONS, T>
template<uint8_t DIMENSIONS,typename T>
class Image {
  static_assert(0 < DIMENSIONS);

  public:
    Space<DIMENSIONS> space;
    Volume<DIMENSIONS, T> volume;

  Image(){}
  Image(Space<DIMENSIONS> a_space, Volume<DIMENSIONS, T> v_volume):
    space(std::move(a_space)),
    volume(std::move(v_volume))
  {}

  //T operator()(Point<DIMENSIONS> point){}

  constexpr const uint32_t& num_cols() const {
    return space.extent.sizes[0];
  }

  constexpr const uint32_t& num_rows () const {
    static_assert(1 < DIMENSIONS);
    return space.extent.sizes[1];
  }

  constexpr const uint32_t& num_slices() const {
    static_assert(2 < DIMENSIONS);
    return space.extent.sizes[2];
  }

  size_t elements() const {
    return volume.elements();
  }
};
