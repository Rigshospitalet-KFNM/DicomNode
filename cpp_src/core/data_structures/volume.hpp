//
// Created by christoffer on 9/2/26.
//
#pragma once
#include "../declarations.hpp"
#include "extent.hpp"

template<typename T, u8 DIMENSIONS>
struct Volume {
  T* data;
  Extent<DIMENSIONS> extent;

  constexpr u64 elements() const noexcept {
    return extent.elements();
  }

  constexpr T& at(u64 index) noexcept {
    return data[index];
  }

  constexpr const T& at(u64 index) const noexcept {
    return data[index];
  }
};