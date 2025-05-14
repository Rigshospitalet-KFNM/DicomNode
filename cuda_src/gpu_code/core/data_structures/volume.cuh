/**
 * @file volume.cuh
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2025-02-27
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include<stdint.h>
#include<vector>

#include"../concepts.cuh"
#include"extent.cuh"

template<uint8_t DIMENSIONS, typename T>
struct Volume {
  T* data = nullptr;
  Extent<DIMENSIONS> m_extent;
  T default_value;

  size_t elements() const {
    return m_extent.elements();
  }

  bool is_allocated() const {
    return data == nullptr;
  }

  dicomNodeError_t set_extent(const std::vector<ssize_t>& dims){
    return m_extent.set_dimensions(dims);
  }

  const Extent<DIMENSIONS>& extent() const {
    return m_extent;
  }
};

//static_assert(CVolume<Volume, 3, float>, "Volume doesn't fulfill volume concepts");