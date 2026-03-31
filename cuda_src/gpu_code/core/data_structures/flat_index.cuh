#pragma once

#include<cuda/std/limits>

struct FlatIndex {
  u32 index = cuda::std::numeric_limits<u32>::max();

  constexpr __device__ __host__ bool has_value() const noexcept {
    return index != cuda::std::numeric_limits<u32>::max();
  }

  constexpr __device__ __host__ operator u32&() noexcept {
    return index;
  }

  constexpr __device__ __host__ operator const u32&() const noexcept {
    return index;
  }

  constexpr __device__ __host__ const u32& value() const noexcept {
    return index;
  }
};