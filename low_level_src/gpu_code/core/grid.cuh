#pragma once
// C standard Library
#include<stdint.h>

/**
   * @brief Calculate the number of GPU kernel blocks, given a thread block
   * size and number of elements
   * @example dim3 blocks = get_grid(1000, 256)
   * kernel<<<blocks, 256>>>(args)
   *
   * @param entries - The Number of elements in the kernel
   * @param threads - The size of thread block
   * @return dim3 - The number of blocks to run in the kernel
   */
template<uint8_t CHUNK = 1>
[[nodiscard]] inline dim3 get_grid(const size_t &entries, const uint16_t &threads) noexcept {
  const size_t entiries_after_chunk = entries % CHUNK == 0 ? entries / CHUNK : entries / CHUNK + 1;

  return entiries_after_chunk % threads == 0 ? entiries_after_chunk / threads : entiries_after_chunk / threads + 1;
}

/**
   * @brief Get the blocks object
   *
   * @example dim3 blocks = get_grid({1920,1080,1}, {32,32,1})
   * kernel<<<blocks, {32,32,1}>>>(args)
   *
   * @param kernel_dim - Calculate the number of blocks, given a thread block size
   * @param threadBlock - The size of the kernel
   * @return dim3 - The number of blocks to run in the kernel
   */
template<uint8_t CHUNK = 1>
[[nodiscard]] inline dim3 get_grid(const dim3 &kernel_dim, const dim3 &threadBlock) noexcept {
  const uint32_t x_dim = kernel_dim.x % threadBlock.x == 0 ? kernel_dim.x / threadBlock.x : kernel_dim.x / threadBlock.x + 1;
  const uint32_t y_dim = kernel_dim.y % threadBlock.y == 0 ? kernel_dim.y / threadBlock.y : kernel_dim.y / threadBlock.y + 1;
  const uint32_t z_dim = kernel_dim.z % threadBlock.z == 0 ? kernel_dim.z / threadBlock.z : kernel_dim.z / threadBlock.z + 1;

  return dim3{x_dim, y_dim, z_dim};
}
