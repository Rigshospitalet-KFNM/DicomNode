#pragma once

#include<stdint.h>

template<uint8_t CHUNK = 1>
[[nodiscard]] inline dim3 get_grid(const size_t &entries, const uint16_t &threads) noexcept;

template<uint8_t CHUNK = 1>
[[nodiscard]] inline dim3 get_grid(const dim3 &kernel_dim, const dim3 &threadBlock) noexcept;