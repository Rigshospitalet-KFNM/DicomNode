#pragma once
#include<stdint.h>

inline constexpr uint8_t WARP_SIZE = 32;
inline constexpr uint8_t LOG_WARP = 5;
inline constexpr uint16_t SCAN_BLOCK_SIZE = 1024;
inline constexpr uint16_t MAP_BLOCK_SIZE = 1024;
inline constexpr uint16_t MIRROR_BLOCK_SIZE = 1024;

inline constexpr ssize_t SSIZE_T_ERROR_VALUE = -1;