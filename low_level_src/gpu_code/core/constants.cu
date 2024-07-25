#pragma once
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

#include<stdint.h>

constexpr uint8_t WARP_SIZE = 32;
constexpr uint8_t LOG_WARP = 5;
constexpr uint16_t SCAN_BLOCK_SIZE = 1024;

constexpr int ARRAY_FLAGS = pybind11::array_t<int>::forcecast | pybind11::array_t<int>::c_style;
