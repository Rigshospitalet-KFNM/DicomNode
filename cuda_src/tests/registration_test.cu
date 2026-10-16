//
// Created by cjen0668 on 7/30/26.
//

#include "../gpu_code/core/core.cuh"
#include "../gpu_code/registration.cuh"
#include "test_helper.cuh"

#include <array>

#include <gtest/gtest.h>

#include "test_helper.cuh"


constexpr u64 x = 8;
constexpr u64 y = 8;
constexpr u64 z = 8;

constexpr u64 elements = x * y * z;
// This is just a 3x3x3 cube at 2 different indexes.
// I should create some functions for easy array creation, but i'm feeling that i'm missing c++23/c++26 features :(

static constexpr std::array<f32, elements> offset_image_data_1{
  // image 1
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 2
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 3
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 4
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 5
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 6
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 7
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 8
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
};

static constexpr std::array<f32, elements> offset_image_data_2{
  // image 1
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 2
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 3
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 4
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 5
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 6
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 7
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 8
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
};

static constexpr std::array<f32, elements> overlap_image_1 = {
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 2
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 3
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 4
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 5
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 6
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 7
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 8
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
};
static constexpr std::array<f32, elements> overlap_image_2 = {
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 2
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 3
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 4
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 5
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 6
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 7
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  // image 8
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
};



constexpr static Space<3> image_space {
  .starting_point = Point<3>{1.0f,1.0f, 1.0f},
  .basis = {
    1.0f,0.0f,0.0f,
    0.0f,1.0f,0.0f,
    0.0f,0.0f,1.0f
  },
  .inverted_basis = {
    1.0f,0.0f,0.0f,
    0.0f,1.0f,0.0f,
    0.0f,0.0f,1.0f
  },
  .extent = Extent<3>{z,y,x}
};

// End of test declaration
TEST(REGISTRATION, VOLUME_SUBTRACT_OF_SAME_IS_ZERO) {
  Volume<3, f32> volume {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  Volume<3, f32>* device_volume;

  f32 difference = -1.0f;

  cudaMalloc(&device_volume, sizeof(Volume<3, f32>));
  cudaMalloc(&volume.data, volume.size());
  cudaMemcpy(volume.data, offset_image_data_1.data(), volume.size(), cudaMemcpyDefault);
  cudaMemcpy(device_volume, &volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  reduce_no_mem<8, REGISTRATION::VolumeDifference<f32>, f32>(
    volume.elements(),
    &difference,
    device_volume,
    device_volume
  );

  EXPECT_FLOAT_EQ(difference, 0.0f);

  cudaFree(volume.data);
  cudaFree(device_volume);
}

TEST(REGISTRATION, IMAGE_DIFFERENCE_RESPECT_SIGNS) {
  Volume<3, f32> volume_1 {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };
  Volume<3, f32> volume_2 {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  Volume<3, f32>* device_volume_1 = nullptr;
  Volume<3, f32>* device_volume_2 = nullptr;

  cudaMalloc(&volume_1.data, volume_1.size());
  cudaMalloc(&volume_2.data, volume_2.size());
  cudaMemcpy(volume_1.data, overlap_image_1.data(), volume_1.size(), cudaMemcpyDefault);
  cudaMemcpy(volume_2.data, overlap_image_2.data(), volume_1.size(), cudaMemcpyDefault);
  cudaMalloc(&device_volume_1, sizeof(Volume<3, f32>));
  cudaMalloc(&device_volume_2, sizeof(Volume<3, f32>));
  cudaMemcpy(device_volume_1, &volume_1, sizeof(Volume<3, f32>), cudaMemcpyDefault);
  cudaMemcpy(device_volume_2, &volume_2, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  f32 difference = -1.0f;

  reduce_no_mem<8, REGISTRATION::VolumeDifference<f32>, f32>(
    volume_1.elements(),
    &difference,
    device_volume_1,
    device_volume_2
  );

  // So you have 2 3x3x3 cubes, one with 1 at (1,1,1) and another with -1 at (1,3,1)
  // None overlapping pixels adds 1 per pixel and there's 18 per cube for a total of 36.
  // overlapping add 4 per pixel with 9 pixels = 36.
  // 36 + 36 = 72

  EXPECT_EQ(difference, 72.0f);

  cudaFree(device_volume_1);
  cudaFree(device_volume_2);
  cudaFree(volume_1.data);
  cudaFree(volume_2.data);
}


TEST(REGISTRATION, CENTER_OF_GRAVITY_MOVES_A_CUBE) {
  // need to copy the data to a non-const area

  Volume<3, f32> volume_1 {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  Volume<3, f32> volume_2 {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  cudaMalloc(&volume_1.data, volume_1.size());
  cudaMalloc(&volume_2.data, volume_2.size());
  cudaMemcpy(volume_1.data, offset_image_data_1.data(), volume_1.size(), cudaMemcpyDefault);
  cudaMemcpy(volume_2.data, offset_image_data_2.data(), volume_2.size(), cudaMemcpyDefault);

  Image image_1{image_space, volume_1};
  Image image_2{image_space, volume_2};

  REGISTRATION::register_to(image_1, image_2);

  cudaFree(volume_1.data);
  cudaFree(volume_2.data);
}

TEST(REGISTRATION, INTERPOLATING_IDENTITY_TRANSLATION_IDENTITY_SPACE) {
  Volume<3, f32> image_volume {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  Volume<3, f32> out_volume {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  cudaMalloc(&image_volume.data, image_volume.size());
  cudaMalloc(&out_volume.data, out_volume.size());
  cudaMemcpy(image_volume.data, offset_image_data_1.data(), image_volume.size(), cudaMemcpyDefault);

  Image image_1{image_space, image_volume};

  DicomNodeRunner runner;

  REGISTRATION::interpolate_for_registration(
    runner,
    image_1,
    {},
    image_space,
    out_volume
  );

  std::array<f32, z * y * x> host_out;
  cudaMemcpy(host_out.data(), out_volume.data, out_volume.size(), cudaMemcpyDefault);

  Volume<3, f32>* device_image_volume = nullptr;
  cudaMalloc(&device_image_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_image_volume, &image_1.volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  Volume<3, f32>* device_out_volume = nullptr;
  cudaMalloc(&device_out_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_out_volume, &out_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  f32 error = NAN;
  REGISTRATION::volume_difference_device(
    image_volume.elements(),
    device_image_volume,
    device_out_volume,
    error
  );

  EXPECT_EQ(0.0f, error);

  cudaFree(image_volume.data);
  cudaFree(out_volume.data);
  cudaFree(device_out_volume);
  cudaFree(device_image_volume);
}


TEST(REGISTRATION, INTERPOLATING_X_TRANSLATION_IDENTITY_SPACE) {

  Volume<3, f32> image_volume {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  Volume<3, f32> out_volume {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  cudaMalloc(&image_volume.data, image_volume.size());
  cudaMalloc(&out_volume.data, out_volume.size());
  cudaMemcpy(image_volume.data, offset_image_data_1.data(), image_volume.size(), cudaMemcpyDefault);

  const Image image_1{image_space, image_volume};

  DicomNodeRunner runner;

  REGISTRATION::interpolate_for_registration(
    runner,
    image_1,
    { // I think I should make some test cases, that highlight XYZ and ZYX difference
      .translations = { 1, 0, 0 }
    },
    image_space,
    out_volume
  );

  // I'm having some issues with nvcc here and comptime
  cuda::std::array<f32, elements> expected_array = create_blank_array<f32, elements>();

  constexpr ArrayDataBlock<f32, 3> translateBy1Block = {
    .value = 1.0f,
    .bounds = {
      Bound{ .lower = 2, .upper = 5},
      Bound{ .lower = 1, .upper = 4},
      Bound{ .lower = 1, .upper = 4},
    }
  };

  apply_bound(expected_array, Extent<3>{z,y,x}, translateBy1Block);

  Volume<3, f32> expected_volume = {
    .m_extent = Extent<3>{z,y,x},
    .default_value = 0.0f
  };

  cudaMalloc(&expected_volume.data, expected_volume.size());
  cudaMemcpy(expected_volume.data, expected_array.data(), expected_volume.size(), cudaMemcpyDefault);

  Volume<3, f32>* device_expected_volume = nullptr;
  cudaMalloc(&device_expected_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_expected_volume, &expected_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  Volume<3, f32>* device_out_volume = nullptr;
  cudaMalloc(&device_out_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_out_volume, &out_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  f32 error = NAN;
  REGISTRATION::volume_difference_device(
    image_volume.elements(),
    device_expected_volume,
    device_out_volume,
    error
  );

  EXPECT_EQ(0.0f, error);

  cudaFree(image_volume.data);
  cudaFree(out_volume.data);
  cudaFree(expected_volume.data);
  cudaFree(device_out_volume);
  cudaFree(device_expected_volume);
}

TEST(REGISTRATION, OPTIMIZER_CAN_OPTIMIZE_WITH_TRANSLATION) {
  constexpr Extent<3> e{8,8,8};

  auto source_data = create_blank_array<f32, e.elements()>();
  apply_bound(source_data, e, ArrayDataBlock<f32, 3>{
    .value = 1.0f, .bounds=cuda::std::array{
      Bound{2,5},
      Bound{2,5},
      Bound{2,5}
  }});

  auto target_data = create_blank_array<f32, e.elements()>();
  apply_bound(target_data, e, ArrayDataBlock<f32, 3>{
    .value = 1.0f, .bounds=cuda::std::array{
      Bound{3,6},
      Bound{1,4},
      Bound{2,5}
  }});

  Volume<3, f32> source_volume {
    .data = nullptr,
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> target_volume {
    .data = nullptr,
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> temp_volume {
    .data = nullptr,
    .m_extent=e,
    .default_value = 0.0f
  };

  EXPECT_EQ(cudaMalloc(&source_volume.data, source_volume.size()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&target_volume.data, target_volume.size()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&temp_volume.data, temp_volume.size()), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(source_volume.data, source_data.data(), source_volume.size(), cudaMemcpyDefault ), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(target_volume.data, target_data.data(), target_volume.size(), cudaMemcpyDefault ), cudaSuccess);

  Volume<3, f32>* device_source_volume = nullptr;
  Volume<3, f32>* device_temp_volume = nullptr;

  EXPECT_EQ(cudaMalloc(&device_source_volume, sizeof(Volume<3, f32>)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&device_temp_volume, sizeof(Volume<3, f32>)), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(device_source_volume, &source_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault ), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(device_temp_volume, &temp_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault ), cudaSuccess);

  Image<3, f32> source_image {image_space, source_volume};
  Image<3, f32> target_image {image_space, source_volume};

  DicomNodeRunner runner;

  auto cost_function = [&](const REGISTRATION::OptimizerParam<f32>& params) {
    f32 error = cuda::std::numeric_limits<f32>::max();
    runner | [&]() {
      return REGISTRATION::interpolate_for_registration(
        target_image,
        params.affine,
        source_image.space,
      temp_volume
      );
    } | [&](){return REGISTRATION::volume_difference_device(
      e.elements(),
      device_source_volume,
      device_temp_volume,
      error
      );};

    return error;
  };

  REGISTRATION::Optimizer<f32> optimizer(cost_function);

  optimizer.max_step_size = REGISTRATION::ImageTranslation{
    .scale = 1.0f,
    .translations = Point<3>(1.0f,1.0f,1.0f),
    .rotations = cuda::std::array{1.0f,1.0f,1.0f}
  };

  std::printf("Current Cost: %f\n", optimizer.best_params.cost);

  optimizer.update_translation(cost_function);

  std::printf("Current Cost: %f\n", optimizer.best_params.cost);

  EXPECT_EQ(cudaFree(source_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(target_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(temp_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(device_source_volume), cudaSuccess);
  EXPECT_EQ(cudaFree(device_temp_volume), cudaSuccess);
}
