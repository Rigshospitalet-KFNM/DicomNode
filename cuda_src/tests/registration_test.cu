//
// Created by cjen0668 on 7/30/26.
//

#include "../gpu_code/core/core.cuh"
#include "../gpu_code/registration.cuh"
#include "test_helper.cuh"

#include <array>
#include <memory>

#include <gtest/gtest.h>

#include "test_helper.cuh"


constexpr u64 x = 8;
constexpr u64 y = 8;
constexpr u64 z = 8;

constexpr u64 elements = x * y * z;
// This is just a 3x3x3 cube at 2 different indexes.
// I should create some functions for easy array creation, but i'm feeling that i'm missing c++23/c++26 features :(

constexpr Extent<3> extent8{8,8,8};
constexpr Extent<3> extent64{64,64,64};
constexpr Extent<3> extent512{512,512,512};

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
};

// End of test declaration
TEST(REGISTRATION, VOLUME_SUBTRACT_OF_SAME_IS_ZERO) {
  Volume<3, f32> volume {
    .m_extent = extent8,
    .default_value = 0.0f
  };

  Volume<3, f32>* device_volume;


  cudaMalloc(&device_volume, sizeof(Volume<3, f32>));
  cudaMalloc(&volume.data, volume.size());
  cudaMemcpy(volume.data, offset_image_data_1.data(), volume.size(), cudaMemcpyDefault);
  cudaMemcpy(device_volume, &volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  f32 difference = NAN;
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

  Image<3, f32> image_1{image_space, volume_1};
  Image<3, f32> image_2{image_space, volume_2};

  REGISTRATION::register_to(image_1, image_2);

  cudaFree(volume_1.data);
  cudaFree(volume_2.data);
}

TEST(REGISTRATION, INTERPOLATING_IDENTITY_TRANSLATION_IDENTITY_SPACE) {
  Image<3, f32> source_image{
    .space=image_space,
    .volume={
      .m_extent = extent8,
      .default_value = 0.0f
    }
  };
  Image<3, f32> target_image{
    .space=image_space,
    .volume={
      .m_extent = extent8,
      .default_value = 0.0f
  }};

  cudaMalloc(&source_image.data(), source_image.size());
  cudaMalloc(&target_image.data(), target_image.size());
  cudaMemcpy(source_image.data(), offset_image_data_1.data(), source_image.size(), cudaMemcpyDefault);
  
  DicomNodeRunner runner;

  REGISTRATION::interpolate_for_registration(
    runner,
    source_image,
    {},
    target_image,
    target_image.volume
  );

  std::array<f32, extent8.elements()> host_out;
  cudaMemcpy(host_out.data(), target_image.data(), target_image.size(), cudaMemcpyDefault);

  Volume<3, f32>* device_image_volume = nullptr;
  cudaMalloc(&device_image_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_image_volume, &source_image.volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  Volume<3, f32>* device_out_volume = nullptr;
  cudaMalloc(&device_out_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_out_volume, &target_image.volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  f32 error = NAN;
  REGISTRATION::volume_difference_device(
    source_image.elements(),
    device_image_volume,
    device_out_volume,
    error
  );

  EXPECT_EQ(0.0f, error);

  cudaFree(source_image.data());
  cudaFree(target_image.data());
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

  const Image<3, f32> image_1{image_space, image_volume};
  const Image<3, f32> image_2{image_space, out_volume};

  DicomNodeRunner runner;

  REGISTRATION::interpolate_for_registration(
    runner,
    image_1,
    { // I think I should make some test cases, that highlight XYZ and ZYX difference
      .translations = { 1, 0, 0 }
    },
    image_2,
    out_volume
  );

  // I'm having some issues with nvcc here and comptime
  cuda::std::array<f32, elements> expected_array = create_blank_array<f32, elements>();

  constexpr ArrayDataBlock<f32, 3> translateBy1Block = {
    .bounds = {
      Bound{ .lower = 2, .upper = 5},
      Bound{ .lower = 1, .upper = 4},
      Bound{ .lower = 1, .upper = 4},
    },
    .value = 1.0f
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

TEST(REGISTRATION, INTERPOLATING_FOR_REGISTRATION_TRANSLATION) {
  const auto source_data_ptr = std::make_unique<cuda::std::array<f32, extent64.elements()>>();
  auto& source_data = *source_data_ptr;

  for (auto& p : source_data) { p = 0.0f; }

  apply_bound(source_data, extent64, ArrayDataBlock<f32, 3>{
    .bounds=cuda::std::array{
      Bound{.lower = 1, .upper = 31},
      Bound{.lower = 1, .upper = 31},
      Bound{.lower = 1, .upper = 31}
    }, .value = 1.0f}
  );

  Image<3, f32> image_source{
      .space = image_space,
      .volume = {.m_extent = extent64, .default_value = 0.0f}};

  Image<3, f32> interpolated_image{
      .space = image_space,
      .volume = {.m_extent = extent64, .default_value = 0.0f}};

  cudaMallocManaged(&image_source.data(), image_source.size());
  cudaMallocManaged(&interpolated_image.data(), interpolated_image.size());
  cudaMemcpy(image_source.data(), source_data.data(), image_source.size(), cudaMemcpyDefault);


  DicomNodeRunner runner;

  REGISTRATION::interpolate_for_registration(
    runner,
    image_source,
    { // I think I should make some test cases, that highlight XYZ and ZYX difference
      .translations = { 29, 29, 29 }
    },
    interpolated_image,
    interpolated_image.volume
  );

  Volume<3, f32> expected_volume = {
    .m_extent = extent64,
    .default_value = 0.0f
  };


  const auto expected_data_ptr = std::make_unique<cuda::std::array<f32, extent64.elements()>>();
  auto& expected_data = *expected_data_ptr;
  for (auto& p : expected_data) { p = 0.0f; }


  apply_bound(expected_data, extent64, ArrayDataBlock<f32, 3>{
  .bounds=cuda::std::array{
    Bound{.lower = 30, .upper = 60},
    Bound{.lower = 30, .upper = 60},
    Bound{.lower = 30, .upper = 60}
  }, .value = 1.0f });


  cudaMalloc(&expected_volume.data, expected_volume.size());
  cudaMemcpy(expected_volume.data, expected_data.data(), expected_volume.size(), cudaMemcpyDefault);

  Volume<3, f32>* device_expected_volume = nullptr;
  cudaMalloc(&device_expected_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_expected_volume, &expected_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);

  Volume<3, f32>* device_interpolated_volume = nullptr;
  cudaMalloc(&device_interpolated_volume, sizeof(Volume<3, f32>));
  cudaMemcpy(device_interpolated_volume, &interpolated_image.volume, sizeof(Volume<3, f32>), cudaMemcpyDefault);


  f32 error = NAN;
  REGISTRATION::volume_difference_device(
    interpolated_image.elements(),
    device_expected_volume,
    device_interpolated_volume,
    error
  );

  EXPECT_EQ(0.0f, error);

  cudaFree(image_source.data());
  cudaFree(interpolated_image.data());
  cudaFree(expected_volume.data);
  cudaFree(device_interpolated_volume);
  cudaFree(device_expected_volume);
}


TEST(REGISTRATION, DATA_BLOCK_IS_DIFERENT_THAN_0) {
  constexpr Extent<3> e{8,8,8};
  constexpr auto zeroes = create_blank_array<f32, e.elements()>();
  auto source_data = create_blank_array<f32, e.elements()>();
  apply_bound(source_data, e, ArrayDataBlock<f32, 3>{
     .bounds=cuda::std::array{
      Bound{2,5},
      Bound{2,5},
      Bound{2,5}
    }, .value = 3.0f}
  );

  Volume<3, f32> source_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> target_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> temp_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32>* device_source_volume = nullptr;
  Volume<3, f32>* device_temp_volume = nullptr;

  constexpr auto volume_size = sizeof(Volume<3, f32>);

  DicomNodeRunner runner;

  runner |
    [&]() {
      return cudaMalloc(&source_volume.data, source_volume.size());
    } | [&]() {
      return cudaMalloc(&target_volume.data, target_volume.size());
    } | [&]() {
      return cudaMalloc(&temp_volume.data, temp_volume.size());
    } | [&]() {
      return cudaMalloc(&device_source_volume, volume_size);
    } | [&]() {
      return cudaMalloc(&device_temp_volume, volume_size);
    } | [&]() {
      return cudaMemcpy(source_volume.data, source_data.data(), source_volume.size(), cudaMemcpyDefault);
    } | [&]() {
      return cudaMemcpy(target_volume.data, zeroes.data(), target_volume.size(), cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_source_volume, &source_volume, volume_size, cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_temp_volume, &temp_volume, volume_size, cudaMemcpyDefault);
    };

  const Image<3, f32> source_image = {
    image_space,
    source_volume
  };

  const Image<3, f32> target_image = {
    image_space,
    target_volume
  };

  f32 cost = cuda::std::numeric_limits<f32>::max();

  REGISTRATION::cost_function<f32>(
    runner,
    source_image,
    target_image, temp_volume,
    device_source_volume,
    device_temp_volume,
    REGISTRATION::OptimizerParam<f32> {},
  cost
  );

  EXPECT_EQ(runner.error(), SUCCESS);

  EXPECT_FLOAT_EQ(cost, 243.0f); // ELEMENTS(3 * 3 * 3) * VALUE(abs(3 - 0) ** 2)

  cudaFree(device_source_volume);
  cudaFree(device_temp_volume);
  cudaFree(source_volume.data);
  cudaFree(target_volume.data);
  cudaFree(temp_volume.data);
}

TEST(REGISTRATION, DATA_BLOCK_TRANSLATION) {
  constexpr Extent<3> e{8,8,8};
  auto source_data = create_blank_array<f32, e.elements()>();
  apply_bound(source_data, e, ArrayDataBlock<f32, 3>{
    .bounds=cuda::std::array{
      Bound{2,5},
      Bound{2,5},
      Bound{2,5}
    }, .value = 1.0f }
  );
  auto target_data = create_blank_array<f32, e.elements()>();
  apply_bound(target_data, e, ArrayDataBlock<f32, 3>{
    .bounds=cuda::std::array{
    Bound{3,6},
    Bound{1,4},
    Bound{2,5}
  }, .value = 1.0f});


  Volume<3, f32> source_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> target_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32> temp_volume = {
    .m_extent = e,
    .default_value = 0.0f
  };

  Volume<3, f32>* device_source_volume = nullptr;
  Volume<3, f32>* device_temp_volume = nullptr;

  constexpr auto volume_size = sizeof(Volume<3, f32>);

  DicomNodeRunner runner;

  runner |
    [&]() {
      return cudaMalloc(&source_volume.data, source_volume.size());
    } | [&]() {
      return cudaMalloc(&target_volume.data, target_volume.size());
    } | [&]() {
      return cudaMalloc(&temp_volume.data, temp_volume.size());
    } | [&]() {
      return cudaMalloc(&device_source_volume, volume_size);
    } | [&]() {
      return cudaMalloc(&device_temp_volume, volume_size);
    } | [&]() {
      return cudaMemcpy(source_volume.data, source_data.data(), source_volume.size(), cudaMemcpyDefault);
    } | [&]() {
      return cudaMemcpy(target_volume.data, target_data.data(), target_volume.size(), cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_source_volume, &source_volume, volume_size, cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_temp_volume, &temp_volume, volume_size, cudaMemcpyDefault);
    };

  const Image<3, f32> source_image = {
    image_space,
    source_volume
  };

  const Image<3, f32> target_image = {
    image_space,
    target_volume
  };

  f32 cost = cuda::std::numeric_limits<f32>::max();

  REGISTRATION::cost_function<f32>(
    runner,
    source_image,
    target_image,
    temp_volume,
    device_source_volume,
    device_temp_volume,
    REGISTRATION::OptimizerParam<f32> {},
  cost
  );

  EXPECT_EQ(runner.error(), SUCCESS);

  EXPECT_FLOAT_EQ(cost, 30.0f);

  cudaFree(device_source_volume);
  cudaFree(device_temp_volume);
  cudaFree(source_volume.data);
  cudaFree(target_volume.data);
  cudaFree(temp_volume.data);
}

TEST(REGISTRATION, centering_space) {
  constexpr Point<3> s{100.0f,100.0f,100.0f};


  SquareMatrix<3> basis = {
    1.0f, 0.0f, 6.0f,
    1.0f, 2.0f, -0.5f,
    0.0f, 3.0f, 1.0f
  };

  Space<3> space {
    .starting_point = s,
    .basis = basis,
    .inverted_basis = basis.inverse(),
  };

  const auto r = space.center_space(extent512);

  constexpr Point<3> zeros{
    0.0f, 0.0f, 0.0f
  };

  EXPECT_EQ(space.at_index(extent512.central_point()), zeros);
}

TEST(REGISTRATION, centering_space_with_reset) {
  constexpr Point<3> s{100.0f,100.0f,100.0f};
  constexpr Extent<3> e{512,512,512};

  SquareMatrix<3> basis = {
    1.0f, 0.0f, 6.0f,
    1.0f, 2.0f, -0.5f,
    0.0f, 3.0f, 1.0f
  };

  Space<3> space {
    .starting_point = s,
    .basis = basis,
    .inverted_basis = basis.inverse(),
  };

  auto space_copy = space;
  const auto r = space.center_space(e);

  space.starting_point += r;

  EXPECT_EQ(space.starting_point, space_copy.starting_point);
}

TEST(REGISTRATION, OPTIMIZER_CAN_OPTIMIZE_WITH_TRANSLATION) {
  constexpr Extent<3> e{8,8,8};

  auto source_data = create_blank_array<f32, e.elements()>();
  apply_bound(source_data, e, ArrayDataBlock<f32, 3>{
    .bounds=cuda::std::array{
      Bound{2,5},
      Bound{2,5},
      Bound{2,5}
  }, .value = 1.0f,});

  auto target_data = create_blank_array<f32, e.elements()>();
  apply_bound(target_data, e, ArrayDataBlock<f32, 3>{
    .bounds=cuda::std::array{
      Bound{3,6},
      Bound{1,4},
      Bound{2,5}
  }, .value = 1.0f,});

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

  Volume<3, f32>* device_source_volume = nullptr;
  Volume<3, f32>* device_temp_volume = nullptr;


  EXPECT_EQ(cudaMalloc(&source_volume.data, source_volume.size()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&target_volume.data, target_volume.size()), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&temp_volume.data, temp_volume.size()), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(source_volume.data, source_data.data(), source_volume.size(), cudaMemcpyDefault ), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(target_volume.data, target_data.data(), target_volume.size(), cudaMemcpyDefault ), cudaSuccess);


  EXPECT_EQ(cudaMalloc(&device_source_volume, sizeof(Volume<3, f32>)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&device_temp_volume, sizeof(Volume<3, f32>)), cudaSuccess);

  EXPECT_EQ(cudaMemcpy(device_source_volume, &source_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault ), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(device_temp_volume, &temp_volume, sizeof(Volume<3, f32>), cudaMemcpyDefault ), cudaSuccess);

  Image<3, f32> source_image {image_space, source_volume};
  Image<3, f32> target_image {image_space, target_volume};

  DicomNodeRunner runner;

  auto cost_function = [&](const REGISTRATION::OptimizerParam<f32>& params) {
    f32 cost = cuda::std::numeric_limits<f32>::max();
    runner | [&]() {

      return REGISTRATION::interpolate_for_registration(
        target_image,
        params.affine,
        source_image,
      temp_volume
      );
    } | [&](){return REGISTRATION::volume_difference_device(
      e.elements(),
      device_source_volume,
      device_temp_volume,
      cost
      );};

    return cost;
  };

  REGISTRATION::Optimizer<f32> optimizer(cost_function);

  EXPECT_FLOAT_EQ(optimizer.best_params.cost, 30.0f);

  optimizer.step_size = REGISTRATION::ImageTranslation{
    .scale = 1.0f,
    .translations = Point<3>(1.0f,1.0f,1.0f),
    .rotations = cuda::std::array{1.0f,1.0f,1.0f}
  };

  optimizer.update_translation(cost_function);

  EXPECT_FLOAT_EQ(optimizer.best_params.affine.translations[0], -1.0f);
  EXPECT_FLOAT_EQ(optimizer.best_params.affine.translations[1],  1.0f);
  EXPECT_FLOAT_EQ(optimizer.best_params.affine.translations[2],  0.0f);

  EXPECT_FLOAT_EQ(optimizer.best_params.cost, 0.0f);

  EXPECT_EQ(cudaFree(source_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(target_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(temp_volume.data), cudaSuccess);
  EXPECT_EQ(cudaFree(device_source_volume), cudaSuccess);
  EXPECT_EQ(cudaFree(device_temp_volume), cudaSuccess);
}

TEST(REGISTRATION, REGISTRATION_FULL_EXAMPLE_TRANSLATION_ONLY) {
  const auto source_data_ptr = std::make_unique<cuda::std::array<f32, extent64.elements()>>();
  auto& source_data = *source_data_ptr;

  apply_bound(source_data, extent64, ArrayDataBlock<f32, 3>{
     .bounds=cuda::std::array{
      Bound{1,32},
      Bound{1,32},
      Bound{1,32}
    }, .value = 1.0f}
  );
  auto target_data = create_blank_array<f32, extent64.elements()>();
  apply_bound(target_data, extent64, ArrayDataBlock<f32, 3>{
  .bounds=cuda::std::array{
    Bound{29,60},
    Bound{29,60},
    Bound{29,60}
  }, .value = 1.0f});


  Volume<3, f32> source_volume = {
    .m_extent = extent64,
    .default_value = 0.0f
  };

  Volume<3, f32> target_volume = {
    .m_extent = extent64,
    .default_value = 0.0f
  };

  Volume<3, f32> temp_volume = {
    .m_extent = extent64,
    .default_value = 0.0f
  };

  Volume<3, f32>* device_source_volume = nullptr;
  Volume<3, f32>* device_temp_volume = nullptr;

  constexpr auto volume_size = sizeof(Volume<3, f32>);

  DicomNodeRunner runner;

  runner |
    [&]() {
      return cudaMalloc(&source_volume.data, source_volume.size());
    } | [&]() {
      return cudaMalloc(&target_volume.data, target_volume.size());
    } | [&]() {
      return cudaMalloc(&temp_volume.data, temp_volume.size());
    } | [&]() {
      return cudaMalloc(&device_source_volume, volume_size);
    } | [&]() {
      return cudaMalloc(&device_temp_volume, volume_size);
    } | [&]() {
      return cudaMemcpy(source_volume.data, source_data.data(), source_volume.size(), cudaMemcpyDefault);
    } | [&]() {
      return cudaMemcpy(target_volume.data, target_data.data(), target_volume.size(), cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_source_volume, &source_volume, volume_size, cudaMemcpyDefault);
    } | [&](){
      return cudaMemcpy(device_temp_volume, &temp_volume, volume_size, cudaMemcpyDefault);
    };

  constexpr static Space<3> space {
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
  };

  const Image<3, f32> source_image = {
    space,
    source_volume
  };

  const Image<3, f32> target_image = {
    space,
    target_volume
  };

  const auto cost_function = [&](REGISTRATION::OptimizerParam<f32>& params) -> f32 {
    f32 cost = cuda::std::numeric_limits<f32>::max();
    runner | [&]() {
      return REGISTRATION::interpolate_for_registration(
        target_image,
        params.affine,
        source_image,
      temp_volume
      );
    } | [&](){return REGISTRATION::volume_difference_device(
      extent64.elements(),
      device_source_volume,
      device_temp_volume,
      cost
      );};

    return cost;
  };

  REGISTRATION::Optimizer<f32> optimizer(cost_function);

  /*
  optimizer.step_size = REGISTRATION::ImageTranslation{
    .scale = 1.0f,
    .translations = Point<3>(1.0f,1.0f,1.0f),
    .rotations = cuda::std::array{0.0f,0.0f,0.0f}
  };
  */

  std::printf("Cost: %f\n", optimizer.cost());

  while (runner.error() == SUCCESS && optimizer.should_continue()) {
    optimizer.optimize(cost_function);
    std::printf("Cost at: %f\n translations: (%f,%f,%f)\n",
      optimizer.best_params.cost,
      optimizer.best_params.affine.translations[0],
      optimizer.best_params.affine.translations[1],
      optimizer.best_params.affine.translations[2]
    );

  }

  std::printf("Ran Iterations: %i\n", optimizer.iterations);
  std::printf("Cost: %f\n", optimizer.cost());

  cudaFree(device_source_volume);
  cudaFree(device_temp_volume);
  cudaFree(source_volume.data);
  cudaFree(target_volume.data);
  cudaFree(temp_volume.data);
}
