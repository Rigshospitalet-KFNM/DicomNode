#pragma once

#include<iostream>

#include<gtest/gtest.h>

#include"../python_interpolation.cpp"

TEST(INTERPOLATION, INTERPOLATE_AT_POINT){
  constexpr size_t z = 3;
  constexpr size_t y = 4;
  constexpr size_t x = 3;
  constexpr size_t points = x * y * z;

  f32 data[ z * y * x ] = {
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,
    310.0f, 130.0f, 240.0f,
    320.0f, 150.0f, 270.0f,

    110.0f, 130.0f, 140.0f,
    120.0f, 150.0f, 170.0f,
    10.0f, 30.0f, 40.0f,
    20.0f, 50.0f, 70.0f,

    -10.0f, 160.0f, -40.0f,
    -20.0f, 150.0f, -720.0f,
    -5.0f, 350.0f, 40.0f,
    -50.0f, -150.0f, -720.0f,
  };

  Space<3> local_space;

  local_space.basis = SquareMatrix<3>{
    .points={
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
    }
  };

  local_space.inverted_basis = SquareMatrix<3>{
    .points={
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
    }
  };

  local_space.starting_point = Point<3>{
    0.0f,0.0f,0.0f
  };

  local_space.extent = Extent<3>{z,y,x};
  Space<3> out_space = local_space;

  f32 *out = new f32[points];

  interpolation<f32, 3>(data, out, local_space, out_space);

  for(size_t idx = 0; idx < points; idx++){
    EXPECT_EQ(out[idx], data[idx]);
  }

  delete[] out;
}

constexpr size_t z = 4;
constexpr size_t y = 4;
constexpr size_t x = 4;

constexpr size_t data_elements = z * y * x;

constexpr size_t out_z = 3;
constexpr size_t out_y = 3;
constexpr size_t out_x = 3;


TEST(INTERPOLATION, INTERPOLATE_UINT8){
  using test_type = u8;

  constexpr u32 x = 10;
  constexpr u32 y = 10;
  constexpr u32 z = 10;

  constexpr u32 ux = x - 1;
  constexpr u32 uy = y - 1;
  constexpr u32 uz = z - 1;


  test_type* data = new test_type[ x * y * z ];
  test_type* result = new test_type[ x * y * z ];


  for(int i = 0; i < x * y * z; i++){
    const uint32_t lx = i % x;
    const uint32_t ly = (i / x) % y;
    const uint32_t lz = i / (x * y);

    data[i] = static_cast<test_type>(std::max(std::max(lx, ly), lz) + 1);
  }

  const Space<3> source_space {
    .starting_point = Point<3>{
      0.0f, 0.0f, 0.0f
    },

    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
   .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
    .extent = Extent<3>{z,y,x}
  };

  const Space<3> target_space {
    .starting_point = Point<3>{
     1.0f, 1.0f, 1.0f
    },
    .basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0, 0.0f,
        0.0f, 0.0f, 1.0f
      }
    },
   .inverted_basis = SquareMatrix<3>{
      .points={
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
      },
    },
    .extent = Extent<3>{uz, uy, ux}
  };

  interpolation<test_type, 3>(data, result, source_space, target_space);

  for(u32 lz = 0; lz < uz; lz++){
    u32 dlz = lz + 1;
    for(u32 ly = 0; ly < uy; ly++){
      u32 dly = ly + 1;
      for(u32 lx = 0; lx < ux; lx++){
        u32 dlx = lx + 1;
        u32 result_index = lz * uy * ux + ly * ux + lx;
        u32 data_index = dlz * y * x + dly * x + dlx;
        EXPECT_EQ(result[result_index], data[data_index]);
      }
    }
  }

  delete[] data;
  delete[] result;
}