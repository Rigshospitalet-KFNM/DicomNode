#include<array>
#include<cuda/std/array>

#include<gtest/gtest.h>

#include "../gpu_code/center_of_gravity.cuh"

namespace CENTER_OF_GRAVITY_TESTS {
  TEST(CENTER_OF_GRAVITY_TEST, INDEX_FUNCTION_2x3x4) {
    constexpr static Extent<3> extent{2,3,4};

    EXPECT_EQ(extent.x(), 4);
    EXPECT_EQ(extent.y(), 3);
    EXPECT_EQ(extent.z(), 2);

    constexpr static std::array<u32, extent.elements()> expected_x {
      0,0,0,0,
      0,0,0,0,
      0,0,0,0,

      1,1,1,1,
      1,1,1,1,
      1,1,1,1
    };

    constexpr static std::array<u32, extent.elements()> expected_y {
      0,0,0,0,
      1,1,1,1,
      2,2,2,2,

      0,0,0,0,
      1,1,1,1,
      2,2,2,2
    };

    constexpr static std::array<u32, extent.elements()> expected_z {
      0,1,2,3,
      0,1,2,3,
      0,1,2,3,

      0,1,2,3,
      0,1,2,3,
      0,1,2,3
    };

    auto actual_x = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32 {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::X>::index(i, extent);
    });

    auto actual_y = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32 {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::Y>::index(i, extent);
    });

    auto actual_z = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32  {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::Z>::index(i, extent);
    });

    for (u64 i = 0; i < extent.elements(); ++i) {
      EXPECT_EQ(expected_x[i], actual_x[i]);
      EXPECT_EQ(expected_y[i], actual_y[i]);
      EXPECT_EQ(expected_z[i], actual_z[i]);
    }
  }

  TEST(CENTER_OF_GRAVITY_TEST, INDEX_FUNCTION_4x3x2) {
    constexpr static Extent<3> extent{4,3,2};

    EXPECT_EQ(extent.x(), 2);
    EXPECT_EQ(extent.y(), 3);
    EXPECT_EQ(extent.z(), 4);

    std::array<u32, extent.elements()> expected_x {
      0,0,
      0,0,
      0,0,

      1,1,
      1,1,
      1,1,

      2,2,
      2,2,
      2,2,

      3,3,
      3,3,
      3,3
    };

    std::array<u32, extent.elements()> expected_y {
      0,0,
      1,1,
      2,2,

      0,0,
      1,1,
      2,2,

      0,0,
      1,1,
      2,2,

      0,0,
      1,1,
      2,2
    };

    std::array<u32, extent.elements()> expected_z {
      0,1,
      0,1,
      0,1,

      0,1,
      0,1,
      0,1,

      0,1,
      0,1,
      0,1,

      0,1,
      0,1,
      0,1
    };

    auto actual_x = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32 {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::X>::index(i, extent);
    });

    auto actual_y = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32 {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::Y>::index(i, extent);
    });

    auto actual_z = initialize_array<extent.elements()>([&](u64 i) constexpr noexcept -> u32  {
      return CENTER_OF_GRAVITY::COG_REDUCE_FUNCTION<f32, DIMENSION::Z>::index(i, extent);
    });

    for (u64 i = 0; i < extent.elements(); ++i) {
      EXPECT_EQ(expected_x[i], actual_x[i]);
      EXPECT_EQ(expected_y[i], actual_y[i]);
      EXPECT_EQ(expected_z[i], actual_z[i]);
    }
  }

  TEST(CENTER_OF_GRAVITY_TEST, FIND_SYMETRIC_CENTER){
    constexpr Extent<3> extent{4,4,4};

    std::array<u32, extent.elements()> data {
        1,0,0,1,
        0,0,0,0,
        0,0,0,0,
        1,0,0,1,

        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,

        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,

        1,0,0,1,
        0,0,0,0,
        0,0,0,0,
        1,0,0,1
    };

    Volume<3, u32> vol {
      .data = nullptr,
      .m_extent = extent,
      .default_value = 0
    };



  }

}