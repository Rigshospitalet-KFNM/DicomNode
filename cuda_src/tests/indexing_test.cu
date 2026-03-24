/**
 * @file indexing_test.cu
 * @author Demiguard (cjen0668@regionh.dk)
 * @brief Testing for gpu_code/core/indexing
 * @version 0.1
 * @date 2024-12-18
 *
 * @copyright Copyright (c) 2024
 *
 */

#include<array>
#include<iostream>
#include<gtest/gtest.h>
#include<cuda/std/optional>

#include"../gpu_code/dicom_node_gpu.cuh"

namespace TEST_INDEXING {


  TEST(INDEXING, CREATION_TEST_MULTIPLE_ARGS){
  const Extent<3> extent(3,4,5);

  EXPECT_EQ(extent.x(), 5);
  EXPECT_EQ(extent.y(), 4);
  EXPECT_EQ(extent.z(), 3);

  const Index<3> index(1,2,3);

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 3);
}

TEST(INDEXING, CREATION_TEST_LIST){
  const Extent<3> extent({3,4,5});

  EXPECT_EQ(extent.x(), 5);
  EXPECT_EQ(extent.y(), 4);
  EXPECT_EQ(extent.z(), 3);

  const Index<3> index({1,2,3});

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 3);
}

TEST(INDEXING, INDEXING_INTO_DATA_STRUCTURES){
  const Extent<3> extent({3,4,5});

  EXPECT_EQ(extent.x(), extent[2]);
  EXPECT_EQ(extent.y(), extent[1]);
  EXPECT_EQ(extent.z(), extent[0]);

  const Index<3> index({1,2,3});

  EXPECT_EQ(index.x(), index[0]);
  EXPECT_EQ(index.y(), index[1]);
  EXPECT_EQ(index.z(), index[2]);
}

TEST(INDEXING, CONTAINS_TAKES_REVERSED_INDEXES_INTO_ACCOUNT){
  const Extent<3> extent(3,4,5);

  const Index<3> index_in(4,3,2);
  const Index<3> index_out(2,3,4);

  EXPECT_TRUE(extent.contains(index_in));
  EXPECT_FALSE(extent.contains(index_out));
}

TEST(INDEXING, TO_FLAT_INDEX){
  const Extent<3> extent(3,4,5);

  cuda::std::optional<uint64_t> flat_index = extent.flat_index(1,1,2);

  EXPECT_TRUE(flat_index.has_value());
  if(flat_index.has_value()){
    const uint64_t& flat_value = flat_index.value();
    EXPECT_EQ(flat_value,(uint64_t)(2*4*5 + 1 * 5 + 1));
  }
}

TEST(INDEXING, FROM_FLAT_INDEX){
  const Extent<3> extent(3,4,5);

  Index<3> index = extent.from_flat_index(1 + 2 * 5 + 2 * 4 * 5);

  EXPECT_EQ(index.x(), 1);
  EXPECT_EQ(index.y(), 2);
  EXPECT_EQ(index.z(), 2);

}


TEST(INDEXING, CREATE_DIMENSIONAL_INDEXES){
  Index<3> dimensional_index_1 = dimensional_offset<3>(0);
  Index<3> dimensional_index_2 = dimensional_offset<3>(1);
  Index<3> dimensional_index_3 = dimensional_offset<3>(2);
  Index<3> dimensional_index_4 = dimensional_offset<3>(3);
  Index<3> dimensional_index_5 = dimensional_offset<3>(4);
  Index<3> dimensional_index_6 = dimensional_offset<3>(5);
  Index<3> dimensional_index_7 = dimensional_offset<3>(6);
  Index<3> dimensional_index_8 = dimensional_offset<3>(7);

  Index<3> expected_dimensional_index_1 = Index{{0,0,0}};
  Index<3> expected_dimensional_index_2 = Index{{1,0,0}};
  Index<3> expected_dimensional_index_3 = Index{{0,1,0}};
  Index<3> expected_dimensional_index_4 = Index{{1,1,0}};
  Index<3> expected_dimensional_index_5 = Index{{0,0,1}};
  Index<3> expected_dimensional_index_6 = Index{{1,0,1}};
  Index<3> expected_dimensional_index_7 = Index{{0,1,1}};
  Index<3> expected_dimensional_index_8 = Index{{1,1,1}};

  EXPECT_EQ(dimensional_index_1, expected_dimensional_index_1);
  EXPECT_EQ(dimensional_index_2, expected_dimensional_index_2);
  EXPECT_EQ(dimensional_index_3, expected_dimensional_index_3);
  EXPECT_EQ(dimensional_index_4, expected_dimensional_index_4);
  EXPECT_EQ(dimensional_index_5, expected_dimensional_index_5);
  EXPECT_EQ(dimensional_index_6, expected_dimensional_index_6);
  EXPECT_EQ(dimensional_index_7, expected_dimensional_index_7);
  EXPECT_EQ(dimensional_index_8, expected_dimensional_index_8);
}

constexpr Extent<3> extent{4,4,4};
TEST(INDEXING, INDEXING_INTO_VOLUME){
  std::array<f32, extent.elements()> data;

  u64 i = 0;
  for( f32& d : data){
    d = static_cast<f32>(++i);
  }

  Volume<3, f32> volume {
    .data=data.data(),
    .m_extent = extent,
    .default_value = 0.0f
  };

  Index<3> index_at_begin{0,0,0};
  Index<3> index_at_end{3,3,3};

  EXPECT_EQ(volume.at(index_at_begin), 1.0f);
  EXPECT_EQ(volume.at(index_at_end), 64.0f);
}

TEST(INDEXING, INTERPOLATING_INTO_A_VOLUME){
  std::array<f32, extent.elements()> data;

  u64 i = 0;
  for( f32& d : data){
    d = static_cast<f32>(++i);
  }

  Volume<3, f32> volume {
    .data=data.data(),
    .m_extent = extent,
    .default_value = 0.0f
  };

  Point<3> index_at_begin{0.5f,0.0f,0.0f};
  EXPECT_FLOAT_EQ(volume.interpolate_at_index_point(index_at_begin), 1.5f);
}




}